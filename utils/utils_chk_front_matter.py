# What it checks / fixes
# C:\dev\GovernEdge_CLI\utils\utils_chk_front_matter.py

# Detects missing or malformed frontmatter fences.
# Flags tabs, trailing spaces in FM keys, and empty keys (e.g., tcodes: → coerces to []).
# Ensures lists for action, tags, tcodes; normalizes list indent to 2 spaces.

# Normalizes values:
# action: lowers, strips leading action_, collapses spaces to underscores.
# category, object: lowers, replaces spaces/hyphens with underscores.
# tags: coerces to simple strings; trims whitespace.
# Rewrites a clean YAML block with --- … --- and reattaches original body unchanged.

# dry-run audit (no changes)
# python llm_core_tst\utils_tst\utils_chk_front_matter.py --root C:\dev\GovernEdge_CLI\data_tst\sap_docs

# write fixes in-place with .bak backups
# python utils_chk_front_matter.py --root C:\dev\GovernEdge_CLI\data_tst\sap_docsv --fix

# restrict to markdown only (default scans .md + .txt)
# python utils_chk_front_matter.py --root ./data_tst\sap_docs --md-only --fix


"""
Audit and (optionally) fix YAML frontmatter blocks in markdown/text files.

Checks performed:
  - Presence of frontmatter block
  - Trailing spaces, tabs, and parse errors
  - Correct list coercion (action/tags/tcodes)
  - Canonical key/value normalization (snake_case, strip)
  - Indentation consistency (2 spaces for lists)

Outputs:
  - Console summary
  - JSON report (_fm_audit_report.json) in root folder
"""

import argparse, re, os, sys, json, shutil, logging
from pathlib import Path

try:
    import yaml
except ImportError:
    print("Please `pip install pyyaml`")
    sys.exit(1)

# ---------------- logging ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("fm_audit")

# ---------------- constants ----------------
FM_SPLIT_RE = re.compile(r"^\ufeff?---\s*\n(.*?)\n(?:---|\.\.\.)\s*\n(.*)\Z", re.S)

LIST_KEYS = {"action", "tags", "tcodes"}   # keys expected to be lists
REWRITE_ORDER = ["title", "title2", "category", "object", "action", "tags", "tcodes"]

# ---------------- helpers ----------------
def _as_list(x):
    """Coerce scalars/None into list form."""
    if x is None: return []
    if isinstance(x, list): return x
    return [x]

def _norm_key(k: str) -> str:
    return str(k).strip()

def _norm_val_str(s: str) -> str:
    return str(s).strip()

def _to_snake(s: str) -> str:
    """Normalize string to snake_case."""
    return re.sub(r"[\s\-]+", "_", str(s).strip().lower())

def _norm_action_item(s: str) -> str:
    """Canonicalize action list items."""
    s = _to_snake(s)
    if s.startswith("action_"):  # strip prefix
        s = s.split("action_", 1)[1]
    return s

def _load_frontmatter(text: str):
    """Extract raw YAML block + body from a file."""
    m = FM_SPLIT_RE.match(text.replace("\r\n", "\n").replace("\r", "\n"))
    if not m:
        return None, text  # no FM present
    return m.group(1), m.group(2)

def _dump_yaml_block(d: dict) -> str:
    """
    Serialize YAML with:
      - Preferred key order
      - Block list style
      - 2-space indent
    """
    class LiteralList(list): pass
    def litlist_representer(dumper, data):
        return dumper.represent_list(list(data))
    yaml.add_representer(LiteralList, litlist_representer)

    out = {}
    for k in REWRITE_ORDER:
        if k in d: out[k] = d[k]
    for k, v in d.items():
        if k not in out: out[k] = v

    return yaml.safe_dump(out, sort_keys=False, allow_unicode=True)

def _audit_frontmatter_block(raw_fm: str, file: Path) -> tuple[dict, list]:
    """
    Parse and normalize a frontmatter block.

    Returns:
        cleaned (dict): normalized keys/values
        issues (list): diagnostics found
    """
    issues = []

    # quick lint checks
    if "\t" in raw_fm:
        issues.append("tabs found in frontmatter")
    for ln in raw_fm.splitlines():
        if ln.strip().startswith("#"):
            continue
        if ":" in ln:
            left = ln.split(":", 1)[0]
            if left.endswith(" "):
                issues.append(f"trailing space before colon in key: {left!r}")
                break

    # YAML parse
    try:
        fm = yaml.safe_load(raw_fm) or {}
        if not isinstance(fm, dict):
            issues.append("frontmatter is not a mapping")
            fm = {}
    except Exception as e:
        issues.append(f"YAML parse error: {e}")
        return {}, issues

    # Normalize
    cleaned = {}
    for k, v in fm.items():
        nk = _norm_key(k)
        if nk in LIST_KEYS:
            vv = _as_list(v)
            if nk == "action":
                vv = [_norm_action_item(x) for x in vv if str(x).strip()]
            elif nk in ("tags", "tcodes"):
                vv = [_norm_val_str(x) for x in vv if str(x).strip()]
            cleaned[nk] = vv
            if isinstance(v, (str, int, float)):
                issues.append(f"{nk} coerced to list")
        else:
            if nk in ("category", "object"):
                cleaned[nk] = _to_snake(v)
            elif nk in ("title", "title2"):
                cleaned[nk] = _norm_val_str(v)
            else:
                cleaned[nk] = v

    # Guarantee presence of tcodes
    if "tcodes" not in cleaned or cleaned.get("tcodes") is None:
        cleaned["tcodes"] = []
        issues.append("tcodes missing/empty -> set to []")

    # Force list type if still wrong
    for lk in LIST_KEYS:
        if not isinstance(cleaned.get(lk, []), list):
            cleaned[lk] = _as_list(cleaned.get(lk))
            issues.append(f"{lk} forced to list")

    # Indentation check (heuristic)
    for key in ("action", "tags", "tcodes"):
        pat = re.compile(rf"^\s*{key}\s*:\s*\n((?:[ \t]*-\s+.*\n)+)", re.M)
        m = pat.search(raw_fm + ("\n" if not raw_fm.endswith("\n") else ""))
        if m:
            bad = False
            for item_ln in m.group(1).splitlines():
                if not item_ln.strip():
                    continue
                lead = len(item_ln) - len(item_ln.lstrip(" "))
                if lead not in (2,):  # expect 2 spaces
                    bad = True; break
            if bad:
                issues.append(f"inconsistent indent under {key} -> will rewrite with 2 spaces")

    return cleaned, issues

def process_file(p: Path, fix: bool, md_only: bool) -> dict:
    """
    Process a single file: audit and optionally fix frontmatter.
    Returns dict with keys: file, has_frontmatter, issues, changed.
    """
    text = p.read_text(encoding="utf-8", errors="ignore")
    raw_fm, body = _load_frontmatter(text)

    report = {
        "file": str(p),
        "has_frontmatter": bool(raw_fm),
        "issues": [],
        "changed": False
    }

    if not raw_fm:
        if p.suffix.lower() == ".md":
            report["issues"].append("missing frontmatter block")
        return report

    cleaned, issues = _audit_frontmatter_block(raw_fm, p)
    report["issues"].extend(issues)

    new_fm_text = _dump_yaml_block(cleaned).rstrip() + "\n"
    new_text = f"---\n{new_fm_text}---\n{body}"

    if new_text != text:
        report["changed"] = True
        if fix:
            bak = p.with_suffix(p.suffix + ".bak")
            if not bak.exists():
                shutil.copyfile(p, bak)
                logger.info("Backup created: %s", bak)
            p.write_text(new_text, encoding="utf-8")
            logger.info("Rewrote file: %s", p)

    return report

# ---------------- CLI ----------------
def main():
    ap = argparse.ArgumentParser(description="Audit YAML frontmatter across a folder of markdown/text files.")
    ap.add_argument("--root", required=True, help="Root folder to scan")
    ap.add_argument("--fix", action="store_true", help="Rewrite frontmatter in-place (creates .bak)")
    ap.add_argument("--md-only", action="store_true", help="Only scan .md files")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        logger.error("❌ Not found: %s", root)
        sys.exit(1)

    exts = [".md"] if args.md_only else [".md", ".txt"]
    results = []
    for path in root.rglob("*"):
        if not path.is_file(): 
            continue
        if path.suffix.lower() not in exts: 
            continue
        try:
            results.append(process_file(path, fix=args.fix, md_only=args.md_only))
        except Exception as e:
            logger.error("Failed on %s: %s", path, e)
            results.append({"file": str(path), "has_frontmatter": None, "issues": [f"error: {e}"], "changed": False})

    # Summary
    offenders = [r for r in results if r["issues"]]
    changed = sum(1 for r in results if r["changed"])
    missing = [r for r in results if "missing frontmatter block" in r["issues"]]

    logger.info("=== Frontmatter Audit Summary ===")
    logger.info("Scanned files     : %d", len(results))
    logger.info("Files with issues : %d", len(offenders))
    logger.info("Files auto-fixed  : %d", changed if args.fix else 0)
    logger.info("Missing FM (md)   : %d", len(missing))

    # Report details
    if offenders:
        logger.info("--- Offenders (top 25) ---")
        for r in offenders[:25]:
            logger.info("* %s", r["file"])
            for iss in r["issues"]:
                logger.info("    - %s", iss)

    # JSON report
    out_json = root / "_fm_audit_report.json"
    try:
        out_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("📝 Full report: %s", out_json)
    except Exception as e:
        logger.warning("⚠️ Could not write report JSON: %s", e)

if __name__ == "__main__":
    main()
