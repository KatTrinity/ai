# C:\dev\GovernEdge_CLI\prepare_docs\frontmatter_canon.py

"""
📄 Purpose:
-----------
This helper handles **JSON front-matter parsing** for Markdown/text docs.

Supported front-matter:

- JSON front-matter between `:::json` and `:::`, e.g.

  :::json
  {
    "doc_id": "3491591",
    "doc_title": "FAQ: Stock consistency...",
    "keyword_tags": ["stock", "serial"],
    "codes": ["LO-MD-SN-2CL"],
    "version": 1,
    "version_previous": 0,
    "date_published": "2024-11-27",
    "date_previous_release": "",
    "date_download": "2025-11-18"
  }
  :::

Typical flow:
- Normalize Unicode + newlines so hashes are stable.
- Detect and strip JSON front-matter block.
- Parse JSON safely.
- Normalize tags into list[str] and store as JSON strings.
- Return (fm_dict, body_without_fm).

Used by:
- discover_hash.py during ingestion (to extract title/tags, then store fm_json).
- Later stages can trust that fm_json in the DB is valid JSON (dict).
"""

from __future__ import annotations

import re
import unicodedata
import argparse
import json
import sqlite3
import logging
import hashlib
from pathlib import Path
from typing import Tuple, Dict, List, Any, Optional

# ---------------- logging ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("frontmatter_canon")

# ---------------- regex ----------------
# JSON FM: optional BOM; block delimited by :::json ... :::
JSON_FM_RE = re.compile(
    r"^\ufeff?:::json\s*\n(.*?)\n:::\s*\n?(.*)\Z",
    re.S,
)

# ---------------- normalization helpers ----------------
def _norm_newlines(s: str) -> str:
    """Unicode NFC + CRLF/CR -> LF."""
    return unicodedata.normalize("NFC", s).replace("\r\n", "\n").replace("\r", "\n")


def _sha256_text(s: str) -> str:
    h = hashlib.sha256()
    h.update(_norm_newlines(s).encode("utf-8", "ignore"))
    return h.hexdigest()


# ---------------- list helpers ----------------
def _clean_list(val: Any) -> list[str]:
    """Normalize lists: drop Nones/empties/whitespace; strings -> single-item list if nonempty."""
    if val is None:
        return []
    if isinstance(val, list):
        out: list[str] = []
        for x in val:
            s = "" if x is None else str(x).strip()
            if s:
                out.append(s)
        return out
    if isinstance(val, str):
        s = val.strip()
        return [s] if s else []
    return []


def _as_lower_list(x: Any) -> list[str]:
    vals = _clean_list(x)
    return [v.lower() for v in vals]


def _as_upper_list(x: Any) -> list[str]:
    vals = _clean_list(x)
    return [v.upper() for v in vals]


def _to_int_or_none(x: Any) -> Optional[int]:
    if x is None or x == "":
        return None
    try:
        return int(x)
    except Exception:
        return None


def _to_str_or_none(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip()
    return s or None


# ---------------- FM helpers ---------------
def ensure_title_and_h1(
    meta: dict,
    body_text: str,
    src_path: str,
) -> tuple[str, str]:
    """
    Return (doc_title, body_with_h1).

    Rules:
    - If the first non-empty line is an H1 (`# `), use that as title.
    - Otherwise:
        * Use meta["title"] if present, else the file stem,
        * Prepend a `# title` line to the body.
    """

    meta = meta or {}
    body = (body_text or "").replace("\r\n", "\n").replace("\r", "\n")

    lines = body.splitlines()
    first_non_empty_idx = None
    for i, ln in enumerate(lines):
        if ln.strip():
            first_non_empty_idx = i
            break

    # Case 1: body already starts with an H1
    if first_non_empty_idx is not None:
        first = lines[first_non_empty_idx]
        if first.lstrip().startswith("# "):
            h1 = first.lstrip()[2:].strip()
            title = h1 if h1 else meta.get("title") or Path(src_path).stem
            return title, body

    # Case 2: no H1 → synthesize one
    fm_title = (meta.get("title") or "").strip()
    if not fm_title:
        fm_title = Path(src_path).stem

    h1_line = f"# {fm_title}\n\n"

    if body.strip():
        new_body = h1_line + body
    else:
        new_body = h1_line

    return fm_title, new_body

def _is_trivial_fm(fm: dict) -> bool:
    """
    Return True if FM is effectively empty/boilerplate.

    For the new JSON FM, having any of these non-empty makes it non-trivial:
      - doc_title / title
      - keyword_tags
      - codes
      - version / version_previous
      - date_published / date_download
    """
    if not isinstance(fm, dict) or not fm:
        return True

    doc_title = (fm.get("doc_title") or "").strip()
    title     = (fm.get("title") or "").strip()
    kw        = _clean_list(fm.get("keyword_tags"))
    codes     = _clean_list(fm.get("codes"))
    ver       = fm.get("version")
    ver_prev  = fm.get("version_previous")
    dp        = (fm.get("date_published") or "").strip()
    dd        = (fm.get("date_download") or fm.get("date_download_iso") or "").strip()

    if doc_title or title or kw or codes or ver or ver_prev or dp or dd:
        return False

    # otherwise treat as trivial
    return True


def canonicalize_fm_fields(fm: dict) -> dict:
    """
    Canonicalize FM into columns we track in doc_ingest.

    We care about:
      - keyword_tags  -> JSON string (lowercase)
      - codes         -> JSON string (UPPER)
      - version, version_previous (ints or NULL)
      - date_published, date_previous_published, date_download (TEXT or NULL)
    """
    import json as _json

    if not isinstance(fm, dict):
        fm = {}

    kw_raw    = fm.get("keyword_tags") or []
    codes_raw = fm.get("codes") or []

    version             = _to_int_or_none(fm.get("version"))
    version_previous    = _to_int_or_none(fm.get("version_previous"))

    date_published           = _to_str_or_none(fm.get("date_published"))
    date_prev_pub_candidate  = (
        fm.get("date_previous_published")
        or fm.get("date_previous_release")
        or fm.get("date_previous")  # extra safety
    )
    date_previous_published  = _to_str_or_none(date_prev_pub_candidate)
    date_download_candidate  = fm.get("date_download") or fm.get("date_download_iso")
    date_download            = _to_str_or_none(date_download_candidate)

    return dict(
        keyword_tags=_json.dumps(_as_lower_list(kw_raw)),
        codes=_json.dumps(_as_upper_list(codes_raw)),
        version=version,
        version_previous=version_previous,
        date_published=date_published,
        date_previous_published=date_previous_published,
        date_download=date_download,
    )


# ---------------- FM parsing ----------------
def parse_front_matter(raw_text: str) -> Tuple[dict, str]:
    """
    Returns (fm_dict, body_without_fm).

    Only JSON front-matter is supported:

        :::json
        { ... }
        :::

    If FM is trivial/boilerplate, returns ({}, body_without_fm).
    """
    if not raw_text:
        return {}, ""

    raw_text = _norm_newlines(raw_text)

    # 1) JSON front-matter
    jm = JSON_FM_RE.match(raw_text)
    if jm:
        fm_src, body = jm.group(1), jm.group(2)
        fm: dict = {}
        try:
            fm = json.loads(fm_src) or {}
        except Exception as e:
            logger.warning("JSON front-matter parse failed: %s", e)
            fm = {}
        if _is_trivial_fm(fm):
            return {}, body
        return fm, body

    # No front-matter block found
    return {}, raw_text


def strip_front_matter_text(text: str) -> str:
    """Convenience: return body only (ignore fm_dict)."""
    _, body = parse_front_matter(text)
    return body


# ---------------- noindex stripping + body canonicalization ----------------
_START_RE  = re.compile(r'<!--\s*noindex(?:\s*:\s*start)?\s*-->', re.I)
_END_RE    = re.compile(r'<!--\s*(?:/noindex|noindex\s*:\s*end)\s*-->', re.I)
_MARKER_RE = re.compile(r'<!--\s*(?:/noindex|noindex(?:\s*:\s*(?:start|end))?)\s*-->', re.I)


def strip_noindex_blocks(text: str) -> Tuple[str, int]:
    """
    Remove any content wrapped by noindex markers.
    - If a start appears without an end, drop to EOF.
    - If an end appears without a start, delete the marker.
    Returns (clean_text, removed_flag).
    """
    if not text:
        return "", 0

    s = _norm_newlines(text)
    out: list[str] = []
    i = 0
    removed = 0
    while True:
        sm = _START_RE.search(s, i)
        if not sm:
            out.append(s[i:])
            break
        out.append(s[i:sm.start()])
        em = _END_RE.search(s, sm.end())
        if em:
            removed = 1
            i = em.end()
        else:
            removed = 1
            i = len(s)
            break
    cleaned = "".join(out)
    cleaned = _MARKER_RE.sub("", cleaned)
    return cleaned, removed


def canonicalize_body(raw_text: str) -> dict:
    txt = _norm_newlines(raw_text or "")
    txt, noidx = strip_noindex_blocks(txt)
    wc = len([t for t in txt.split() if t.strip()])
    return dict(
        body_md=txt,
        body_word_count=wc,
        noindex_removed=int(noidx),
        body_hash=_sha256_text(txt),
    )


# ---------------- DB write ----------------
def process_file_frontmatter(conn: sqlite3.Connection, doc_id: int, file_path: str | Path) -> None:
    file_path = Path(file_path)
    raw = file_path.read_text(encoding="utf-8", errors="ignore")

    # Parse front matter
    fm, body = parse_front_matter(raw)
    fm = fm if isinstance(fm, dict) else {}
    fm_json = json.dumps(fm, ensure_ascii=False)

    # 🔧 Ensure:
    #   - doc title derived from H1 or FM or filename
    #   - body gets an injected H1 if missing
    title, body_with_h1 = ensure_title_and_h1(
        meta=fm,
        body_text=body,
        src_path=str(file_path),
    )

    # Canonicalize body (with H1 injection applied)
    body_fields = canonicalize_body(body_with_h1)

    # Check if front matter is meaningful
    if not _is_trivial_fm(fm):
        fields = canonicalize_fm_fields(fm)
        conn.execute(
            """
            UPDATE doc_ingest
            SET title=?,
                keyword_tags=?,
                codes=?,
                version=?,
                version_previous=?,
                date_published=?,
                date_previous_published=?,
                date_download=?,
                fm_json=?,
                body_md=?,
                body_word_count=?,
                noindex_removed=?,
                body_hash=?,
                updated_at=CURRENT_TIMESTAMP
            WHERE doc_id=?;
            """,
            (
                title,
                fields["keyword_tags"],
                fields["codes"],
                fields["version"],
                fields["version_previous"],
                fields["date_published"],
                fields["date_previous_published"],
                fields["date_download"],
                fm_json,
                body_fields["body_md"],
                body_fields["body_word_count"],
                body_fields["noindex_removed"],
                body_fields["body_hash"],
                doc_id,
            ),
        )
    else:
        # FM trivial → leave title alone, update body + json
        conn.execute(
            """
            UPDATE doc_ingest
            SET fm_json=?,
                body_md=?,
                body_word_count=?,
                noindex_removed=?,
                body_hash=?,
                updated_at=CURRENT_TIMESTAMP
            WHERE doc_id=?;
            """,
            (
                fm_json,
                body_fields["body_md"],
                body_fields["body_word_count"],
                body_fields["noindex_removed"],
                body_fields["body_hash"],
                doc_id,
            ),
        )

    # Mirror into doc_canonical
    conn.execute(
        """
        INSERT INTO doc_canonical (doc_id, canonical_body, updated_at)
        VALUES (?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(doc_id) DO UPDATE SET
            canonical_body = excluded.canonical_body,
            updated_at     = CURRENT_TIMESTAMP;
        """,
        (doc_id, body_fields["body_md"]),
    )


# ---------------- test helpers ----------------
def _has_frontmatter(text: str) -> bool:
    s = _norm_newlines(text)
    return (
        s.startswith(":::json")
        or s.startswith("\ufeff:::json")
    )


def test_strip(paths: List[Path]) -> None:
    for p in paths:
        raw = p.read_text(encoding="utf-8", errors="ignore")
        _, body = parse_front_matter(raw)
        assert len(body) > 0, f"Empty body after strip: {p}"
        if _has_frontmatter(raw):
            assert not _has_frontmatter(body), f"Front-matter not stripped: {p}"
        assert body == body.lstrip("\n"), f"Leading newlines remain in: {p}"


# ---------------- CLI entry ----------------
SAMPLES = [
    r"C:\dev\GovernEdge_CLI\data_tst\sap_docs\mat_mngt\foo.md",
    r"C:\dev\GovernEdge_CLI\data_tst\sap_docs\sales_distr\bar.md",
]

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Front-matter parser & tester")
    ap.add_argument("--test", nargs="*", help="Run strip tests on given files (or built-in samples if none provided).")
    ap.add_argument("--file", help="Parse a single file and print FM + preview of body.")
    args = ap.parse_args()

    try:
        if args.test is not None:
            paths = [Path(p) for p in (args.test or SAMPLES)]
            test_strip(paths)
            logger.info("✅ frontmatter_canon tests passed.")

        if args.file:
            p = Path(args.file)
            raw = p.read_text(encoding="utf-8", errors="ignore")
            fm, body = parse_front_matter(raw)
            logger.info("Parsed front-matter for %s", p)
            print("— FM —")
            print(json.dumps(fm, indent=2, ensure_ascii=False))
            print("\n— BODY (preview) —")
            print(canonicalize_body(body)["body_md"][:200].replace("\n", "\\n"))

        if args.test is None and not args.file:
            logger.info("Tip: use --test (optional paths) or --file PATH")

    except AssertionError as e:
        logger.error("❌ Test failed: %s", e)
        raise
