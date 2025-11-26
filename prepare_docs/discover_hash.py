# C:\dev\GovernEdge_CLI\prepare_docs\discover_hash.py

"""
📄 Purpose:
-----------
Stage 1 of the ingestion pipeline.
- Walk source folders under DOCS_DIR.
- Compute file-level SHA-256 hashes.
- Extract JSON front-matter (doc_title, keyword_tags, codes, dates, etc.).
- UPSERT into SQLite (doc_ingest).
- Decide which files are *changed* (new hash) and need further processing.

This script does NOT:
- Split into sections or chunks (that's chunker.py).
- Clean with spaCy (that's spacy_cleaner.py).
- Embed or vectorize (that's builder scripts).
"""

import json
import sqlite3
import hashlib
import logging
from typing import Tuple, Optional, Iterable, List, Dict
from pathlib import Path
from datetime import datetime

from prepare_docs.frontmatter_canon import parse_front_matter, fm_title_or_filename
from prepare_docs.keyword_hunter import ensure_frontmatter_llm_enriched, compute_freshness_score
from prepare_docs.db_io import ensure_db

# ---------------- logging ----------------
logger = logging.getLogger(__name__)

# ---------------- config -----------------
DB_PATH  = Path(r"C:\dev\GovernEdge_CLI\database_tst\chat_logs.sqlite")
DOCS_DIR = Path(r"C:\dev\GovernEdge_CLI\data_tst\sap_docs")  # Root folder of corpus

# ---------------- hashing ----------------
def sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()


def sha256_text(s: str) -> str:
    return sha256_bytes(s.encode("utf-8", "ignore"))


def compute_file_hash(file_path: str | Path) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------- helpers ----------------
def get_existing_by_path(cur: sqlite3.Cursor, file_path: str) -> Optional[Tuple[int, str]]:
    cur.execute("SELECT doc_id, file_hash FROM doc_ingest WHERE file_path = ?", (file_path,))
    return cur.fetchone()


UPSERT_SQL = """
INSERT INTO doc_ingest
  (file_name, file_path, folder_resource,
   file_size_bytes, file_mtime_ns,
   file_hash, fm_hash,
   title, fm_json,
   date_published, date_previous_published,
   date_download, date_freshness_score,
   date_ingested_iso)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(file_path) DO UPDATE SET
    folder_resource         = excluded.folder_resource,
    file_size_bytes         = excluded.file_size_bytes,
    file_mtime_ns           = excluded.file_mtime_ns,
    file_hash               = excluded.file_hash,
    fm_hash                 = excluded.fm_hash,
    title                   = excluded.title,
    fm_json                 = excluded.fm_json,
    date_published          = excluded.date_published,
    date_previous_published = excluded.date_previous_published,
    date_download           = excluded.date_download,
    date_freshness_score    = excluded.date_freshness_score,
    date_ingested_iso       = excluded.date_ingested_iso,
    updated_at              = CURRENT_TIMESTAMP;
"""

def upsert_file(cur: sqlite3.Cursor, file_path: str | Path, docs_root: Path) -> Tuple[int, bool]:
    file_path = Path(file_path)
    file_name = file_path.name

    try:
        folder_resource = str(file_path.parent.relative_to(docs_root)).replace("\\", "/")
    except Exception:
        folder_resource = ""

    # stat info
    try:
        st = file_path.stat()
        file_size_bytes = int(st.st_size)
        file_mtime_ns   = int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9)))
    except OSError:
        file_size_bytes = None
        file_mtime_ns   = None

    # detect change via full file hash
    new_hash = compute_file_hash(file_path)
    existing = get_existing_by_path(cur, str(file_path))
    if existing and existing[1] == new_hash:
        logger.debug("Unchanged file: %s", file_path)
        return existing[0], False

    ext = file_path.suffix.lower()
    fm: dict = {}
    eff_title = file_path.stem.replace("_", " ").title()
    fm_json = "{}"
    fm_hash: Optional[str] = None
    date_published: Optional[str] = None
    date_previous_published: Optional[str] = None
    date_download: Optional[str] = None
    date_freshness_score: float = 0.0

    # front-matter (texty files only; binaries handled at file level)
    if ext in (".md", ".txt"):
        try:
            raw = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            logger.warning("Failed to read text from %s: %s", file_path, e)
            raw = ""

        fm, _body = parse_front_matter(raw)
        if not isinstance(fm, dict):
            fm = {}

        # nice title from fm (doc_title / title / filename)
        eff_title = fm_title_or_filename(fm, file_name)

        # dates from FM (all optional)
        date_published = (fm.get("date_published") or "").strip() or None
        date_previous_published = (
            fm.get("date_previous_published")
            or fm.get("date_previous_release")
            or fm.get("date_previous")
            or ""
        ).strip() or None
        date_download = (
            fm.get("date_download")
            or fm.get("date_download_iso")
            or ""
        ).strip() or None

        # freshness score based on dates we actually have
        date_freshness_score = compute_freshness_score(
            date_published=date_published,
            date_previous_published=date_previous_published,
        )

        # canonical FM JSON + hash
        fm_json = json.dumps(fm, ensure_ascii=False)
        fm_hash = sha256_text(fm_json) if fm_json else None

    elif ext == ".csv":
        fm = {}
        fm_json = "{}"
        fm_hash = None
        date_freshness_score = 0.0
    else:
        # binaries or other formats: no FM here, body will be handled downstream if needed
        fm = {}
        fm_json = "{}"
        fm_hash = None
        date_freshness_score = 0.0

    # ingestion timestamp (today) – this is for the doc_ingest row itself
    date_ingested_iso = datetime.now().date().isoformat()

    cur.execute(
        UPSERT_SQL,
        (
            file_name,
            str(file_path),
            folder_resource,
            file_size_bytes,
            file_mtime_ns,
            new_hash,
            fm_hash,
            eff_title,
            fm_json,
            date_published,
            date_previous_published,
            date_download,
            date_freshness_score,
            date_ingested_iso,
        ),
    )

    cur.execute("SELECT doc_id FROM doc_ingest WHERE file_path = ?", (str(file_path),))
    row = cur.fetchone()
    doc_id = int(row[0]) if row else -1

    logger.info("Upserted file: %s (doc_id=%s, changed=True)", file_path, doc_id)
    return doc_id, True

# ---------------- walker ----------------
SKIP_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".pdf", ".zip", ".exe", ".dll"}

def iter_docs(root: Path):
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.name.startswith(("~", ".")):
            continue
        if p.suffix.lower() in SKIP_EXTS:
            continue
        yield p

# ---------------- orchestrator wrapper ----------------
def discover_changed_files(
    conn: sqlite3.Connection,
    src_dirs,
    exts=None,
    include_unchanged: bool = False
) -> List[Dict]:
    cur = conn.cursor()
    exts_set = {e.lower().lstrip(".") for e in (exts or [])}
    seen: set[str] = set()
    changed_rows: list[dict] = []

    for root in map(Path, src_dirs):
        if not root.exists():
            logger.warning("Source dir not found: %s", root)
            continue

        for path in iter_docs(root):
            if exts_set and path.suffix.lower().lstrip(".") not in exts_set:
                continue
            spath = str(path.resolve())
            if spath in seen:
                continue
            seen.add(spath)

            # Let the enrichment pass update JSON frontmatter on disk first
            ensure_frontmatter_llm_enriched(path)

            doc_id, did_change = upsert_file(cur, path, root)
            if did_change or include_unchanged:
                changed_rows.append(
                    {
                        "doc_id": doc_id,
                        "file_path": spath,
                        "changed": bool(did_change),
                    }
                )

    conn.commit()
    logger.info(
        "Discovered %d files (%d changed/new)",
        len(changed_rows),
        sum(r["changed"] for r in changed_rows),
    )
    return changed_rows

if __name__ == "__main__":
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        ensure_db(conn)
        rows = discover_changed_files(conn, [DOCS_DIR], exts=None, include_unchanged=False)
        logger.info(
            "✅ Scanned %d tracked file(s); upserted %d changed/new rows.",
            len(rows),
            sum(r["changed"] for r in rows),
        )
