
# C:\dev\GovernEdge_CLI\prepare_docs\ingest_docs.py
from __future__ import annotations
from pathlib import Path
import sys

# Add repo root to sys.path universally (…/GovernEdge_CLI)
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse, sqlite3, time, json
from contextlib import closing
from config_base.config import Config 


# ── Helpers you wrote in steps 1–6 ─────────────────────────────────────────
# Adjust these imports to match your actual module paths if needed.
from prepare_docs.discover_hash import discover_changed_files
from prepare_docs.db_io import get_conn
from prepare_docs.frontmatter_canon import process_file_frontmatter  # -> parses FM, stores fm_json & canonical body path/bytes
from prepare_docs.fts_mirror import mirror_from_db   
from prepare_docs.db_io import ensure_db


from chunker import rebuild_chunks_for_doc                                   # -> writes doc_chunks rows with chunk_hash_raw 
#from taxonomies.apply_taxonomy import tag_chunks_for_doc                          # -> updates chunk metadata: action/object/category

# spaCy cache step can be called as a function; keep as a module import
from prepare_docs.spacy_clean import spacy_main                              # -> exposes main(db_path, nlp_version, keep_pos


# === 🔧 Config ===
DB_PATH  = Path(getattr(Config, "DB_PATH", "database/chat_logs.sqlite"))
MASTER_DB_PATH  = Path(getattr(Config, "MASTER_DB_PATH", "database/master_data.duckdb"))
DOCS_DIR = Path(getattr(Config, "DOCS_DIR", "data_tst/sap_docs"))

# ── Tiny timers ────────────────────────────────────────────────────────────
class Timer:
    def __init__(self): self.t0 = time.time()
    def ms(self): return int((time.time() - self.t0) * 1000) 

# ── Orchestration pieces ───────────────────────────────────────────────────
def ensure_base_schema(conn: sqlite3.Connection) -> None:
    """
    Delegates to your db_io.ensure_db() which should create:
      # 1) CORE (parents → children)
        _exec_block(conn, "CORE", DDL_CORE)

        # 2) CHAT (everything EXCEPT log_chat_sources)
        _exec_block(conn, "CHAT_BASE", DDL_CHAT_BASE)

        # 3) FTS last (it references doc_chunks)
        _exec_block(conn, "FTS", DDL_FTS)

        # 4) Finally, create the child table that FK-references parents
        _exec_block(conn, "CHAT_SOURCES", DDL_CHAT_SOURCES)

        # 4) Finally, create the child table that FK-references parents
        _exec_block(conn, "CHAT_VIEWS", DDL_VIEWS)
    """
    ensure_db(conn)  # make sure FTS is present early

def step1_discover(conn: sqlite3.Connection, src_dirs: list[str], exts: set[str]) -> list[dict]:
    """
    Walks source folders, computes file hashes, UPSERTs doc_ingest, and returns changed/new rows.
    discover_changed_files(conn, src_dirs, exts) -> [{"doc_id": int, "file_path": str, "changed": bool}, ...]
    """
    changed = discover_changed_files(conn, src_dirs, exts)
    conn.commit()
    return [r for r in changed if r.get("changed")]

def step2_frontmatter(conn: sqlite3.Connection, changed_docs: list[dict]) -> None:
    """
    For each changed doc: parse front-matter, normalize body,
    update title/fm_json in doc_ingest, and upsert canonical body into doc_canonical.
    """
    for r in changed_docs: 
        process_file_frontmatter(conn, r["doc_id"], r["file_path"])
    conn.commit()

def step3_chunk(conn: sqlite3.Connection, changed_docs: list[dict], target_char_window: int, overlap: int) -> None:
    """
    For each changed doc: (re)build sections + micro-chunks, assign stable IDs, compute chunk_hash_raw.
    rebuild_chunks_for_doc(conn, doc_id, char_window=400, overlap=60) -> None
    """
    for r in changed_docs:
        rebuild_chunks_for_doc(conn, r["doc_id"], char_window=target_char_window, overlap=overlap)
    conn.commit()

#def step4_taxonomy(conn: sqlite3.Connection, changed_docs: list[dict]) -> None:
    #"""
    #Apply taxonomy scalars on raw chunks (action/object/category); FM hints override.
    #tag_chunks_for_doc(conn, doc_id) -> None
    #"""
    #for r in changed_docs:
        #tag_chunks_for_doc(conn, r["doc_id"])
    #conn.commit()

def step5_spacy_cache(db_path: str, nlp_version: str, keep_pos: bool) -> tuple[int, int]:
    """
    Compute cleaned text per chunk, keyed by (chunk_id, nlp_version) with change detection via text_hash.
    Uses spacy_clean.main(db_path, nlp_version, keep_pos) which prints its own summary.
    Returns (updated, skipped) parsed from stdout if you want; here we just run inline.
    """
    # We’re calling the module’s entrypoint directly (no subprocess).
    spacy_main(db_path, nlp_version, keep_pos=keep_pos)
    # spacy_cleaner already prints stats; you can enhance it to return counts if desired.
    return (-1, -1)

def step6_fts(conn: sqlite3.Connection, doc_ids_filter: list[int] | None) -> tuple[int, int]:
    """
    Mirror raw chunks into FTS5. If doc_ids_filter is provided, limit to those docs.
    """
    if doc_ids_filter:
        placeholders = ",".join("?" for _ in doc_ids_filter)
        where = f"c.doc_id IN ({placeholders})"
        updated, skipped = mirror_from_db(conn, where_clause=where, where_args=tuple(doc_ids_filter))
    else:
        updated, skipped = mirror_from_db(conn)
    conn.commit()
    return updated, skipped

# ── CLI runner ─────────────────────────────────────────────────────────────
def run_pipeline(
    db_path: str,
    src_dirs: list[str],
    include_exts: list[str],
    nlp_version: str,
    keep_pos: bool,
    char_window: int,
    overlap: int,
    limit_docs: int | None,
    only_changed: bool,
    skip_spacy: bool,
    skip_fts: bool,
) -> None:
    t_all = Timer()
    src_dirs = [str(Path(p)) for p in src_dirs]
    exts = {e.lower().lstrip(".") for e in include_exts}

    with get_conn(DB_PATH) as conn:
        #ensure_db(conn)  # ensures chunk_embedding_state exists (and everything else)
        #conn.row_factory = sqlite3.Row
        ensure_base_schema(conn)

        # 1) Discover + Hash
        t = Timer()
        changed_docs = step1_discover(conn, src_dirs, exts)
        if not only_changed:
            # include unchanged as well (full rebuild path)
            # We pull doc_ids directly from doc_ingest to force a full run.
            rows = conn.execute("SELECT doc_id, file_path FROM doc_ingest").fetchall()
            changed_docs = [{"doc_id": r[0], "file_path": r[1], "changed": False} for r in rows]
        if limit_docs and len(changed_docs) > limit_docs:
            changed_docs = changed_docs[:limit_docs]
        print(f"① Discover & Hash → {len(changed_docs)} docs (in {t.ms()} ms)")

        if not changed_docs:
            print("Nothing to do. (No new/changed docs and not forcing full rebuild.)")
            return

        # Collect doc_ids for targeted later steps
        target_doc_ids = [r["doc_id"] for r in changed_docs]

        # 2) Front‑matter + Canonical
        t = Timer()
        step2_frontmatter(conn, changed_docs)
        print(f"② Front‑matter & Canonical (in {t.ms()} ms)") 

        # 3) Chunking
        t = Timer()
        step3_chunk(conn, changed_docs, target_char_window=char_window, overlap=overlap)
        print(f"③ Sectioning & Chunking (in {t.ms()} ms)")

        # 4) Taxonomy on Raw
        #t = Timer()
        #step4_taxonomy(conn, changed_docs)
        #print(f"④ Taxonomy annotate (in {t.ms()} ms)")

        # 5) spaCy Cleaning Cache
        if not skip_spacy:
            t = Timer()
            step5_spacy_cache(db_path, nlp_version=nlp_version, keep_pos=keep_pos)
            print(f"⑤ spaCy clean cache (in {t.ms()} ms)")
        else:
            print("⑤ spaCy clean cache (skipped)")

        # 6) FTS Mirror
        if not skip_fts:
            t = Timer()
            upd, skp = step6_fts(conn, doc_ids_filter=target_doc_ids)
            print(f"⑥ FTS mirror → updated={upd}, skipped={skp} (in {t.ms()} ms)")
        else:
            print("⑥ FTS mirror (skipped)")

    print(f"✅ Done in {t_all.ms()} ms")

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="End‑to‑end ingest: discover → FM → chunk → taxonomy → spaCy cache → FTS.")
    ap.add_argument("--db", required=True, help="SQLite path (e.g., C:/dev/GovernEdge_CLI/database_tst/chat_logs.sqlite)")
    ap.add_argument("--src", nargs="+", required=True, help="One or more source folders to walk")
    ap.add_argument("--ext", nargs="+", default=["md","txt","csv"], help="File extensions to include (no dots)")
    ap.add_argument("--nlp-version", default="spacy.v1", help="NLP cache version key (bump to force refresh)")
    ap.add_argument("--keep-pos", action="store_true", help="Store POS JSON in nlp cache (debug)")
    ap.add_argument("--char-window", type=int, default=420, help="Target char window for micro‑chunks")
    ap.add_argument("--overlap", type=int, default=60, help="Overlap chars between adjacent windows")
    ap.add_argument("--limit-docs", type=int, default=None, help="Process at most N docs (for quick tests)")
    ap.add_argument("--only-changed", action="store_true", help="Process only changed/new docs")
    ap.add_argument("--skip-spacy", action="store_true", help="Skip Step 5 (useful for quick dry runs)")
    ap.add_argument("--skip-fts", action="store_true", help="Skip Step 6 (FTS mirror)")

    return ap.parse_args(argv)

#python .\ingest_docs.py --db "C:\dev\GovernEdge_CLI\database\chat_logs.sqlite" --src "C:\dev\GovernEdge_CLI\docs" --ext md txt csv --nlp-version spacy.v1 --keep-pos --char-window 420 --overlap 60 --limit-docs 10 --only-changed --skip-spacy --skip-fts
#python .\ingest_docs.py --db "C:\dev\GovernEdge_CLI\database\chat_logs.sqlite" --src "C:\dev\GovernEdge_CLI\docs" --ext md txt csv --nlp-version spacy.v1 --keep-pos --char-window 420 --overlap 60 

# USE THIS ONE
# delete the nlp cache then re-run spacy clean to get part of speech
# python -m prepare_docs.spacy_clean --db "C:\dev\GovernEdge_CLI\database\chat_logs.sqlite" --nlp-version spacy.v1 --keep-pos
# python .\ingest_docs.py --db "C:\dev\GovernEdge_CLI\database\chat_logs.sqlite" --src "C:\dev\GovernEdge_CLI\data_tst\sap_docs" --ext md txt csv --nlp-version spacy.v1 --char-window 420 --overlap 60 
 

if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        db_path=args.db,
        src_dirs=args.src,
        include_exts=args.ext,
        nlp_version=args.nlp_version,
        keep_pos=args.keep_pos,
        char_window=args.char_window,
        overlap=args.overlap,
        limit_docs=args.limit_docs,
        only_changed=args.only_changed,
        skip_spacy=args.skip_spacy,
        skip_fts=args.skip_fts,
    )
