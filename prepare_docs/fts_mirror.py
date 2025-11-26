
# C:\dev\GovernEdge_CLI\prepare_docs\fts_mirror.py

"""
fts_mirror.py

Mirrors raw chunks into SQLite FTS5 tables for full-text search.
- Mode A: from doc_chunks table in DB
- Mode B: from in-memory LangChain Documents
Tracks hash state to avoid redundant inserts.
"""

from __future__ import annotations
import sqlite3, logging
from contextlib import closing
from typing import Iterable, Optional
from .db_io import DDL_FTS   # package-relative import so -m works

# ---------------- logging ----------------
logger = logging.getLogger(__name__)


def ensure_fts(conn: sqlite3.Connection) -> None:
    """
    Ensure FTS5 tables exist by running the DDL_FTS script.
    """
    conn.executescript(DDL_FTS)


# ── Core mirror op: delete + insert + state upsert ─────────────────────────
def _mirror_row(
    conn: sqlite3.Connection,
    *,
    title: str,
    header_path: str,
    body_raw: str,
    chunk_id: str,
    doc_id: int,
    chunk_hash_raw: str,
) -> None:
    """
    Insert or update a single chunk into FTS tables.
    Skips if hash unchanged; deletes old row first since FTS5 has no ON CONFLICT.
    """
    row = conn.execute(
        "SELECT chunk_hash FROM doc_chunks_fts_state WHERE chunk_id=?",
        (chunk_id,),
    ).fetchone()
    if row and row[0] == chunk_hash_raw:
        logger.debug("Skipped unchanged chunk_id=%s", chunk_id)
        return

    # delete old row if present
    conn.execute("DELETE FROM doc_chunks_fts WHERE chunk_id=?", (chunk_id,))

    # insert fresh into FTS
    conn.execute(
        """
        INSERT INTO doc_chunks_fts (title, header_path, body_raw, chunk_id, doc_id)
        VALUES (?, ?, ?, ?, ?)
        """,
        (title or "", header_path or "", body_raw or "", chunk_id, doc_id),
    )

    # track state in parallel table
    conn.execute(
        """
        INSERT INTO doc_chunks_fts_state (chunk_id, chunk_hash, updated_at)
        VALUES (?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(chunk_id) DO UPDATE SET
          chunk_hash=excluded.chunk_hash,
          updated_at=CURRENT_TIMESTAMP
        """,
        (chunk_id, chunk_hash_raw),
    )
    logger.debug("Mirrored chunk_id=%s (doc_id=%s)", chunk_id, doc_id)


# ── Mode A: mirror from doc_chunks table ───────────────────────────────────
def mirror_from_db(
    conn: sqlite3.Connection,
    where_clause: Optional[str] = None,
    where_args: tuple = (),
) -> tuple[int, int]:
    """
    Mirror rows from doc_chunks into FTS5.

    Args:
        conn: sqlite3 connection
        where_clause: optional WHERE clause (string, no 'WHERE' prefix)
        where_args: parameters for the WHERE clause

    Returns:
        (updated, skipped): counts of processed chunks
    """
    ensure_fts(conn)

    # tolerate legacy hash column name
    info = conn.execute("PRAGMA table_info(doc_chunks)").fetchall()
    col_name = "chunk_hash_raw" if any(c[1] == "chunk_hash_raw" for c in info) else None
    if not col_name and any(c[1] == "chunk_hash" for c in info):
        col_name = "chunk_hash"
    if not col_name:
        raise RuntimeError("doc_chunks must include 'chunk_hash_raw' (or legacy 'chunk_hash').")

    sql = f"""
      SELECT
        c.chunk_id,
        c.doc_id,
        COALESCE(c.title, d.title, d.file_name, '')       AS title,
        COALESCE(c.header_path, '')                        AS header_path,
        c.body_raw,
        {col_name}                                         AS chunk_hash_raw
      FROM doc_chunks c
      JOIN doc_ingest d ON d.doc_id = c.doc_id
      {"WHERE " + where_clause if where_clause else ""}
      ORDER BY c.doc_id, c.chunk_id
    """
    updated = skipped = 0
    for cid, did, title, header, body, chash in conn.execute(sql, where_args):
        row = conn.execute(
            "SELECT chunk_hash FROM doc_chunks_fts_state WHERE chunk_id=?",
            (cid,),
        ).fetchone()
        if row and row[0] == chash:
            skipped += 1
            continue

        _mirror_row(
            conn,
            title=title,
            header_path=header,
            body_raw=body or "",
            chunk_id=cid,
            doc_id=did,
            chunk_hash_raw=chash,
        )
        updated += 1

    logger.info("Mirror from DB complete → updated=%d, skipped=%d", updated, skipped)
    return updated, skipped


# ── Mode B: mirror from in-memory Documents ────────────────────────────────
def mirror_from_documents(conn: sqlite3.Connection, chunks: Iterable) -> tuple[int, int]:
    """
    Mirror chunks from in-memory LangChain Documents into FTS5.

    Args:
        chunks: iterable of Documents with:
            d.page_content
            d.metadata['chunk_id'], d.metadata['doc_id']
            d.metadata['header_path'] (optional)
            d.metadata['title'] (optional)
            d.metadata['chunk_hash_raw'] or ['chunk_hash']

    Returns:
        (updated, skipped): counts of processed chunks
    """
    ensure_fts(conn)
    updated = skipped = 0
    for d in chunks:
        m = getattr(d, "metadata", {}) or {}
        body = getattr(d, "page_content", "") or ""
        cid  = m.get("chunk_id")
        did  = m.get("doc_id")
        header = m.get("header_path", "") or ""
        title  = m.get("title", "") or ""
        chash  = m.get("chunk_hash_raw") or m.get("chunk_hash")

        if not (cid and did is not None and chash):
            logger.warning("Skipping invalid chunk: missing ids/hash")
            continue

        row = conn.execute(
            "SELECT chunk_hash FROM doc_chunks_fts_state WHERE chunk_id=?",
            (cid,),
        ).fetchone()
        if row and row[0] == chash:
            skipped += 1
            continue

        _mirror_row(
            conn,
            title=title,
            header_path=header,
            body_raw=body,
            chunk_id=cid,
            doc_id=int(did),
            chunk_hash_raw=chash,
        )
        updated += 1

    logger.info("Mirror from documents complete → updated=%d, skipped=%d", updated, skipped)
    return updated, skipped


# ── CLI entrypoint ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    ap = argparse.ArgumentParser(description="Mirror raw chunks into SQLite FTS5.")
    ap.add_argument("--db", required=True, help="Path to SQLite DB")
    ap.add_argument("--where", default=None, help="Optional SQL WHERE clause for doc_chunks (e.g., doc_id=123)")
    args = ap.parse_args()

    with closing(sqlite3.connect(args.db)) as conn:
        conn.execute("PRAGMA foreign_keys=ON;")
        conn.execute("PRAGMA journal_mode=WAL;")
        upd, skp = mirror_from_db(conn, where_clause=args.where)
        conn.commit()
        logger.info("✅ FTS mirror complete → updated=%d, skipped=%d", upd, skp)
