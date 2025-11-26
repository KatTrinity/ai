
# C:\dev\GovernEdge_CLI\prepare_docs\nlp_cache.py

"""
nlp_cache.py

Handles caching of NLP-cleaned text in SQLite:
- Computes SHA-256 for change detection
- Fetches chunks needing processing
- Upserts cleaned text + POS tagging into doc_nlp_cache
"""

import sqlite3, json, hashlib, logging
from contextlib import closing
from typing import Iterable, List, Dict, Any, Tuple

# ---------------- logging ----------------
logger = logging.getLogger(__name__)


# ---------------- hashing ----------------
def sha256(s: str) -> str:
    """
    Compute SHA-256 hex digest of a string.

    Args:
        s: input string

    Returns:
        hex digest
    """
    h = hashlib.sha256()
    h.update(s.encode("utf-8", "ignore"))
    return h.hexdigest()


# ---------------- fetch ----------------
def fetch_chunks_to_process(conn: sqlite3.Connection, nlp_version: str) -> List[Dict[str, Any]]:
    """
    Fetch all chunks and their cached NLP hashes for a given version.

    Args:
        conn: sqlite3 connection
        nlp_version: version string of your NLP pipeline (e.g., "nlp.v1")

    Returns:
        list of dicts: [{"chunk_id", "raw_text", "cached_hash"}, ...]
        - cached_hash is None if never cached
    """
    logger.info("Fetching chunks for NLP version=%s", nlp_version)

    rows = conn.execute(
        """
        SELECT c.chunk_id, c.body_raw AS raw_text, n.text_hash AS cached_hash
        FROM doc_chunks c
        LEFT JOIN doc_nlp_cache n
          ON n.chunk_id = c.chunk_id AND n.nlp_version = ?
        """,
        (nlp_version,),
    ).fetchall()

    logger.debug("Fetched %d chunks for processing", len(rows))
    return [{"chunk_id": r[0], "raw_text": r[1] or "", "cached_hash": r[2]} for r in rows]


# ---------------- upsert ----------------
def upsert_clean_batch(
    conn: sqlite3.Connection,
    nlp_version: str,
    batch: Iterable[Tuple[str, str, str, str | None]],
) -> None:
    """
    Upsert a batch of cleaned chunks into doc_nlp_cache.

    Args:
        conn: sqlite3 connection
        nlp_version: version string
        batch: iterable of tuples (chunk_id, text_hash, cleaned_text, pos_json_or_None)
    """
    batch = list(batch)
    if not batch:
        logger.info("No cleaned chunks to upsert for version=%s", nlp_version)
        return

    conn.executemany(
        """
        INSERT INTO doc_nlp_cache (chunk_id, nlp_version, text_hash, cleaned_text, pos_json, updated_at)
        VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(chunk_id, nlp_version) DO UPDATE SET
          text_hash=excluded.text_hash,
          cleaned_text=excluded.cleaned_text,
          pos_json=excluded.pos_json,
          updated_at=CURRENT_TIMESTAMP
        """,
        [(cid, nlp_version, th, ct, pj) for (cid, th, ct, pj) in batch],
    )
    logger.info("Upserted %d cleaned chunks into doc_nlp_cache", len(batch))
