

# C:\dev\GovernEdge_CLI\engine_query\insert_chat_logs.py

"""
 Persist chat interactions to SQLite and fetch recent turns for inspection.
 `insert_chat` writes one Q/A turn plus its ranked sources to the log tables,
 while `get_recent_chats` retrieves the latest entries for debugging/metrics.
"""

import sqlite3
import logging, json, time, hashlib
from pathlib import Path
from config_base.config import Config
#from prepare_docs.db_io import get_conn
from prepare_docs.db_io import get_conn, DB_PATH as DEFAULT_DB_PATH

# --- logging ---
logger = logging.getLogger(__name__)
#logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# --- config-resolved DB path (explicit is better than implicit) ---
DB_PATH = Path(getattr(Config, "DB_PATH", "database/chat_logs.sqlite"))

#----HELPER

def _normalize(obj):
    # Normalize strings and recurse into lists/dicts
    if isinstance(obj, str):
        return obj.replace("\r\n", "\n").replace("\r", "\n")
    if isinstance(obj, float):
        return float(f"{obj:.8g}")  # stable-ish float repr
    if isinstance(obj, dict):
        return {k: _normalize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_normalize(v) for v in obj]
    return obj

def canonicalize_messages(messages):
    """Stable string for Chat-style messages (role/content/tool fields)."""
    stable = _normalize(messages)
    return json.dumps(stable, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

def upsert_prompt(conn, system_prompt: str | None, messages_or_text) -> str:
    if isinstance(messages_or_text, str):
        full = _normalize(messages_or_text)
    else:
        full = _normalize(messages_or_text)

    body = {"sys": system_prompt or "", "full": full}
    canonical = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    h = hashlib.sha256(canonical.encode("utf-8", "ignore")).hexdigest()

    with conn:  # transactional; safe if already in a txn
        conn.execute("""
          INSERT INTO log_prompts(prompt_hash, system_prompt, prompt_text)
          VALUES (?, ?, ?)
          ON CONFLICT(prompt_hash) DO NOTHING
        """, (h, system_prompt, canonical))
    return h


def insert_chat(
    *,
    conn: sqlite3.Connection | None = None,
    db_path: str | Path | None = None,
    session_id: str,
    question: str,
    response: str,
    llm_key: str,
    embed_key: str,
    ce_key: str,
    ce_model: str | None = None,
    top_k: int = 0,
    k_rerank: int | None = None,
    keep_top: int | None = None,
    per_doc_cap: int | None = None,
    collection: str | None = None,
    latency_ms: int = 0,
    dense_ms: int | None = None,
    ce_ms: int | None = None,
    similarity_avg: float | None = None,
    prefiltered: int = 0,
    prompt_hash: str,
    docs: list | None = None,
    scores: list | None = None,
    run_meta: str | None = None 
) -> int:
    """Insert one chat turn + its final reranked sources. Return new turn_id."""
    own_conn = False
    if conn is None:
        conn = get_conn(db_path or DEFAULT_DB_PATH, ensure=True)
        own_conn = True

    conn.execute("PRAGMA foreign_keys=ON;")
    #conn.execute("PRAGMA foreign_keys=OFF;")
    #conn.execute("PRAGMA busy_timeout=2500;")

    # Normalize basic values
    k_dense_i      = int(top_k or 0)
    k_rerank_i     = int(k_rerank) if k_rerank is not None else None
    keep_top_i     = int(keep_top) if keep_top is not None else None
    per_doc_cap_i  = int(per_doc_cap) if per_doc_cap is not None else None
    latency_i      = int(latency_ms or 0)
    dense_i        = int(dense_ms) if dense_ms is not None else None
    ce_i           = int(ce_ms) if ce_ms is not None else None
    prefiltered_i  = 1 if prefiltered else 0

    with conn:  # one transaction
    # Parent row first
        cur = conn.execute("""
            INSERT INTO log_chat_turns (
            session_id, question, response,
            prompt_hash, run_meta,
            llm_key, embed_key, ce_key, ce_model,
            k_dense, k_rerank, keep_top, per_doc_cap, collection,
            latency_ms, dense_ms, ce_ms,
            prefiltered, similarity_avg
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            session_id, question, response,
            prompt_hash, run_meta,
            llm_key, embed_key, ce_key, ce_model,
            k_dense_i, k_rerank_i, keep_top_i, per_doc_cap_i, collection,
            latency_i, dense_i, ce_i,
            prefiltered_i, similarity_avg
        ))
        turn_id = int(cur.lastrowid)

        # ---- BULK parent existence check for docs/chunks ----
        existing_doc_ids: set[int] = set()
        existing_chunk_ids: set[str] = set()
        if docs:
            cand_doc_ids = []
            cand_chunk_ids = []
            for d in docs:
                m = getattr(d, "metadata", None) or (d if isinstance(d, dict) else {}) or {}
                raw_doc_id = m.get("doc_id")
                try:
                    if raw_doc_id is not None:
                        cand_doc_ids.append(int(raw_doc_id))
                except Exception:
                    pass
                cid = m.get("chunk_id")
                if cid:
                    cand_chunk_ids.append(str(cid))

            # de-dup
            cand_doc_ids = sorted(set(cand_doc_ids))
            cand_chunk_ids = sorted(set(cand_chunk_ids))

            if cand_doc_ids:
                q = f"SELECT doc_id FROM doc_ingest WHERE doc_id IN ({','.join('?'*len(cand_doc_ids))})"
                existing_doc_ids = {r[0] for r in conn.execute(q, cand_doc_ids).fetchall()}

            if cand_chunk_ids:
                q = f"SELECT chunk_id FROM doc_chunks WHERE chunk_id IN ({','.join('?'*len(cand_chunk_ids))})"
                existing_chunk_ids = {r[0] for r in conn.execute(q, cand_chunk_ids).fetchall()}

        # ---- Child rows (NULL out missing parents to satisfy FKs) ----
        if docs:
            rows = []
            for rank, d in enumerate(docs, 1):
                m = getattr(d, "metadata", None) or (d if isinstance(d, dict) else {}) or {}
                # dense_score alignment
                ds = None
                if scores is not None and 0 <= (rank - 1) < len(scores):
                    ds = scores[rank - 1]

                # doc_id (int or None) with membership guard
                raw_doc_id = m.get("doc_id")
                try:
                    doc_id = int(raw_doc_id) if raw_doc_id is not None else None
                except Exception:
                    doc_id = None
                if doc_id is not None and doc_id not in existing_doc_ids:
                    doc_id = None  # <- prevent FK failure

                # chunk_id (str or None) with membership guard
                chunk_id = m.get("chunk_id") or None
                if chunk_id is not None and chunk_id not in existing_chunk_ids:
                    chunk_id = None  # <- prevent FK failure

                rows.append((
                    turn_id, rank,
                    doc_id,
                    chunk_id,
                    m.get("section_id"),
                    m.get("file_path"),
                    m.get("title"),
                    m.get("header_path"),
                    m.get("span_start"),
                    m.get("span_end"),
                    (float(m["ce_score"]) if m.get("ce_score") is not None else None),
                    (float(ds) if ds is not None else None),
                    int(bool(m.get("used_in_prompt", 1))),
                    json.dumps(m.get("score_json") or {}, ensure_ascii=False),
                ))

            conn.executemany("""
                INSERT INTO log_chat_sources (
                turn_id, rank, doc_id, chunk_id, section_id, file_path, title,
                header_path, span_start, span_end, ce_score, dense_score,
                used_in_prompt, score_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, rows)


    if own_conn:
        conn.close()
    return turn_id



def get_recent_chats(limit=5):
    """
    Retrieve the N most recent chats with their questions and responses.
    """
    conn = get_conn(DB_PATH)
    cursor = conn.cursor()

    try:
        cursor.execute(
            "SELECT question, response, created_at FROM log_chat_turns ORDER BY created_at DESC LIMIT ?",
            (int(limit),)
        )
        rows = cursor.fetchall()
        logger.info("Fetched %d recent chats (limit=%d).", len(rows), limit)
        return rows
    except sqlite3.Error as e:
        logger.error("❌ Failed to fetch recent chats: %s", e)
        return []
    finally:
        conn.close()
