
# C:\dev\GovernEdge_CLI\engine_query\retriever_fts.py

"""
Lightweight FTS5 retriever utilities for hybrid search.
Builds a safe MATCH query, executes an FTS lookup (optionally constrained by facet SQL),
and normalizes BM25 scores to a [0..1] “higher-is-better” scale for fusion.
"""

from __future__ import annotations
import time, sqlite3, logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from pathlib import Path

from config_base.config import Config 
from prepare_docs.db_io import get_conn

# --- logging ---
logger = logging.getLogger(__name__)
#logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# --- Config-resolved default DB path (explicit beats implicit) ---
DB_PATH = Path(getattr(Config, "DB_PATH", "database/chat_logs.sqlite"))


def _fts_safe_match_text(q: str, max_terms: int = 8) -> str:
    """
    Build a permissive FTS5 MATCH string:
    - Keep quoted phrases ("...") as-is.
    - For the rest, drop punctuation (incl. commas) → spaces.
    - Take up to N tokens (len > 1) and join with OR.
    """
    import re
    q = (q or "").strip()
    if not q:
        return ""

    # 1) pull out quoted phrases first
    phrases = re.findall(r'"([^"]+)"', q)
    q_wo_phrases = re.sub(r'"[^"]+"', " ", q)

    # 2) kill commas and most punctuation → spaces
    q_wo_commas = q_wo_phrases.replace(",", " ")
    q_norm = re.sub(r"[^\w\-\/_ ]+", " ", q_wo_commas)  # keep IDs like GL-100, S/4

    # 3) tokens
    toks = [t for t in q_norm.split() if len(t) > 1][:max_terms]

    # 4) reassemble: phrases OR tokens
    parts = [f'"{p.strip()}"' for p in phrases if p.strip()]
    parts += toks
    if not parts:
        return ""

    return " OR ".join(parts)

#facet_sql=None, facet_params=None
def fts_search_rows(db_path, match_text, k):
    q_raw = (match_text or "")
    q = " ".join(q_raw.replace('"', " ").replace("'", " ").split()).strip()
    logger.info(f"FTS q_raw='{q_raw}' q_sanitized='{q}'")

    if not q:
        logger.warning("FTS skipped: empty query after sanitize/strip")
        return []

    # SUPER SIMPLE termization: use OR between first 8 tokens (avoid over-ANDing)
    toks = [t for t in q.split() if len(t) > 1][:8]
    if not toks:
        logger.warning("FTS skipped: no usable tokens (len>1)")
        return []
    
    match_param = _fts_safe_match_text(match_text)
    if not match_param:
        logger.warning("FTS skipped: empty/unsuitable query after sanitize")
        return []

    sql = """
        SELECT chunk_id, doc_id, title, header_path, body_raw,
            bm25(doc_chunks_fts) AS bm25_score
        FROM doc_chunks_fts
        WHERE doc_chunks_fts MATCH ?
        ORDER BY bm25(doc_chunks_fts)
        LIMIT ?;
        """
    params = [match_param, int(k)]

    try:
        with sqlite3.connect(db_path) as con:
            con.row_factory = sqlite3.Row
            cur = con.execute(sql, params)
            rows = [dict(r) for r in cur.fetchall()]
        logger.info(f"FTS hits={len(rows)}")
    except Exception as e:
        logger.exception(f"FTS error: {e}")
        rows = []

    return rows


def normalize_bm25_to_pos(rows: List[Dict]) -> Dict[str, float]:
    """
    Convert BM25 (lower-is-better) to a positive score (higher-is-better) for fusion.
    Uses 1/(1+bm25) and guards against negative/None inputs.
    """
    out: Dict[str, float] = {} 
    for r in rows:
        try:
            bm25 = float(r["bm25_score"])
        except (TypeError, ValueError):
            bm25 = 0.0
        out[r["chunk_id"]] = 1.0 / (1.0 + max(0.0, bm25))
    return out


# ---------- result ----------
@dataclass
class FTSResult:
    rows: List[Dict[str, Any]]
    rank: Dict[str, int]        # chunk_id -> position (1-based)
    by_id: Dict[str, float]     # chunk_id -> normalized score [0..1]
    ms: float

# ---------- one-call API ----------
def fts_search(db_path: str, question: str, k: int,
               facet: Optional[Dict[str, Any]] = None) -> FTSResult:
    q = "" if question is None else str(question)
    t0 = time.perf_counter()
    rows = fts_search_rows(db_path, q, k) #facet=facet
    ms = (time.perf_counter() - t0) * 1000.0

    by_id = normalize_bm25_to_pos(rows)
    rank  = {str(r["chunk_id"]): i for i, r in enumerate(rows, 1)}

    logger.info("FTS recall → %d rows in %.1f ms", len(rows), ms)
    return FTSResult(rows=rows, rank=rank, by_id=by_id, ms=ms)