#!/usr/bin/env python3
# debug_engine.py
"""
Standalone probes & fusion for GovernEdge — no edits to your QueryEngine.

What you get:
- probe_dense(question, embed_key, k, facet_filter=None)
- probe_fts(question, k, facet_filter=None, db_path=None, fts_table=None, chunks_table=None)
- probe_sql(sql, params=(), db_path=None)
- fuse_candidates(dense_hits, fts_hits, weights={"dense":0.7, "fts":0.3}, rrf=False, rrf_k=60)
- debug_search(question, embed_key, k, facet_filter=None, weights=..., rrf=False, rrf_k=60)
- CLI entrypoints (see `python debug_engine.py -h`)

Notes:
- Uses Config.* maps/paths just like QueryEngine.from_env()
- Loads embedder/vectorstore directly (same loaders your engine uses)
- Hydrates FTS rows into LangChain Documents so you can compare apples-to-apples
"""

from __future__ import annotations
import time, json, argparse, sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from contextlib import closing
from collections import defaultdict

# LangChain bits
from langchain.schema import Document

# Your project loaders (same ones QueryEngine uses)
from config_base.config import Config
from engine_query.engine_load_models import load_embedding_model
from vector_index.vector_index_loader import load_vectorstore_and_metadata_test
from engine_query.engine_cross_encoder import apply_query_prefix_for_embedder

# ---------- helpers to read env / config ----------

def _get_db_path() -> str:
    return str(getattr(Config, "DB_PATH", Path("database") / "chat_logs.sqlite"))

def _get_fts_table() -> str:
    return getattr(Config, "FTS_TABLE", "doc_chunks_fts")

def _get_chunks_table() -> str:
    return getattr(Config, "CHUNKS_TABLE", "doc_chunks")

def _get_embed_map() -> Dict[str, str]:
    return getattr(Config, "EMBEDDING_MODEL_MAP", {})

def _get_chroma_dir_map() -> Dict[str, str]:
    return getattr(Config, "CHROMA_DIR_MAP", {})

def _open_ro_conn(db_path: str) -> sqlite3.Connection:
    # URI ro mode to avoid accidental writes
    return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, check_same_thread=False)

def _to_chroma_where(facet: dict | None):
    """Convert facet filter dict into Chroma filter format."""
    if not facet:
        return None
    clauses = []
    for k, v in facet.items():
        if v is None:
            continue
        if isinstance(v, (list, tuple, set)):
            vals = [str(x) for x in v if x is not None]
            if not vals:
                continue
            clauses.append({k: {"$in": vals}})
        else:
            clauses.append({k: {"$eq": str(v)}})
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}

# ---------- probes ----------

def probe_dense(
    question: str,
    embed_key: str,
    k: int = 8,
    facet_filter: dict | None = None,
) -> List[dict]:
    """
    Returns a list of dicts with: chunk_id, doc_id, title, header, score, preview
    """
    embed_map = _get_embed_map()
    if embed_key not in embed_map:
        raise ValueError(f"Unknown embed_key '{embed_key}'. Known: {list(embed_map)}")

    embed_model_name = embed_map[embed_key]
    embedder = load_embedding_model(embed_model_name)
    vs, _meta = load_vectorstore_and_metadata_test(embedder, model_key=embed_key)

    q_dense = apply_query_prefix_for_embedder(question, embed_model_name)
    where = _to_chroma_where(facet_filter or None)

    t0 = time.time()
    res = vs.similarity_search_with_score(q_dense, k=k, filter=where) if where else vs.similarity_search_with_score(q_dense, k=k)
    ms = int((time.time() - t0) * 1000)

    out = []
    for d, s in res:
        out.append({
            "chunk_id": d.metadata.get("chunk_id"),
            "doc_id": d.metadata.get("doc_id"),
            "title": d.metadata.get("title"),
            "header_path": d.metadata.get("header_path"),
            "score": float(s),
            "preview": (d.page_content or "")[:200],
        })
    # Attach timing as pseudo-row (client can ignore)
    return out

def _fts_query_sql(fts_table: str, match_text: str, facet_sql: str | None = None) -> tuple[str, list]:
    base = f"""
    SELECT chunk_id, doc_id, title, header_path, body_raw,
           bm25({fts_table}) AS bm25_score
    FROM {fts_table}
    WHERE {fts_table} MATCH ?
    """
    if facet_sql:
        base += f" {facet_sql}\n"
    base += " ORDER BY bm25_score LIMIT ?"
    return base, [match_text]

def _facet_to_sql(chunks_table: str, facet_filter: Optional[dict]) -> tuple[str, list]:
    if not facet_filter:
        return "", []
    clauses, params = [], []
    for key, val in facet_filter.items():
        clauses.append(f"{key} = ?")
        params.append(val)
    where = f" AND chunk_id IN (SELECT chunk_id FROM {chunks_table} WHERE " + " AND ".join(clauses) + ")"
    return where, params

def probe_fts(
    question: str,
    k: int = 8,
    facet_filter: dict | None = None,
    db_path: str | None = None,
    fts_table: str | None = None,
    chunks_table: str | None = None,
) -> List[dict]:
    """
    Returns a list of dicts with: chunk_id, doc_id, title, header_path, bm25, preview
    """
    match_text = (question or "").strip()
    if not match_text:
        return []

    db_path = db_path or _get_db_path()
    fts_table = fts_table or _get_fts_table()
    chunks_table = chunks_table or _get_chunks_table()

    facet_sql, facet_params = _facet_to_sql(chunks_table, facet_filter)
    sql, params = _fts_query_sql(fts_table, match_text=match_text, facet_sql=facet_sql)
    bind = params + facet_params + [int(k)]

    t0 = time.time()
    with _open_ro_conn(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = [dict(r) for r in conn.execute(sql, bind).fetchall()]
    ms = int((time.time() - t0) * 1000)

    for r in rows:
        r["bm25"] = float(r.pop("bm25_score", 0.0))
        r["preview"] = (r.get("body_raw") or "")[:200]
    return rows

def probe_sql(
    sql: str,
    params: tuple = (),
    db_path: str | None = None,
) -> List[dict]:
    db_path = db_path or _get_db_path()
    with _open_ro_conn(db_path) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(r) for r in conn.execute(sql, params).fetchall()]

# ---------- fusion & hydration ----------

def _normalize_dense_scores(scores: List[float]) -> List[float]:
    return [max(0.0, min(1.0, float(s))) for s in scores]

def _normalize_bm25_to_pos_score(rows: List[dict]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for r in rows:
        bm25 = float(r.get("bm25", 0.0))
        out[r["chunk_id"]] = 1.0 / (1.0 + max(0.0, bm25))
    return out

def fuse_candidates(
    dense_hits: List[dict],
    fts_hits: List[dict],
    weights: Dict[str, float] = {"dense": 0.7, "fts": 0.3},
    rrf: bool = False,
    rrf_k: int = 60,
) -> Tuple[List[str], Dict[str, float], Dict[str, int], Dict[str, int]]:
    """
    Returns:
      top_ids, fused_score_by_id, dense_rank, fts_rank
    (No hydration here—just IDs & scores)
    """
    dense_by_id = {h["chunk_id"]: float(h["score"]) for h in dense_hits}
    dense_by_id = {k: v for k, v in dense_by_id.items() if k}  # guard None
    dense_rank  = {h["chunk_id"]: i for i, h in enumerate(dense_hits, start=1) if h.get("chunk_id")}

    fts_by_id = _normalize_bm25_to_pos_score(fts_hits)
    fts_rank  = {h["chunk_id"]: i for i, h in enumerate(fts_hits, start=1) if h.get("chunk_id")}

    all_ids = set(dense_by_id) | set(fts_by_id)

    if rrf:
        def _rrf(ranks: Dict[str, int]) -> Dict[str, float]:
            return {cid: 1.0 / (rrf_k + r) for cid, r in ranks.items()}
        fused_score = {cid: _rrf(dense_rank).get(cid, 0.0) + _rrf(fts_rank).get(cid, 0.0) for cid in all_ids}
    else:
        w_dense = float(weights.get("dense", 0.7))
        w_fts   = float(weights.get("fts", 0.3))
        fused_score = {cid: w_dense * dense_by_id.get(cid, 0.0) + w_fts * fts_by_id.get(cid, 0.0) for cid in all_ids}

    top_ids = sorted(all_ids, key=lambda cid: fused_score[cid], reverse=True)
    return top_ids, fused_score, dense_rank, fts_rank

def _hydrate_docs(
    dense_hits: List[dict],
    fts_hits: List[dict],
    top_ids: List[str],
    keep_top: int = 12,
) -> List[Document]:
    """
    Build LangChain Documents for the fused set.
    Dense hits already have text in the vectorstore Document; here we only have dicts,
    so hydrate from FTS rows when needed.
    """
    by_id_doc: Dict[str, Document] = {}

    # Dense: we only have previews/scores here. In real engine you'd keep Document objects.
    # For debugging, we can’t access vectorstore docs directly, so we’ll skip populating dense text.
    # FTS: hydrate from body_raw (safe for debug).
    for r in fts_hits:
        cid = r.get("chunk_id")
        if cid and cid not in by_id_doc:
            page = r.get("body_raw") or ""
            by_id_doc[cid] = Document(
                page_content=page,
                metadata={
                    "chunk_id": cid,
                    "doc_id": r.get("doc_id"),
                    "title": r.get("title"),
                    "header_path": r.get("header_path"),
                },
            )

    # If a top_id isn’t found (dense-only hit), we’ll create a stub Document with empty body.
    docs: List[Document] = []
    for cid in top_ids[:keep_top]:
        d = by_id_doc.get(cid) or Document(page_content="", metadata={"chunk_id": cid})
        docs.append(d)
    return docs

def debug_search(
    question: str,
    embed_key: str,
    k: int = 12,
    facet_filter: dict | None = None,
    weights: Dict[str, float] = {"dense": 0.7, "fts": 0.3},
    rrf: bool = False,
    rrf_k: int = 60,
    db_path: str | None = None,
    fts_table: str | None = None,
    chunks_table: str | None = None,
) -> Tuple[List[Document], Dict[str, Any]]:
    """
    End-to-end debug run (no LLM):
      - probe dense
      - probe fts
      - fuse
      - hydrate
    Returns:
      (docs, dbg_dict)
    """
    t0 = time.time()
    dense_hits = probe_dense(question, embed_key=embed_key, k=k, facet_filter=facet_filter)
    dense_ms = int((time.time() - t0) * 1000)

    t1 = time.time()
    fts_hits = probe_fts(question, k=k, facet_filter=facet_filter, db_path=db_path, fts_table=fts_table, chunks_table=chunks_table)
    fts_ms = int((time.time() - t1) * 1000)

    top_ids, fused_score, dense_rank, fts_rank = fuse_candidates(dense_hits, fts_hits, weights=weights, rrf=rrf, rrf_k=rrf_k)
    docs = _hydrate_docs(dense_hits, fts_hits, top_ids, keep_top=min(k, 12))

    dbg = {
        "q": question,
        "k": k,
        "timing_ms": {"dense": dense_ms, "fts": fts_ms},
        "dense": [
            {
                "chunk_id": h["chunk_id"],
                "doc_id": h.get("doc_id"),
                "score": float(h.get("score", 0.0)),
                "title": h.get("title"),
                "header": h.get("header_path"),
                "preview": h.get("preview"),
            } for h in dense_hits[:k]
        ],
        "fts": [
            {
                "chunk_id": r.get("chunk_id"),
                "doc_id": r.get("doc_id"),
                "score": float(1.0 / (1.0 + max(0.0, r.get("bm25", 0.0)))),  # normalized display
                "title": r.get("title"),
                "header": r.get("header_path"),
                "preview": r.get("preview"),
            } for r in fts_hits[:k]
        ],
        "fused": [
            {
                "chunk_id": cid,
                "score": float(fused_score[cid]),
                "source_dense": float(next((h["score"] for h in dense_hits if h["chunk_id"] == cid), 0.0)),
                "source_fts": float(1.0 / (1.0 + max(0.0, next((r["bm25"] for r in fts_hits if r["chunk_id"] == cid), 0.0)))),
            } for cid in top_ids[:k]
        ],
    }
    return docs, dbg

# ---------- CLI ----------

def _print_json(obj):
    print(json.dumps(obj, ensure_ascii=False, indent=2))

def main(argv=None):
    ap = argparse.ArgumentParser(description="GovernEdge debug probes (dense / fts / sql / fusion).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_dense = sub.add_parser("dense", help="Probe dense vector recall")
    ap_dense.add_argument("--q", required=True, help="Question text")
    ap_dense.add_argument("--embed", required=True, help="Embedding key from Config.EMBEDDING_MODEL_MAP")
    ap_dense.add_argument("-k", type=int, default=8)
    ap_dense.add_argument("--facet-json", default=None, help='Optional facet filter JSON, e.g. {"action":"fix"}')

    ap_fts = sub.add_parser("fts", help="Probe FTS recall")
    ap_fts.add_argument("--q", required=True)
    ap_fts.add_argument("-k", type=int, default=8)
    ap_fts.add_argument("--db", default=None)
    ap_fts.add_argument("--fts-table", default=None)
    ap_fts.add_argument("--chunks-table", default=None)
    ap_fts.add_argument("--facet-json", default=None)

    ap_sql = sub.add_parser("sql", help="Run an arbitrary SQL against your DB")
    ap_sql.add_argument("--db", default=None)
    ap_sql.add_argument("--sql", required=True)

    ap_run = sub.add_parser("run", help="End-to-end: dense + fts + fusion + hydrate (no LLM)")
    ap_run.add_argument("--q", required=True)
    ap_run.add_argument("--embed", required=True)
    ap_run.add_argument("-k", type=int, default=12)
    ap_run.add_argument("--weights", default='{"dense":0.7,"fts":0.3}')
    ap_run.add_argument("--rrf", action="store_true")
    ap_run.add_argument("--rrf-k", type=int, default=60)
    ap_run.add_argument("--facet-json", default=None)
    ap_run.add_argument("--db", default=None)
    ap_run.add_argument("--fts-table", default=None)
    ap_run.add_argument("--chunks-table", default=None)

    args = ap.parse_args(argv)

    # Parse facet JSON if provided
    import json as _json
    facet = None
    if getattr(args, "facet_json", None):
        facet = _json.loads(args.facet_json)

    if args.cmd == "dense":
        out = probe_dense(args.q, embed_key=args.embed, k=args.k, facet_filter=facet)
        _print_json(out)
        return

    if args.cmd == "fts":
        out = probe_fts(args.q, k=args.k, facet_filter=facet, db_path=args.db, fts_table=args.fts_table, chunks_table=args.chunks_table)
        _print_json(out)
        return

    if args.cmd == "sql":
        out = probe_sql(args.sql, db_path=args.db)
        _print_json(out)
        return

    if args.cmd == "run":
        weights = _json.loads(args.weights) if args.weights else {"dense":0.7, "fts":0.3}
        docs, dbg = debug_search(
            question=args.q,
            embed_key=args.embed,
            k=args.k,
            facet_filter=facet,
            weights=weights,
            rrf=args.rrf,
            rrf_k=args.rrf_k,
            db_path=args.db,
            fts_table=args.fts_table,
            chunks_table=args.chunks_table,
        )
        _print_json(dbg)
        return

if __name__ == "__main__":
    main()
