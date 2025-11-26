
# C:\dev\GovernEdge_CLI\engine_query\engine_cross_encoder.py

# --- helpers (drop near the top of query_engine.py) ---
"""
This module provides helper functions for hybrid retrieval reranking. 
It manages cross-encoder loading, applies query prefixes for specific embedders, 
and handles CPU thread hygiene for efficient scoring and reranking.
"""

import os, torch, logging
from functools import lru_cache
from sentence_transformers import CrossEncoder

# --- logging setup ---
logger = logging.getLogger(__name__)
#logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def apply_query_prefix_for_embedder(q: str, embed_model_name: str) -> str:
    """Prefix queries for certain embedding models that expect it."""
    if "nomic" in embed_model_name.lower() or "snow" in embed_model_name.lower():
        prefixed = f"Query: {q}"
        logger.debug("Applied query prefix for embedder=%s", embed_model_name)
        return prefixed
    return q


# good CPU hygiene
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
try:
    torch.set_num_threads(min(os.cpu_count() or 4, 8))
    torch.set_num_interop_threads(1)
    logger.info("✅ CPU threading parameters set for torch")
except Exception as e:
    logger.warning("⚠️ Failed to set torch CPU thread params: %s", e)


@lru_cache(maxsize=4)
def get_cross_encoder(model_name: str = "BAAI/bge-reranker-base"):
    """
    Load and cache a CrossEncoder for reranking candidate passages.
    Default is bge-reranker-base; MiniLM-L6-v2 is a lighter alternative.
    """
    logger.info("Loading cross encoder model: %s", model_name)
    device = "cpu"
    return CrossEncoder(model_name, device=device, max_length=512) 


def build_passage_text(doc, head_chars=1200, tail_chars=600):
    """Construct a truncated passage text from head and tail for reranking."""
    title  = doc.metadata.get("title", "")
    header = doc.metadata.get("header_path", "")
    body   = doc.page_content or ""
    head = body[:head_chars]
    tail = body[-tail_chars:] if len(body) > head_chars + tail_chars else ""
    return f"{title}\n{header}\n\n{head}{('\n' + tail) if tail else ''}"


def rerank_candidates(user_query: str, candidates, ce, batch_size=32, per_doc_cap=3, keep_top=12):
    """Rerank candidate documents with a cross-encoder, enforcing diversity and per-doc caps."""
    logger.info("Reranking %d candidates with CE (batch=%d, keep_top=%d)", len(candidates), batch_size, keep_top)
    pairs  = [(user_query, build_passage_text(d)) for d in candidates]
    scores = ce.predict(pairs, batch_size=batch_size, show_progress_bar=False)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)

    selected, seen = [], {}
    for d, s in ranked:
        did = d.metadata.get("doc_id")
        if did is not None and seen.get(did, 0) >= per_doc_cap:
            continue
        if did is not None:
            seen[did] = seen.get(did, 0) + 1
        d.metadata["ce_score"] = float(s)
        selected.append(d)
        if len(selected) >= keep_top:
            break
    logger.info("✅ Reranked down to %d final docs", len(selected))
    return selected, ranked


def choose_cpu_rerank(k_dense: int):
    """
    Auto-size rerank parameters for CPU.
    Returns dict with k_rerank, keep_top, and batch size.
    """
    if k_dense <= 80:
        plan = dict(k_rerank=min(24, k_dense), keep_top=10, batch=24)
    elif k_dense <= 120:
        plan = dict(k_rerank=32, keep_top=12, batch=24)
    else:
        plan = dict(k_rerank=32, keep_top=12, batch=24)
    logger.debug("CPU rerank plan for k_dense=%d: %s", k_dense, plan)
    return plan

