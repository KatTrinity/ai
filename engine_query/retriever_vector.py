
# C:\dev\GovernEdge_CLI\engine_query\retriever_vector.py

"""
Vectorstore helpers for loading/inspecting Chroma collections used in hybrid retrieval.
Provides a cached loader (`get_vectorstore`), a safe corpus counter, and a utility
to translate facet filters into Chroma metadata queries.
"""
 #logger.info(f"FTS hits={len(rows)}")
 
from __future__ import annotations
from typing import Any, Dict, Optional, Tuple, List 
from pathlib import Path
from langchain_chroma import Chroma
#from langchain_community import similary_search_with_score 
from langchain_core.embeddings import Embeddings
from config_base.config import Config
from engine_query.fusion_helpers import build_rank, build_score_map
from engine_query.engine_load_models import apply_query_prefix_for_embedder
from engine_query.engine_load_models import get_embedder
from dataclasses import dataclass
from statistics import mean
import time, logging


# --- logging ---
logger = logging.getLogger(__name__)
#logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# --- Config fallbacks (keeps things resilient if keys are missing) ---
DIR_MAP: Dict[str, str] = getattr(Config, "CHROMA_DIR_MAP", {})

# ------------------------- internals ----------------------------------------
def resolve_persist_dir(model_key: str, explicit_dir: Optional[str]) -> str:
    """
    Pick a persist dir in order of precedence:
      1) explicit_dir (function arg)
      2) DIR_MAP[model_key]  # ← from Config.CHROMA_DIR_MAP at module import
      3) ./vectorstores/{model_key}
    Ensures the directory exists.
    """
    if explicit_dir:
        base = Path(explicit_dir)
    else:
        base = Path(DIR_MAP.get(model_key) or (Path("vectorstores") / model_key))
    base.mkdir(parents=True, exist_ok=True)
    return str(base.resolve())


def cache_key(model_key: str, persist_dir: str, collection_name: str) -> Tuple[str, str, str]:
    """
    Use a 3-tuple cache key so the same model_key can be opened
    from different dirs or with different collection names without collisions.
    """
    return (model_key, persist_dir, collection_name)

# ------------------------- public API ---------------------------------------
def get_vectorstore(
    cache: Dict[Tuple[str, str, str], Any],
    model_key: str,
    embeddings: Embeddings,
    *,
    persist_dir: Optional[str] = None,
    collection_name: Optional[str] = None,
):
    """
    Load/cache a Chroma collection keyed by (model_key, persist_dir, collection_name).
    """
    # Defensive guard to catch swapped args early:
    if not isinstance(cache, dict):
        raise TypeError(f"cache must be dict, got {type(cache)}")
    if persist_dir is None:
        raise ValueError("persist_dir is required")

    # sensible default if you keep a naming scheme elsewhere
    if not collection_name:
        collection_name = f"my_docs_tst__{model_key}"

    ckey = (model_key, persist_dir, collection_name)
    if ckey not in cache:
        cache[ckey] = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=persist_dir,
        )
    return cache[ckey]


def count_corpus(
    model_key: str,
    embeddings: Embeddings,
    cache: Dict[Tuple[str, str, str], Any],
    *,
    persist_dir: Optional[str] = None,
    collection_name: Optional[str] = None,
) -> int:
    """
    Return the number of vectors currently stored in the collection.
    Falls back to 0 on failure.
    """
    vs = get_vectorstore(
        cache=cache,
        model_key=model_key,
        embeddings=embeddings,
        persist_dir=persist_dir,
        collection_name=collection_name,
    )
    try:
        n = int(vs._collection.count())
        logger.debug("Collection '%s' has %d vectors", getattr(vs._collection, "name", "?"), n)
        return n
    except Exception as e:
        logger.warning("Could not read corpus count for '%s': %s", model_key, e)
        return 0

def to_chroma_filter(facet: Optional[dict]):
    """
    Convert a simple facet dict into a Chroma where-filter:
      {"field": "x"}        -> {"field": {"$eq": "x"}}
      {"field": ["a","b"]}  -> {"field": {"$in": ["a","b"]}}
    Multiple keys combine with $and.
    """
    if not facet: 
        return None

    clauses = []
    for k, v in facet.items(): 
        if v is None:
            continue
        if isinstance(v, (list, tuple, set)):
            vals = [str(x) for x in v if x is not None]
            if vals:
                clauses.append({k: {"$in": vals}})
        else:
            clauses.append({k: {"$eq": str(v)}})

    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}

__all__ = ["get_vectorstore", "count_corpus", "to_chroma_filter"]

@dataclass
class DenseSearchResult:
    pairs:   List[Tuple[object, float]]  # (Document, score)
    docs:    List[object]
    scores:  List[float]
    by_id:   dict[str, float]
    rank:    dict[str, int]
    ms:      int
    used_filter: bool
    embed_key: str
    model_name: str
    # Optional diagnostics you might add later:
    # collection: str | None = None
    # persist_dir: str | None = None
    # corpus_n: int | None = None

def dense_search(self, *, question: str, embed_key: str, k: int = 20,
                 facet_filter: Optional[dict] = None,
                 emb=None, vs=None) -> DenseSearchResult:
    embed_model_name = self._embed_map[embed_key]
    if emb is None:
        emb = get_embedder(self._embedder_cache, embed_model_name)
    if vs is None:
        vs  = get_vectorstore(
                cache=self._vs_cache,
                model_key=embed_key,
                embeddings=emb,
                persist_dir=self._chroma_dir_map[embed_key],
                collection_name=f"my_docs_tst__{embed_key}",
             )

    q_dense = apply_query_prefix_for_embedder(question, embed_model_name)
    where   = to_chroma_filter(facet_filter) if facet_filter else None

    logger.info("[VEC] model=%s k=%s filter_raw=%r q_len=%d",
                embed_model_name, k, facet_filter, len(q_dense or ""))
    if where:
        logger.info("[VEC] filter_chroma=%r", where)

    def _search(_where):
        t0 = time.time()
        pairs = vs.similarity_search_with_score(q_dense, k=k, filter=_where) if _where \
                else vs.similarity_search_with_score(q_dense, k=k)
        ms = int((time.time() - t0) * 1000)
        docs   = [d for d, _ in pairs]
        scores = [float(s) for _, s in pairs]
        by_id  = {str(d.metadata.get("chunk_id")): s
                  for d, s in zip(docs, scores) if d.metadata.get("chunk_id")}
        if docs:
            logger.info("[VEC] hits=%d time=%d ms score[min/avg/max]=[%.4f/%.4f/%.4f]",
                        len(docs), ms, min(scores), mean(scores), max(scores))
        else:
            logger.warning("[VEC] 0 hits time=%d ms (filter=%r)", ms, _where)
        return pairs, docs, scores, by_id, ms

    pairs, docs, scores, by_id, ms = _search(where)
    used_filter = bool(where)
    if not docs and where:
        logger.warning("[VEC] retrying WITHOUT filter (diagnostic fallback)")
        pairs, docs, scores, by_id, ms = _search(None)
        used_filter = False

    rank  = build_rank(pairs, name="dense") 
    by_id = build_score_map(pairs)

    return DenseSearchResult(
        pairs=pairs, docs=docs, scores=scores, by_id=by_id, rank=rank, ms=ms,
        used_filter=used_filter, embed_key=embed_key, model_name=embed_model_name,
    )