
# C:\dev\GovernEdge_CLI\utils\utils_k_dense.py

import logging
from math import ceil
from langchain_chroma import Chroma
from config_base.config import Config

# ----------------------- logging -------------------------------------------
logger = logging.getLogger(__name__)
 
# ---- Auto sizing helpers ----
def choose_k_dense(
    num_chunks: int,
    pct: float = 0.006,     # 0.6% default (rule-of-thumb: 0.3–1.0%)
    cap: int = 200,
    floor: int = 60,
    messy_query: bool = False,
    prefiltered: bool = False,
) -> int:
    """
    Heuristically size the dense-retriever candidate pool.

    Args:
        num_chunks: total corpus size (vectors in store).
        pct: percentage of corpus to pull (rule-of-thumb 0.3–1.0%).
        cap: upper bound (don’t ever pull more than this many).
        floor: lower bound (minimum pool size).
        messy_query: if True, enlarge pool (short/ambiguous queries).
        prefiltered: if True, shrink pool (facet prefilter already applied).

    Returns:
        k: integer candidate pool size.
    """
    base = max(floor, int(round(num_chunks * pct)))
    k = min(cap, base)

    if messy_query:
        k = min(cap, int(ceil(k * 1.3)))  # widen pool by 30%

    if prefiltered:
        # shrink pool if you already constrained the search space
        k = max(int(floor * 0.7), int(k * 0.7))

    final_k = max(1, k)
    logger.debug(
        f"choose_k_dense → num_chunks={num_chunks}, base={base}, "
        f"messy={messy_query}, prefiltered={prefiltered}, final_k={final_k}"
    )
    return final_k


def chroma_count(collection_name: str, persist_dir: str, embeddings) -> int:
    """
    Count entries in a Chroma collection (direct low-level call).

    Args:
        collection_name: Chroma collection identifier.
        persist_dir: directory where collection is persisted.
        embeddings: embedding function used by this collection.

    Returns:
        int count of vectors in the collection.
    """
    try:
        store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=persist_dir,
        )
        count = store._collection.count()
        logger.info(f"Chroma collection '{collection_name}' count={count}")
        return count
    except Exception as e:
        logger.error(f"❌ Failed to count Chroma collection '{collection_name}': {e}")
        return 0


def resolve_collection_name(model_key: str) -> str:
    """
    Build a collection name for a given embedder key.

    Default behavior: prefix from Config.CHROMA_COLLECTION_PREFIX
    plus the model_key (e.g., "my_docs_tst__minilm").

    If you want a single shared collection, update this to return
    Config.CHROMA_COLLECTION_NAME instead.
    """
    prefix = getattr(Config, "CHROMA_COLLECTION_PREFIX", "my_docs_tst__")
    name = f"{prefix}{model_key}"
    logger.debug(f"Resolved collection name for '{model_key}' → {name}")
    return name


