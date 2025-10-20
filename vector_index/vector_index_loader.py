
# C:\dev\GovernEdge_CLI\vector_index\vector_index_loader.py

"""
Vectorstore loader utilities for Chroma.

This module provides helpers to resolve the persist directory and collection name
for a given model key, then load the corresponding Chroma vectorstore. It ensures
that collection naming stays consistent with your builder/retriever, and returns
both the vectorstore and metadata for inspection or debugging.
"""

import os
import sys
import logging
from config_base.config import Config
from langchain_community.vectorstores import Chroma
from utils.utils_k_dense import resolve_collection_name

# --- Path setup ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
os.environ["CHROMA_TELEMETRY"] = "False"

# --- Logging ---
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def persist_dir(model_key: str, index_dir: str | None) -> str:
    """Resolve the persist directory from config or an explicit override."""
    if index_dir:
        return index_dir
    d = Config.CHROMA_DIR_MAP.get(model_key)
    if not d:
        raise ValueError(f"❌ No index directory for model_key={model_key}")
    return d


def collection_name(model_key: str, variant: str, base: str | None = None) -> str:
    """Resolve the collection name consistently with builder/retriever logic."""
    base = base or resolve_collection_name(model_key)
    return base if variant == "chunks" else f"{base}__sections"


def load_vectorstore_and_metadata_test(
    embedder,
    model_key: str | None = None,
    index_dir: str | None = None,
    collection_name: str | None = None,
    variant: str = "chunks",          # or "sections" if you maintain a second collection
    probe_query: str | None = None,   # set to "test" if you want to probe after load
    k_probe: int = 1,
):
    """
    Load a Chroma collection for the given model_key.
    If index_dir/collection_name are not provided, they are resolved via config + resolve_collection_name().
    Returns (vectorstore, metadata).
    """
    if embedder is None:
        raise RuntimeError("❌ embedding_function must be passed explicitly.")

    # --- resolve persist dir
    if index_dir is None:
        if model_key is None:
            raise ValueError("❌ 'model_key' is required if 'index_dir' is not provided.")
        index_dir = persist_dir(model_key, None)
        logger.info(f"📂 Using default persist_dir for '{model_key}': {index_dir}")

    # --- resolve collection name
    if collection_name is None:
        if model_key is None:
            raise ValueError("❌ 'model_key' is required if 'collection_name' is not provided.")
        collection_name = collection_name or collection_name or resolve_collection_name(model_key)
        if variant != "chunks":
            collection_name = f"{collection_name}__sections"

    logger.info(f"🔍 Loading Chroma collection '{collection_name}' from: {index_dir}")

    # --- open store
    vs = Chroma(
        persist_directory=index_dir,
        collection_name=collection_name,
        embedding_function=embedder,
    )

    # --- defensive patch
    if getattr(vs, "_embedding_function", None) is None:
        logger.warning("⚠️ Vectorstore missing embedder after load — patching.")
        vs._embedding_function = embedder
    coll = getattr(vs, "_collection", None)
    if coll is None:
        raise RuntimeError("❌ Could not access underlying Chroma collection (API mismatch).")
    if getattr(coll, "_embedding_function", None) is None:
        coll._embedding_function = embedder

    # --- metadata
    entry_count = coll.count()
    logger.info(f"✅ Loaded collection '{coll.name}' with {entry_count} vectors.")

    sample = None
    if probe_query:
        try:
            sample = vs.similarity_search(probe_query, k=k_probe)
            if sample:
                logger.info(f"🧾 Probe preview: {sample[0].page_content[:120].replace('\n', ' ')}")
            else:
                logger.warning("⚠️ Probe returned no docs.")
        except Exception as e:
            logger.warning(f"⚠️ Probe query failed: {e}")

    metadata = {
        "collection_name": coll.name,
        "persist_dir": index_dir,
        "entry_count": entry_count,
        "model_key": model_key,
        "variant": variant,
    }
    return vs, metadata
