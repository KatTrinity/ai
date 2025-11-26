
# C:\dev\GovernEdge_CLI\utils\utils_ghost_check.py
"""
ghost_check.py

Audit a Chroma collection against your taxonomy YAMLs.
- Detects "ghosts": taxonomy entries never observed in the vectorstore.
- Detects "strays": metadata values found in the vectorstore but not in taxonomy YAML.

Useful for validating ingestion → embedding → taxonomy pipelines.
"""

import argparse, logging
from collections import defaultdict
from pathlib import Path

from engine_query.engine_load_models import resolve_embed_config  # ✅ use canonical embed resolver
from utils.utils_k_dense import resolve_collection_name
from taxonomies.apply_taxonomy import TaxonomyMatcher
from langchain_community.vectorstores import Chroma

# ---------------- logging ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ghost_check")

# ---------------- config -----------------
TAXONOMY_DIR = Path(r"C:\dev\GovernEdge_CLI\taxonomies")

# ---------------- main audit ----------------
def ghost_check(model_key: str = "minilm", peek: int = 5000) -> dict:
    """
    Compare taxonomy YAML against values present in a Chroma vectorstore.

    Args:
        model_key: short key for embedding model (e.g. "minilm", "snow", "nomic").
        peek: number of vectors to sample from Chroma.

    Returns:
        dict with ghosts/strays per facet: {"action": (ghosts, strays), ...}
    """
    # Resolve embedding + persist_dir from config
    embed_model_name, persist_dir = resolve_embed_config(model_key)
    coll_name = resolve_collection_name(model_key)

    logger.info("📂 Opening Chroma collection '%s' (dir=%s)", coll_name, persist_dir)
    vs = Chroma(collection_name=coll_name,
                persist_directory=persist_dir,
                embedding_function=None)  # no need for embeddings when just peeking

    # Collect metadata seen in index
    seen = defaultdict(set)
    try:
        batch = vs._collection.peek(limit=peek)
    except Exception as e:
        logger.error("❌ Failed to peek Chroma collection: %s", e)
        return {}

    for m in batch.get("metadatas", []):
        for k in ("action", "object", "category"):
            v = (m or {}).get(k)
            if v:
                seen[k].add(str(v).strip().lower())

    logger.info("Collected %d metadata entries (sample size=%d)", sum(len(s) for s in seen.values()), peek)

    # Load canonical keys from taxonomy YAMLs
    tm = TaxonomyMatcher.from_folder(TAXONOMY_DIR)
    yaml_keys = {
        "action": set(tm.actions_rx.keys()),
        "object": set(tm.objects_rx.keys()),
        "category": set(tm.categories_rx.keys()),
    }
    logger.info("Loaded taxonomy YAMLs from %s", TAXONOMY_DIR)

    # Compare taxonomy vs index
    out: dict[str, tuple[list[str], list[str]]] = {}
    for k in ("action", "object", "category"):
        ghosts = sorted(yaml_keys[k] - seen[k])   # in YAML but not in index
        strays = sorted(seen[k] - yaml_keys[k])   # in index but not in YAML
        out[k] = (ghosts, strays)

        if ghosts:
            logger.warning("[%s] Ghosts (YAML but not in index): %s", k, ghosts)
        else:
            logger.info("[%s] Ghosts: none", k)

        if strays:
            logger.warning("[%s] Strays (Index but not in YAML): %s", k, strays)
        else:
            logger.info("[%s] Strays: none", k)

    return out

# ---------------- CLI ----------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Audit vectorstore taxonomy coverage (ghosts vs strays).")
    ap.add_argument("--model_key", default="minilm", help="Model key (e.g., minilm, nomic, snow)")
    ap.add_argument("--peek", type=int, default=5000, help="Number of vectors to peek from Chroma")
    args = ap.parse_args()

    results = ghost_check(args.model_key, args.peek)

    # Exit code: fail if any ghosts/strays found
    offenders = sum(len(g) + len(s) for g, s in results.values())
    if offenders:
        logger.error("❌ Found %d taxonomy mismatches (ghosts+strays).", offenders)
        raise SystemExit(1)
    else:
        logger.info("✅ No taxonomy mismatches found.")
        raise SystemExit(0)
