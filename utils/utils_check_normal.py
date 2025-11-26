
# C:\dev\GovernEdge_CLI\utils\utils_check_normal.py

"""

Utility script to validate Chroma vectorstore health:
  - Embedding norms are ~1.0
  - Cosine vs dot product consistency
  - Self-similarity ≈ 1.0
  - Nearest-neighbor similarity < ~0.99
  - Potential duplicate detection

Can be run across multiple vectorstore folders.
"""

import argparse, sys, random, logging
import numpy as np
import chromadb

# --- logging config ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("chroma_check")

# --- Default folders (adjust to your env) ---
BASE_FOLDERS_DEFAULT = [
    r"C:\dev\GovernEdge_CLI\vectorstores\snow",
    r"C:\dev\GovernEdge_CLI\vectorstores\nomic",
    r"C:\dev\GovernEdge_CLI\vectorstores\minilm",
]

# ---------------- Norms & Cosine helpers ----------------
def check_norms(embeddings, atol=1e-3):
    """
    Verify that embeddings are unit-normalized (L2 norm ≈ 1.0).
    Returns (is_normal, norms).
    """
    norms = np.linalg.norm(embeddings, axis=1)
    is_normal = np.allclose(norms, 1.0, atol=atol)
    return is_normal, norms

def cosine(a, b):
    """
    Cosine similarity between two vectors (safe against divide-by-zero).
    """
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

# ---------------- Deep validation ----------------
def deep_checks(col, limit=512, sample=64):
    """
    Extra validation beyond norms:
      - Cosine vs dot-product agreement (should be ~0 diff for unit vectors)
      - Self-similarity ≈ 1.0
      - Nearest-neighbor similarity < ~0.99 on average
      - Duplicate-rate estimate (pairs > 0.999 sim)
    """
    try:
        results = col.peek(limit=limit)
    except Exception as e:
        logger.warning("  (deep) peek failed: %s", e)
        return

    embs = np.array(results.get("embeddings", []))
    ids  = results.get("ids", [])
    n = len(embs)

    if n == 0:
        logger.warning("  (deep) ⚠️ No embeddings to test.")
        return
    if n == 1:
        logger.info("  (deep) Only one embedding; skipping pairwise checks.")
        return

    # sample subset
    k = min(sample, n)
    idx = random.sample(range(n), k=k)

    # 1) Cosine vs dot on normalized vectors
    max_abs_diff = 0.0
    outer = max(2, min(k, 16))
    inner = max(2, min(k, 16))
    for i in idx[:outer]:
        for j in idx[:inner]:
            c = cosine(embs[i], embs[j])
            d = float(np.dot(embs[i], embs[j]))
            max_abs_diff = max(max_abs_diff, abs(c - d))
    logger.info("  (deep) cos vs dot max|Δ|: %.2e  %s",
                max_abs_diff, "✅" if max_abs_diff < 1e-6 else "❌")

    # 2) Self-similarity & nearest-neighbor probe
    highs, seconds, dup_count = [], [], 0
    for i in idx:
        sims = embs @ embs[i]  # dot == cosine for normalized vectors
        order = np.argsort(-sims)
        highs.append(float(sims[i]))
        # next best (skip self at pos 0)
        nxt = next(k for k in order if k != i)
        s2 = float(sims[nxt])
        seconds.append(s2)
        if s2 > 0.999:
            dup_count += 1

    logger.info("  (deep) self-sim mean: %.4f (expect ~1.0000)", np.mean(highs))
    logger.info("  (deep) nn-sim  mean: %.4f (expect < 0.99)", np.mean(seconds))
    if dup_count:
        rate = dup_count / len(seconds)
        logger.warning("  (deep) potential duplicates (>0.999 sim): %d/%d (%.1f%%)",
                       dup_count, len(seconds), rate*100)
    else:
        logger.info("  (deep) potential duplicates (>0.999 sim): 0")

# ---------------- Folder-level scan ----------------
def check_chroma_folder(folder, do_deep=False):
    """
    Scan one vectorstore folder and run checks across all collections.
    Returns True if all collections look normalized, False otherwise.
    """
    logger.info("📁 Scanning: %s", folder)
    try:
        client = chromadb.PersistentClient(path=folder)
        collections = client.list_collections()
        if not collections:
            logger.warning("  ⚠️ No collections found.")
            return True  # not an error; just empty

        ok = True
        for col in collections:
            name = col.name
            logger.info("  📦 Collection: %s", name)
            col = client.get_collection(name)
            results = col.peek(limit=5000)
            embeddings = np.array(results.get("embeddings", []))

            if embeddings.size == 0:
                logger.warning("    ⚠️ No embeddings found.")
                continue

            is_norm, norms = check_norms(embeddings)
            logger.info("    → Norms | Min: %.4f | Max: %.4f | Mean: %.4f",
                        norms.min(), norms.max(), norms.mean())
            logger.info("    %s", "✅ Normalized." if is_norm else "❌ NOT normalized.")
            ok = ok and is_norm

            if do_deep:
                deep_checks(col)

        return ok
    except Exception as e:
        logger.error("  ❌ Failed to scan %s: %s", folder, e)
        return False

# ---------------- CLI ----------------
def main():
    ap = argparse.ArgumentParser(description="Validate norms and consistency of Chroma vectorstores.")
    ap.add_argument("--folders", nargs="*", default=BASE_FOLDERS_DEFAULT,
                    help="Vectorstore folders to scan")
    ap.add_argument("--deep", action="store_true",
                    help="Run deep checks (cos vs dot, NN, dup-rate)")
    args = ap.parse_args()

    all_ok = True
    for folder in args.folders:
        ok = check_chroma_folder(folder, do_deep=args.deep)
        all_ok = all_ok and ok

    # Exit non-zero if any collection had problems
    sys.exit(0 if all_ok else 1)

if __name__ == "__main__":
    main()
