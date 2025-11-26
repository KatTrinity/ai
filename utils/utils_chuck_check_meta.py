# --- check_index.py ---
import os, sys

# ✅ Add project root to the path for reliable imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import argparse
import logging
from llm_core_tst.models_embedding_tst.get_embed_model import get_embedding_components
from llm_core_tst.utils_tst.utils_k_dense import resolve_collection_name
from langchain_community.vectorstores import Chroma  # keep consistent with your builder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)

def apply_query_prefix_for_embedder(q: str, embed_model_name: str) -> str:
    name = embed_model_name.lower()
    if "nomic" in name or "snow" in name:
        return f"Query: {q}"
    return q

def main(model_key: str, query: str, k: int):
    # 1) load embedding + persist_dir exactly like the builder
    embeddings, persist_dir = get_embedding_components(model_key)
    collection_name = resolve_collection_name(model_key)  # e.g. f"my_docs_tst__{model_key}"

    log.info(f"📦 Vectorstore dir: {persist_dir}")
    log.info(f"📚 Collection: {collection_name}")
    log.info(f"🧠 Embedding key: {model_key}")

    # 2) open store
    vs = Chroma(
        persist_directory=persist_dir,
        collection_name=collection_name,
        embedding_function=embeddings,
    )

    # 3) prefix query only if that embedder expects it
    q_dense = apply_query_prefix_for_embedder(query, getattr(embeddings, "model_name", str(embeddings)))

    log.info(f"🔎 Similarity search: {query!r} (k={k})")
    try:
        results = vs.similarity_search_with_score(q_dense, k=k)
        docs = [d for d, _ in results]
        scores = [float(s) for _, s in results]
    except Exception:
        docs = vs.similarity_search(q_dense, k=k)
        scores = [None] * len(docs)

    if not docs:
        log.warning("⚠️ No documents returned.")
        return

    for i, d in enumerate(docs, 1):
        m = d.metadata or {}
        preview = (d.page_content or "")[:220].replace("\n", " ").strip()
        log.info(
            "\n📄 Result #%d\n"
            "   score        : %s\n"
            "   doc_id       : %s\n"
            "   chunk_id     : %s\n"
            "   section_id   : %s\n"
            "   header_path  : %s\n"
            "   title        : %s\n"
            "   file_path    : %s\n"
            "   span         : %s–%s\n"
            "   preview      : %s\n%s",
            i,
            f"{scores[i-1]:.4f}" if scores[i-1] is not None else "n/a",
            m.get("doc_id"),
            m.get("chunk_id"),
            m.get("section_id"),
            m.get("header_path"),
            m.get("title"),
            m.get("file_path"),
            m.get("span_start"),
            m.get("span_end"),
            preview,
            "-" * 80,
        )

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_key", required=True, help="embedding model key (matches your builder)")
    ap.add_argument("--query", default="test")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()
    main(args.model_key, args.query, args.k)
