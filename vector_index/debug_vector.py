
# C:\dev\GovernEdge_CLI\vector_index\debug_vector.py

#!/usr/bin/env python3
"""
debug_vectorstore.py

Utility script to connect to a Chroma vectorstore and run a simple query.
- Supports configurable embedding models.
- Prints top-K retrieved chunks and their metadata.
"""

import sys
import logging
from langchain_community.vectorstores import Chroma
from langchain.vectorstores.base import VectorStoreRetriever
from langchain_community.embeddings import HuggingFaceEmbeddings

# ---------------- logging ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("debug_vectorstore")

# ---------------- defaults ----------------
DEFAULT_CHROMA_PATH = "C:/dev/GovernEdge_CLI/vectorstores/nomic"
DEFAULT_EMBED_MODEL = "nomic-ai/nomic-embed-text-v1"

# ---------------- helpers ----------------
def connect_vectorstore(embed_model_name: str, chroma_path: str = DEFAULT_CHROMA_PATH) -> Chroma:
    """
    Connect to a Chroma vectorstore with the specified embedding model.

    Args:
        embed_model_name: name of the HuggingFace embedding model to use.
        chroma_path: path to the persisted Chroma DB directory.

    Returns:
        A Chroma vectorstore instance.
    """
    logger.info("🔧 Using embedder: %s", embed_model_name)

    # Handle models that require trust_remote_code (e.g. nomic)
    if embed_model_name == "nomic-ai/nomic-embed-text-v1":
        embeddings = HuggingFaceEmbeddings(
            model_name=embed_model_name,
            model_kwargs={"trust_remote_code": True},
        )
    else:
        embeddings = HuggingFaceEmbeddings(model_name=embed_model_name)

    logger.info("📂 Connecting to Chroma at: %s", chroma_path)
    return Chroma(persist_directory=chroma_path, embedding_function=embeddings)


def test_vectorstore(query: str, embed_model_name: str, k: int = 5):
    """
    Run a simple retrieval test on a vectorstore.

    Args:
        query: natural language query string.
        embed_model_name: embedding model to use.
        k: number of results to retrieve.
    """
    logger.info("Running test query='%s' with top-k=%d", query, k)

    # Connect + get retriever
    vectorstore = connect_vectorstore(embed_model_name)
    retriever: VectorStoreRetriever = vectorstore.as_retriever(search_kwargs={"k": k})

    try:
        results = retriever.get_relevant_documents(query)
    except Exception as e:
        logger.error("❌ Retrieval failed: %s", e)
        return

    logger.info("Retrieved %d documents", len(results))

    # Pretty-print results
    print(f"\n🔍 Top {k} results for query: '{query}'\n")
    for i, doc in enumerate(results, start=1):
        print(f"--- Result {i} ---")
        print("📄 Chunk Preview:", doc.page_content[:150].replace("\n", " ") + "...")
        print("🧷 Metadata:", doc.metadata)


# ---------------- CLI ----------------
if __name__ == "__main__":
    # Usage: python debug_vectorstore.py "your query" "embed_model"
    query = sys.argv[1] if len(sys.argv) > 1 else "test"
    embed_model = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_EMBED_MODEL

    test_vectorstore(query, embed_model)
