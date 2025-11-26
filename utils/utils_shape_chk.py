# tools/shape_check.py
from collections import Counter
import re
from llm_core_tst.models_embedding_tst.get_embed_model import get_embedding_components
from llm_core_tst.utils_tst.utils_k_dense import resolve_collection_name
from langchain_community.vectorstores import Chroma

SNAKE_RX = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")

def shape_check(model_key="minilm_tst", peek=5000):
    embeddings, persist_dir = get_embedding_components(model_key)
    coll_name = resolve_collection_name(model_key)
    vs = Chroma(collection_name=coll_name, persist_directory=persist_dir, embedding_function=embeddings)

    bad = {k: [] for k in ("action","object","category")}
    nulls = Counter()

    batch = vs._collection.peek(limit=peek)
    for m in batch.get("metadatas", []):
        for k in ("action","object","category"):
            v = (m or {}).get(k)
            if v is None:
                nulls[k] += 1
                continue
            s = str(v).strip()
            if not SNAKE_RX.match(s):
                bad[k].append(s)

    for k in ("action","object","category"):
        uniq_bad = sorted(set(bad[k]))
        print(f"\n[{k}] null count: {nulls[k]}")
        print(f"[{k}] non-snake-case values: {uniq_bad if uniq_bad else '—'}")

if __name__ == "__main__":
    shape_check()
