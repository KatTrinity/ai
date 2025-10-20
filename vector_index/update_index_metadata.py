import os
from pathlib import Path
from chromadb import PersistentClient
from taxonomies.apply_taxonomy import TaxonomyMatcher

TAXONOMY_DIR = Path(r"C:\dev\GovernEdge_CLI\taxonomies")
matcher = TaxonomyMatcher.from_folder(TAXONOMY_DIR)

client = PersistentClient(path="./vectorstores/chroma_store")

CHROMA_DIR_MAP = {
        "nomic": os.getenv("CHROMA_DIR_NOMIC", "vectorstores/nomic"),
        "minilm": os.getenv("CHROMA_DIR_MINILM", "vectorstores/minilm"),
        "snow" : os.getenv("CHROMA_DIR_SNOW", "vectorstores/snow")
    }

for coll_name in CHROMA_DIR_MAP:
    coll = client.get_collection(name=coll_name)
    results = coll.get(include=["metadatas", "documents"], limit=999999)

    ids = []
    new_metas = []
    for doc_id, meta, doc_text in zip(results["ids"], results["metadatas"], results["documents"]):
        tags = matcher.annotate_metadata(doc_text, fm=meta)
        meta.update({k: v for k, v in tags.items() if v is not None})
        ids.append(doc_id)
        new_metas.append(meta)

    coll.update(ids=ids, metadatas=new_metas)
    print(f"✅ {coll_name}: updated {len(ids)} docs")

