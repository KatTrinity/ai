
#C:\dev\GovernEdge_CLI\vector_index\vector_index_builder.py

# Run Commands 
# python vector_index\vector_index_builder.py --model_key nomic
# python -m vector_index.vector_index_builder --model_key snow
# python -m vector_index.vector_index_builder --model_key minilm
# python -m vector_index.vector_index_builder --all

"""
Build Chroma vector indexes from spaCy-cleaned chunks stored in SQLite.
For each embedder model key, this script (re)embeds only the chunks whose content-hash
has changed, then upserts sanitized Documents into a per-model Chroma collection.
It keeps collection naming consistent via resolve_collection_name() and logs progress clearly.
"""

import os, sys, json, sqlite3, logging, argparse, hashlib
from contextlib import closing
from pathlib import Path 

# Ensure project root is importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from dotenv import load_dotenv
from config_base.config import Config
from config_base.env_check import validate_env
from langchain_core.documents import Document
from prepare_docs.db_io import ensure_db, get_conn   # ✅ use get_conn for consistent PRAGMAs/schema
from langchain_chroma import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from engine_query.engine_load_models import load_embedding_model, get_embedder, resolve_embed_config
from utils.utils_k_dense import resolve_collection_name  # ✅ single source of truth for collection names
from typing import Dict, Any



# --- Env & logging ----------------------------------------------------------
load_dotenv()
validate_env()



logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)
logger = logging.getLogger("vector_index_builder")
logger.info("📦 EMBEDDING_MODEL_MAP (effective): %r", getattr(Config, "EMBEDDING_MODEL_MAP", {}))
logger.info("🔧 sys.argv: %r", sys.argv)

# --- Config ----------------------------------------------------------------
DB_PATH = Path(getattr(Config, "DB_PATH", "database/chat_logs.sqlite"))
SCHEMA_VERSION = "kb.v1"

def resolve_persist_dir(model_key: str) -> str:
    """Resolve the Chroma persist directory for a given model key (Config first, then sensible fallback)."""
    d = getattr(Config, "CHROMA_DIR_MAP", {}).get(model_key)
    return d if d else f"vectorstores/{model_key}"

# Try to pull the embedder map from config; fall back to ENV or a local dict.
def load_embedding_model_map() -> dict[str, str]:
    try:
        from config_base.config import EMBEDDING_MODEL_MAP  # adjust path/name if needed
        return dict(EMBEDDING_MODEL_MAP)
    except Exception:
        env_json = os.getenv("EMBEDDING_MODEL_MAP_JSON")
        if env_json:
            return json.loads(env_json)
        return {
            "nomic":  "nomic-ai/nomic-embed-text-v1",
            "snow":   "cnmoro/snowflake-arctic-embed-m-v2.0-cpu",
            "minilm": "sentence-transformers/all-MiniLM-L12-v2",
        }

# --- DB fetch ---------------------------------------------------------------
def fetch_chunks(conn: sqlite3.Connection, nlp_version: str):
    sql = """
    SELECT
      c.chunk_id,
      c.doc_id,
      c.header_path,
      c.title                     AS chunk_title,
      n.cleaned_text              AS body_text,
      n.text_hash                 AS text_hash,
      i.file_name,
      i.file_path,
      COALESCE(json_extract(i.fm_json, '$.title'), i.title) AS doc_title,
      i.fm_json,
      c.action,
      c.object,
      c.category
    FROM doc_chunks c
    JOIN doc_ingest i ON i.doc_id = c.doc_id
    JOIN doc_nlp_cache n  ON n.chunk_id = c.chunk_id AND n.nlp_version = ?
    ORDER BY c.doc_id, c.chunk_id
    """
    return conn.execute(sql, (nlp_version,)).fetchall()

# --- Gating & bookkeeping ---------------------------------------------------
def hash_model_chunk(schema_version: str, model_key: str, text_hash: str) -> str:
    h = hashlib.sha256()
    h.update(schema_version.encode()); h.update(b"\x00")
    h.update(model_key.encode());      h.update(b"\x00")
    h.update(text_hash.encode())
    return h.hexdigest()

def chunk_needs_embed(conn: sqlite3.Connection, chunk_id: str, model_key: str, gate_hash: str) -> bool:
    row = conn.execute(
        "SELECT content_hash FROM chunk_embedding_state WHERE chunk_id=? AND model_key=?",
        (chunk_id, model_key)
    ).fetchone()
    return (row is None) or (row[0] != gate_hash)

def mark_chunk_embedded(conn: sqlite3.Connection, chunk_id: str, model_key: str, gate_hash: str) -> None:
    conn.execute("""
      INSERT INTO chunk_embedding_state (chunk_id, model_key, content_hash, embedded_at)
      VALUES (?, ?, ?, CURRENT_TIMESTAMP)
      ON CONFLICT(chunk_id, model_key) DO UPDATE SET
        content_hash=excluded.content_hash,
        embedded_at=CURRENT_TIMESTAMP
    """, (chunk_id, model_key, gate_hash))

# --- Text prefixing for certain embedders ----------------------------------
def apply_doc_prefix(text: str, model_key: str) -> str:
    # Only for the embed text; keeps DB/FTS untouched
    if any(k in model_key for k in ["nomic", "snow"]):
        return "Document: " + text
    return text

# --- Core build -------------------------------------------------------------
GLOBAL_EMB_CACHE: Dict[str, Any] = {}
def build_index_for(model_key: str, nlp_version: str) -> None:
    logger.info("🔧 Building vector index for model: %s (nlp=%s)", model_key, nlp_version)

    # 1) Resolve once
    model_name, persist_dir = resolve_embed_config(model_key)
    logger.info("🔧 Building vector index for key='%s' → model='%s' (nlp=%s)",
                model_key, model_name, nlp_version)
    logger.info("📂 Persist dir: %s", persist_dir)

    # 2) Get (or create) the embedder for the **resolved repo id**
    embedder = get_embedder(GLOBAL_EMB_CACHE, model_name)

    # 3) Ensure vectorstore directory exists
    os.makedirs(persist_dir, exist_ok=True)

    # ✅ Use get_conn so PRAGMAs and schema are consistent
    with closing(get_conn(DB_PATH)) as conn:
        ensure_db(conn)  # ensure tables exist
        conn.row_factory = sqlite3.Row

        rows = fetch_chunks(conn, nlp_version=nlp_version)
        if not rows:
            logger.info("✅ No chunks found. Did you run ingest + spaCy cache?")
            return

        docs_to_upsert: list[Document] = []
        ids: list[str] = []

        for row in rows:
            body = row["body_text"] or ""
            text_hash = (row["text_hash"] or "").strip()
            if not body or not text_hash:
                continue

            fm = {}
            if row["fm_json"]:
                try:
                    fm = json.loads(row["fm_json"])
                except Exception:
                    fm = {}

            gate = hash_model_chunk(SCHEMA_VERSION, model_key, text_hash)
            if not chunk_needs_embed(conn, row["chunk_id"], model_key, gate):
                continue

            title = (row["doc_title"] or row["chunk_title"] or row["file_name"] or "").strip()[:300]

            meta = {
                "chunk_id": row["chunk_id"],
                "doc_id":   row["doc_id"],
                "file_path": row["file_path"],
                "file_name": row["file_name"],
                "title":     title,
                "header_path": row["header_path"] or "",
                "schema_version": SCHEMA_VERSION,
                "model_key": model_key,
                "chunk_hash_clean": text_hash,   # hash of CLEANED text
                "action": row["action"],
                "object": row["object"],
                "category": row["category"],
                "front_matter": fm,
                "_gate": gate,
            }

            doc = Document(page_content=apply_doc_prefix(body, model_key), metadata=meta)
            docs_to_upsert.append(doc)
            ids.append(row["chunk_id"])

        if not docs_to_upsert:
            logger.info("✅ Nothing to (re)embed for this model.")
            return

        # Move FM to scalar fm_json for Chroma and drop dict
        for d in docs_to_upsert:
            fm = d.metadata.get("front_matter")
            if fm is not None:
                d.metadata["fm_json"] = json.dumps(fm, ensure_ascii=False)
                d.metadata.pop("front_matter", None)

        sanitized_docs = filter_complex_metadata(docs_to_upsert)

        # ✅ Use resolve_collection_name for a single, consistent naming scheme
        coll_name = resolve_collection_name(model_key)
        store = Chroma(
            collection_name=coll_name,
            embedding_function=embedder,
            persist_directory=persist_dir,
        )

        # Idempotent replace
        try:
            store.delete(ids=ids)
        except Exception:
            pass

        store.add_documents(documents=sanitized_docs, ids=ids)
        logger.info("🧠 Upserted %d chunks → %s", len(ids), coll_name)

        # Mark success
        for d in sanitized_docs:
            mark_chunk_embedded(conn, d.metadata["chunk_id"], model_key, d.metadata["_gate"])
        conn.commit()
        logger.info("✅ Persist dir: %s", persist_dir)

# --- CLI -------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="Build vector indexes from spaCy-cleaned chunks.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--model_key", help="Single model key (e.g., minilm)")
    g.add_argument("--all", action="store_true", help="Build for all models in EMBEDDING_MODEL_MAP")
    ap.add_argument("--nlp-version", default="spacy.v1", help="NLP cache version (must match ingest)")
    args = ap.parse_args(argv)

    model_map = load_embedding_model_map()

    if args.all:
        for key in model_map.keys():
            build_index_for(key, nlp_version=args.nlp_version)
        logger.info("🎉 all models done")
    else:
        if args.model_key not in model_map:
            raise SystemExit(f"Unknown --model_key '{args.model_key}'. Known: {list(model_map)}")
        build_index_for(args.model_key, nlp_version=args.nlp_version)

if __name__ == "__main__":
    main()
