
# C:\dev\GovernEdge_CLI\config_base\config.py

import os
import sys
from pathlib import Path  

# ✅ Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

class Config:

# ── LLM Settings ─────────────────────────────
    LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434")
    llm = None  # Placeholder for runtime LLM instance

# ── Databases  ─────────────────────────────
    DB_PATH = Path(os.getenv(
        "GOVERNEDGE_DB_PATH", r"C:\dev\GovernEdge_CLI\database\chat_logs.sqlite" 
    ))

    MASTER_DB_PATH = Path(os.getenv(
        "GOVERNEDGE_MASTER_DB_PATH", r"C:\dev\GovernEdge_CLI\database\master_data.duckdb"
    ))
    

    # (optional but recommended) other sane defaults so getattr() never explodes
    FTS_TABLE = "doc_chunks_fts"
    # CHUNKS_TABLE = "doc_chunks"
    FUSION_WEIGHTS = {"dense": 0.55, "fts": 0.30, "sql": 0.15} 
    # RRF_K = 60

# ── Embedding Model Settings ─────────────────
    EMBEDDING_MODEL_MAP = {

        #8192-Token General-Purpose
        "jina": "jinaai/jina-embeddings-v3",
        "bge_m3": "BAAI/bge-m3",

        #512 ish token
        "snow": "Snowflake/snowflake-arctic-embed-m",      
        "nomic": "nomic-ai/nomic-embed-text-v1.5",          
        "minilm": "sentence-transformers/all-MiniLM-L12-v2", 

        # BGE (classic)
        #"bge_base": "BAAI/bge-base-en-v1.5",
        #"bge_large": "BAAI/bge-large-en-v1.5",

        

    }

# ── LLM Model Settings ───────────────── 
    LLM_MODEL_MAP = {
       
        # Tooling
        "qwen3" : "qwen3:8b",
        "mistral": "mistral:7b",
        "hermes3":"hermes3:8b",
        "cogito" : "cogito:8b",
        "qwen3:4b-thinking": "qwen3:4b-thinking", 
        "qwen3:4b-thinking-2507-q4_K_M" : "qwen3:4b-thinking-2507-q4_K_M",  
        "qwen3:1.7b-q4_K_M" : "qwen3:1.7b-q4_K_M",
        "qwen3:4b-instruct-2507-q4_K_M" : "qwen3:4b-instruct-2507-q4_K_M",
        
        # No Tooling
        "openhermes": "openhermes:v2.5",
        "comm-r35b" : "command-r:35b-08-2024-q4_K_M",
        "comm-r7b" : "command-r7b:latest",

        "tinyllama": "tinyllama", # speed/smoke tests ONLY
    }

# ── Cross Encorder Model Settings ─────────────────
    CE_MODEL_MAP = {

        #8192-Token General-Purpose
        "bge_rerank": "BAAI/bge-reranker-v2-m3",
        "jina_rerank": "jinaai/jina-reranker-v3",


        "bge_base":   "BAAI/bge-reranker-base",                          
        "minilm_l12": "cross-encoder/ms-marco-MiniLM-L-12-v2", # middle ground between speed and precision
        "none":       None,                           # disables rerank

        #-------------------------
        "minilm_l6":  "cross-encoder/ms-marco-MiniLM-L-6-v2",  # speed/smoke tests ONLY
    }
    CE_DEFAULT_KEY = "bge_base"  # or env override if you want 


# ── Vectorstore Paths by Model ───────────────
    CHROMA_COLLECTION_PREFIX = "my_docs_tst__"  
    CHROMA_DIR_MAP = {
        "nomic": os.getenv("CHROMA_DIR_NOMIC", "vectorstores/nomic"),
        "minilm": os.getenv("CHROMA_DIR_MINILM", "vectorstores/minilm"),
        "snow" : os.getenv("CHROMA_DIR_SNOW", "vectorstores/snow"),
        "jina" : os.getenv("CHROMA_DIR_JINA", "vectorstores/jina"),
        "bge_m3" : os.getenv("CHROMA_DIR_BGE_M3", "vectorstores/bge_m3"),
    }

# ── Document Chunking ────────────────────────
    CHUNK_DIR = os.getenv("CHUNK_DIR", "data_tst\sap_docs")
    CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "my_docs_tst")
    CHROMA_REBUILD = os.getenv("CHROMA_REBUILD", "false").strip().lower() == "true"

    SYSTEM_PROMPT_TEXT = "@prompts/queryengine_system.txt"


    def __init__(self):
        print("✅ Config loaded successfully.")

config = Config() 