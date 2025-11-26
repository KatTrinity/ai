
# C:\dev\GovernEdge_CLI\engine_query\engine_load_models.py 
""" 
This module is the central loader for embedding models and Ollama LLMs. 
It resolves short config keys from Config into their full model names, 
lazy-loads the appropriate HuggingFace embedding or Ollama client, 
and caches them to avoid re-initialization.
Logging is built in so you can see when keys are resolved, models are loaded, 
and cached clients are reused.
"""

from __future__ import annotations
import os, logging
#import custom_st
from typing import Any, Dict, Optional
from config_base.config import Config
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama as OllamaLLM
from utils.utils_custom_normalizer import NormalizedEmbeddingWrapper
from engine_query.engine_cross_encoder import apply_query_prefix_for_embedder

# --- Logging setup ---
#logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)

logger = logging.getLogger(__name__)
#if not logger.handlers:
    #handler = logging.StreamHandler()
    #formatter = logging.Formatter(
        #"%(asctime)s [%(levelname)s] %(name)s: %(message)s", "%H:%M:%S"
    #)
    #handler.setFormatter(formatter)
    #logger.addHandler(handler)
#logger.setLevel(logging.INFO)

# -------------------------------------------------------------------
# Embeddings
# -------------------------------------------------------------------
def resolve_embed_config(model_key: str) -> tuple[str, str]:
    """
    Returns (embed_model_name, persist_dir).
    Accepts either a short key ('nomic') or a full HF id ('nomic-ai/nomic-embed-text-v1.5').
    """
    key = (model_key or "").strip()
    if "/" in key:  # treat as full HF id
        model_name = key
        persist_dir = os.path.join("vectorstores", key.split("/")[-1])
        logger.info(f"🔑 Using explicit HF id '{model_name}'")
        return model_name, persist_dir

    try:
        model_name = Config.EMBEDDING_MODEL_MAP[key]
        logger.info(f"🔑 Resolved embed key '{key}' -> '{model_name}'")
    except KeyError as e:
        valid = ", ".join(Config.EMBEDDING_MODEL_MAP.keys())
        logger.error(f"❌ Unsupported embed key: {key}. Valid keys: {valid} or pass a full HF repo id")
        raise

    persist_dir = os.path.join("vectorstores", key)
    return model_name, persist_dir

def load_embedding_model(name: str):
    n = name.lower()
    logger.info(f"⚙️ Loading embedding model: {name}")

    # Common kwargs
    encode_kwargs = {"normalize_embeddings": True}
    model_kwargs = {"device": "cpu"}

    if "nomic" in n:
        model_kwargs["trust_remote_code"] = True
        base = HuggingFaceEmbeddings(
            model_name=name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )
    elif "snowflake" in n or "arctic" in n or "snow" in n:
        model_kwargs["trust_remote_code"] = True
        base = HuggingFaceEmbeddings(
            model_name=name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )
        
    elif "minilm" in n:
        base = HuggingFaceEmbeddings(
            model_name=name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )

    elif "bge_m3" in n:
        base = HuggingFaceEmbeddings(
            model_name=name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )

    elif "jina" in n:
        model_kwargs["trust_remote_code"] = True
        base = HuggingFaceEmbeddings(
            model_name=name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )

    else:
        # Allow any full repo id to pass through (don’t hard-fail on unknowns)
        base = HuggingFaceEmbeddings(
            model_name=name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )

    logger.info(f"✅ Embedding model '{name}' loaded successfully")
    return NormalizedEmbeddingWrapper(base)

def get_embedder(cache: Dict[str, Any], embed_model_name: str) -> Embeddings:
    """Lazy-load + cache by FULL model name to avoid duplicates."""
    if embed_model_name not in cache:
        logger.info(f"🆕 Caching embedder for '{embed_model_name}'")
        cache[embed_model_name] = load_embedding_model(embed_model_name)
    else:
        logger.debug(f"♻️ Using cached embedder for '{embed_model_name}'")
    return cache[embed_model_name]

# -------------------------------------------------------------------
# Ollama LLM
# -------------------------------------------------------------------
def resolve_llm_name(llm_key: str) -> str:
    try:
        model_name = Config.LLM_MODEL_MAP[llm_key]
        logger.info(f"🔑 Resolved llm key '{llm_key}' -> '{model_name}'")
        return model_name
    except KeyError:
        logger.error(f"❌ Unsupported llm key: {llm_key}")
        raise

def load_ollama(model_name: str, base_url: str = "http://localhost:11434"): 
    logger.info(f"⚙️ Loading Ollama model '{model_name}' from {base_url}")
    return OllamaLLM(model=model_name, base_url=base_url)

def get_llm(cache: Dict[str, Any], llm_model_name: str,
            base_url: Optional[str] = None):
    """Lazy-load + cache Ollama LLM client."""
    key = (llm_model_name, base_url or "http://localhost:11434")
    if key not in cache:
        logger.info(f"🆕 Creating Ollama client for '{llm_model_name}' @ {key[1]}")
        cache[key] = load_ollama(llm_model_name, base_url=key[1])
    else:
        logger.debug(f"♻️ Using cached Ollama client for '{llm_model_name}' @ {key[1]}")
    return cache[key]


# -------------------------------------------------------------------
# Exports
# -------------------------------------------------------------------
__all__ = [
    "get_embedder", "get_llm",
    "resolve_embed_config", "resolve_llm_name",
    "apply_query_prefix_for_embedder",
]
