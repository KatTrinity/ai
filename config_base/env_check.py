import os, logging
from pathlib import Path
from config_base.config import Config

logger = logging.getLogger(__name__)


def _embed_env_key(key: str) -> str:
    """
    Turn an embed key like 'bge-m3' into a safe env var name:
    'CHROMA_DIR_BGE_M3'.
    """
    safe = "".join(ch if ch.isalnum() else "_" for ch in key.upper())
    return f"CHROMA_DIR_{safe}"


def validate_env(create_vector_dirs: bool = False) -> None:
    """
    Validate environment + Config coherence.

    - Ensures core env vars are set.
    - Ensures every embed key has a usable vectorstore directory
      (env override OR Config.CHROMA_DIR_MAP OR 'vectorstores/{key}').
    - Checks CE_DEFAULT_KEY is in CE_MODEL_MAP.
    - Warns if maps are empty.
    - Optionally creates missing vectorstore dirs.
    Raises EnvironmentError on hard failures.
    """

    hard_missing: list[str] = []

    # ── 1) Required generics (only things that truly must be env vars) ──
    required_env = [
        "LLM_BASE_URL",          # used by your Ollama/client
        "CHUNK_DIR",
        "CHROMA_COLLECTION_NAME",
    ]

    for key in required_env:
        if not os.getenv(key):
            hard_missing.append(key)

    # ── 2) Config map sanity ──
    llm_map = getattr(Config, "LLM_MODEL_MAP", {})
    emb_map = getattr(Config, "EMBEDDING_MODEL_MAP", {})
    ce_map  = getattr(Config, "CE_MODEL_MAP", {})
    ce_def  = getattr(Config, "CE_DEFAULT_KEY", None)

    if not llm_map:
        hard_missing.append("Config.LLM_MODEL_MAP (empty)")
    if not emb_map:
        hard_missing.append("Config.EMBEDDING_MODEL_MAP (empty)")
    if not ce_map:
        hard_missing.append("Config.CE_MODEL_MAP (empty)")
    if ce_def and ce_def not in ce_map:
        hard_missing.append(f"Config.CE_DEFAULT_KEY='{ce_def}' not in CE_MODEL_MAP")

    if hard_missing:
        lines = "\n  - ".join(hard_missing)
        raise EnvironmentError(f"❌ Missing configuration:\n  - {lines}")

    # ── 3) CHUNK_DIR existence ──
    chunk_dir = Path(os.getenv("CHUNK_DIR"))
    if not chunk_dir.exists():
        logger.warning(f"⚠️ CHUNK_DIR does not exist: {chunk_dir}")

    # ── 4) Vectorstore dirs per embedding key ──
    embed_keys = list(emb_map.keys())
    chroma_dir_map = getattr(Config, "CHROMA_DIR_MAP", {})

    for k in embed_keys:
        env_key = _embed_env_key(k)
        # 1) env override, 2) Config.CHROMA_DIR_MAP, 3) default path
        raw = os.getenv(env_key)
        dir_str = raw or chroma_dir_map.get(k) or str(Path("vectorstores") / k)
        p = Path(dir_str)

        if raw:
            logger.debug(f"Using {env_key}={raw} for embed key '{k}'")
        elif k in chroma_dir_map:
            logger.debug(f"Using Config.CHROMA_DIR_MAP['{k}']={dir_str}")
        else:
            logger.debug(f"No env/Config dir for '{k}', defaulting to {dir_str}")

        if not p.exists():
            if create_vector_dirs:
                try:
                    p.mkdir(parents=True, exist_ok=True)
                    logger.info(f"📁 Created vectorstore dir for '{k}': {p}")
                except Exception as e:
                    logger.warning(f"⚠️ Could not create {p}: {e}")
            else:
                logger.warning(f"⚠️ Vectorstore dir missing for '{k}': {p}")

    logger.info("✅ Environment + Config validated successfully.")
