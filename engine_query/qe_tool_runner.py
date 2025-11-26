# qe_tool_runner.py 
import uuid, logging
from config_base.config_k import KConfig
from engine_query.query_engine import QueryEngine
from metrics.doc_freshness_update import chk_freshness_metrics
from vectorstore_mem.mem_vector import (
    insert_memory,
    upsert_memory_vector,
    retrieve_user_memory,
)
from config_base.logging_config import setup_logging
setup_logging()

logger = logging.getLogger(__name__)
logger.info("Point")
engine = QueryEngine.from_env()

DEFAULT_USER_ID = 1

def as_text(msg) -> str: 
    # Works for str, LangChain BaseMessage/AIMessage, or dict-like
    try:
        from langchain_core.messages import BaseMessage
        if isinstance(msg, BaseMessage):
            return msg.content or ""
    except Exception:
        pass
    if isinstance(msg, dict) and "content" in msg:
        logger.info("Point")
        return str(msg.get("content", ""))
    return str(msg or "")

def run_query_engine(
    question: str,
    *,
    llm_key: str,
    embed_key: str,
    ce_key: str,
    session_id: str | None = None,
    prefilter_mode: str | None = None,
) -> dict:
    """
    One-stop call that:
      1) Prefilters
      2) Computes k
      3) Runs search
    and returns both answer + metadata. 
    """
    if session_id is None:
        session_id = f"session_{uuid.uuid4()}"
    
    chk_freshness_metrics()

    logger.info("Point")

    user_id = DEFAULT_USER_ID

    # Could make this smarter later (length, punctuation, etc.)
    messy = len(question.split()) < KConfig.MESSY_WORDS_LT

    # 1) Resolve facet filter (use config default if None)
    pf_mode = prefilter_mode or KConfig.PREFILTER_MODE
    prefiltered, facet_filter, mode_hint = engine.prefilter(question, pf_mode)

    # 2) Compute k
    k_used, corpus_n = engine.compute_k(
        embed_key=embed_key,
        messy=messy,
        prefiltered=prefiltered,
        pct=KConfig.PCT,
        cap=KConfig.CAP,
        floor=KConfig.FLOOR,
    )
    logger.info("Point")

    mem_docs = retrieve_user_memory(
        user_id=user_id,
        query=question,
        k=5,
    )
    logger.info("Point")
    # 3) Search
    response, turn_id, store_path, docs, latency_ms = engine.search(
        question=question,
        llm_key=llm_key,
        embed_key=embed_key,
        ce_key=ce_key,
        k=int(k_used),
        session_id=session_id,
        facet_filter=facet_filter,
    )

    # Replace Later
    #candidate_memories = extract_new_memories(
    #user_message=question,
    #assistant_message=as_text(response),
#)

    candidate_memories = [{
        "content": f"User asked about: {question}",
        "label": "question_topic",
        "importance": 1,
        "memory_type": "fact",
        "agent_scope": "global",
    }]
    logger.info("Point")
    for mem in candidate_memories:
        mid = insert_memory(
            user_id=user_id,
            content=mem["content"],
            label=mem.get("label"),
            importance=mem.get("importance", 3),
            memory_type=mem.get("memory_type", "fact"),
            agent_scope=mem.get("agent_scope", "global"),
        )
        upsert_memory_vector(
            memory_id=mid,
            user_id=user_id,
            content=mem["content"],
            )


    return {
        "answer": as_text(response),
        "turn_id": turn_id,
        "store_path": store_path,
        "docs": docs,
        "latency_ms": latency_ms,
        "k_used": k_used,
        "corpus_n": corpus_n,
        "prefiltered": prefiltered,
        "facet_filter": facet_filter,
        "mode_hint": mode_hint,
        "session_id": session_id,
        "mem_docs": mem_docs,
    }

 