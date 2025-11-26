# governedge_agent.py
#from LightAgent import LightAgent
import os

from dolphin import DolphinAgent 


from config_base.config import Config
from engine_query.qe_tool_runner import run_query_engine



def make_qe_tool(llm_key: str, embed_key: str, ce_key: str, session_id: str):
    """
    Returns a tool function bound to specific llm/embed/ce keys.
    """
    def qe_answer(question: str) -> str:
        result = run_query_engine(
            question,
            llm_key=llm_key,
            embed_key=embed_key,
            ce_key=ce_key,
            session_id=session_id,
        )
        # You could stash result in a global or return JSON, but simplest:
        return result["answer"]

    qe_answer.tool_info = {
        "tool_name": "qe_answer",
        "tool_description": "Answer questions using my local query engine, documents, and SQL.",
        "tool_params": [
            {
                "name": "question",
                "description": "The user question.",
                "type": "string",
                "required": True,
            },
        ],
    }

    return qe_answer


def make_agent(llm_key: str, embed_key: str, ce_key: str, session_id: str) -> DolphinAgent:
    model_id =  Config.LLM_MODEL_MAP[llm_key]  # e.g. "qwen3:8b"
    base_url = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
    api_key  = os.getenv("LOCAL_LLM_API_KEY", "no-key-needed")

    qe_tool = make_qe_tool(llm_key, embed_key, ce_key, session_id)

    agent_dolphin = DolphinAgent(
        model=model_id,
        api_key=api_key,
        base_url=base_url,
        tools=[qe_tool],
    )
    return agent_dolphin
