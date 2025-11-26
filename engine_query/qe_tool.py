# governedge_agent.py
from LightAgent import LightAgent
from config_base.config import Config
from qe_tool_runner import run_query_engine
import os

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


def make_agent(llm_key: str, embed_key: str, ce_key: str, session_id: str) -> LightAgent:
    cfg = Config.LLM_MODEL_MAP[llm_key]

    model_id = cfg["model_id"]           # e.g. "qwen3:8b"
    base_url = cfg.get("base_url", "https://api.openai.com/v1")
    api_env  = cfg.get("api_env", "OPENAI_API_KEY")
    api_key  = os.getenv(api_env, "no-key-needed")

    qe_tool = make_qe_tool(llm_key, embed_key, ce_key, session_id)

    agent = LightAgent(
        model=model_id,
        api_key=api_key,
        base_url=base_url,
        tools=[qe_tool],
    )
    return agent
