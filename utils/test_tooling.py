# pip install -U langchain langchain-ollama
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import AIMessage

@tool
def add(a: int, b: int) -> int:
    "Add two integers."
    return a + b

llm = ChatOllama(model="deepseek-r1:8b", temperature=0).bind_tools([add])  # tool-tuned model
resp = llm.invoke("Add 7 and 12 using the tool.")
print(type(resp), getattr(resp, "tool_calls", None))
