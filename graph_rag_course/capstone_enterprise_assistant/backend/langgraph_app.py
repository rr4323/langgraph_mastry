"""Enterprise LangGraph app: multi-agent with tracing & timeout."""
import os, logging
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from stage3_multi_agent.30_agents import researcher_agent, analyst_agent, writer_agent
from langchain.chat_models import ChatOpenAI

load_dotenv()
logger = logging.getLogger("capstone.graph")
chat = ChatOpenAI(temperature=0.0, request_timeout=15)

QUALITY_THRESHOLD = 0.85

def coordinator(state):
    if "context" not in state:
        return {**state, "next": "research"}
    if "facts" not in state:
        return {**state, "next": "analyze"}
    return {**state, "next": "write"}

def research(state):
    ctx = researcher_agent(state["question"])
    return {**state, "context": ctx, "next": "analyze"}

def analyze(state):
    facts = analyst_agent(state["question"], state["context"])
    return {**state, "facts": facts, "next": "write"}

def write(state):
    answer = writer_agent(state["facts"])
    # simple heuristic quality (length)
    score = len(answer.split()) / 50.0
    logger.info("quality score=%.2f", score)
    if score < QUALITY_THRESHOLD:
        return {**state, "next": "analyze"}
    return {"answer": answer, "next": END}

sg = StateGraph(dict)
sg.add_node("coord", coordinator)
sg.add_node("research", research)
sg.add_node("analyze", analyze)
sg.add_node("write", write)
sg.add_conditional_edges("coord", lambda s: s["next"], {"research": "research", "analyze": "analyze", "write": "write"})
sg.add_conditional_edges("research", lambda s: s["next"], {"analyze": "analyze"})
sg.add_conditional_edges("analyze", lambda s: s["next"], {"write": "write"})
sg.add_conditional_edges("write", lambda s: s["next"], {"analyze": "analyze", "end": END})
sg.set_entry_point("coord")
app = sg.compile()

if __name__ == "__main__":
    while True:
        q = input("Ask: ")
        res = app.invoke({"question": q})
        print(res["answer"])
