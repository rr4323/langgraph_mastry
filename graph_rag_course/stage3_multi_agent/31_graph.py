"""LangGraph workflow with Coordinator, Researcher, Analyst, Writer and feedback loop."""
from langgraph.graph import StateGraph, END
from langchain.schema import AIMessage, HumanMessage
from langchain.chat_models import ChatOpenAI
from 30_agents import researcher_agent, analyst_agent, writer_agent

chat = ChatOpenAI(temperature=0.0)

QUALITY_THRESHOLD = 0.8  # Dummy score; in real life use eval model

def coordinator(state):
    if "context" not in state:
        return {**state, "next": "researcher"}
    if "facts" not in state:
        return {**state, "next": "analyst"}
    return {**state, "next": "writer"}

def researcher_node(state):
    context = researcher_agent(state["question"])
    return {**state, "context": context, "next": "analyst"}

def analyst_node(state):
    facts = analyst_agent(state["question"], state["context"])
    return {**state, "facts": facts, "next": "writer"}

def writer_node(state):
    answer = writer_agent(state["facts"])
    print("\nDraft answer:\n", answer)
    score = 1.0  # Placeholder quality metric
    if score < QUALITY_THRESHOLD:
        return {**state, "next": "analyst"}  # loop back for refinement
    return {"answer": answer, "next": END}

sg = StateGraph(dict)
sg.add_node("researcher", researcher_node)
sg.add_node("analyst", analyst_node)
sg.add_node("writer", writer_node)
sg.add_node("coord", coordinator)

sg.add_conditional_edges("coord", lambda s: s["next"], {"researcher": "researcher", "analyst": "analyst", "writer": "writer"})
sg.add_conditional_edges("researcher", lambda s: s["next"], {"analyst": "analyst"})
sg.add_conditional_edges("analyst", lambda s: s["next"], {"writer": "writer"})
sg.add_conditional_edges("writer", lambda s: s["next"], {"analyst": "analyst", "end": END})
sg.set_entry_point("coord")
app = sg.compile()

if __name__ == "__main__":
    while True:
        q = input("Ask (or exit): ")
        if q.lower() == "exit":
            break
        res = app.invoke({"question": q})
        print("\nFinal answer:\n", res["answer"])
