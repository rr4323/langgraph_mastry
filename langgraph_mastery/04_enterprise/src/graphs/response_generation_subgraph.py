"""
Sub-graph for collaborative response generation.

This module defines a sub-graph where two agents, a drafter and a reviewer,
collaborate to generate a high-quality response to a user's query.

This sub-graph is intended to be called from the main workflow graph.
"""

import logging
from typing import TypedDict, Annotated, List

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from src.agents.agent_factory import create_agent

logger = logging.getLogger(__name__)

# --- State for the Sub-graph ---
class ResponseGenerationState(TypedDict):
    """The state for the response generation sub-graph."""
    query: str
    retrieved_documents: List[str]
    messages: Annotated[list[BaseMessage], add_messages]

# --- Nodes for the Sub-graph ---
def drafting_node(state: ResponseGenerationState) -> dict:
    """The node for the drafting agent."""
    logger.info("Running drafting node")
    drafter = create_agent("drafter")
    context = "\n\n".join(state["retrieved_documents"])
    prompt = f"Based on the following context, please draft a comprehensive response to the user's query.\n\nContext:\n{context}\n\nQuery: {state['query']}"
    response = drafter.invoke({"messages": [HumanMessage(content=prompt)]})
    return {"messages": [response]}

def reviewing_node(state: ResponseGenerationState) -> dict:
    """The node for the reviewing agent."""
    logger.info("Running reviewing node")
    reviewer = create_agent("reviewer")
    last_message = state["messages"][-1].content
    prompt = f"Please review the following draft response. If it is good, say 'OK'. Otherwise, provide feedback on how to improve it.\n\nDraft:\n{last_message}"
    response = reviewer.invoke({"messages": [HumanMessage(content=prompt)]})
    return {"messages": [response]}

# --- Conditional Edge Logic ---
def should_continue(state: ResponseGenerationState) -> str:
    """Determines whether to continue drafting or finish."""
    logger.info("Checking if response is final")
    last_message = state["messages"][-1].content
    if "OK" in last_message.upper():
        logger.info("Response is final. Finishing.")
        return "end"
    else:
        logger.info("Response needs revision. Continuing draft.")
        return "continue"

# --- Create the Sub-graph ---
def create_response_generation_graph() -> StateGraph:
    """Creates the response generation sub-graph."""
    builder = StateGraph(ResponseGenerationState)
    builder.add_node("drafter", drafting_node)
    builder.add_node("reviewer", reviewing_node)

    builder.set_entry_point("drafter")
    builder.add_edge("drafter", "reviewer")

    builder.add_conditional_edges(
        "reviewer",
        should_continue,
        {
            "continue": "drafter",
            "end": END
        }
    )

    return builder.compile()
