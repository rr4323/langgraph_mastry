"""
This script provides a tutorial on conditional branching in LangGraph, a fundamental
concept for creating dynamic and intelligent workflows.

Conditional branching allows your graph to make decisions and route execution to different
nodes based on the current state. This is the core mechanism that enables complex behaviors
like agentic supervisors, loops, and reactive systems.

This tutorial demonstrates a common use case: creating an agent that first classifies a
user's query and then decides whether to use a tool or respond directly.

Key Concepts Illustrated:
- **Router Function**: A function (`classify_query`) that inspects the graph's state and
  returns a string indicating the next node to execute.
- **`add_conditional_edges`**: The method used to connect a decision point (the classifier)
  to multiple possible next steps.
- **Dynamic Path Selection**: The graph will follow a different path depending on the
  outcome of the classification, making the workflow adaptive to the input.

Workflow:
1.  The user provides a query.
2.  The `classify_query` node is called. It uses an LLM with function-calling to decide
    if the query requires a web search or can be answered directly.
3.  The conditional edge evaluates the classifier's output.
4.  If the output is "use_search_tool", the graph moves to the `search_node`.
5.  If the output is "direct_response", the graph moves to the `direct_response_node`.
6.  The selected node executes, and the graph finishes.

To run this script:
```bash
# Make sure you have a GOOGLE_API_KEY set in your environment variables.
python langgraph_mastery/02_intermediate/06_conditional_branching.py
```
"""

import os
from typing import TypedDict, Literal, Annotated

from langchain_core.messages import HumanMessage, BaseMessage, AIMessage
from langchain_core.pydantic_v1 import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

# --- 1. Define the State ---

class AgentState(TypedDict):
    """The state for our agent."""
    messages: Annotated[list[BaseMessage], add_messages]

# --- 2. Define Nodes ---

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0)

def search_node(state: AgentState) -> AgentState:
    """A node that simulates using a search tool."""
    print("---NODE: Search Node---")
    query = state["messages"][-1].content
    print(f"---NODE: Search Node--- (Simulating search for '{query}')")
    # In a real app, this would use a tool like Tavily.
    search_result = f"The capital of France is Paris. It is known for the Eiffel Tower."
    return {"messages": [HumanMessage(content=search_result)]}

def direct_response_node(state: AgentState) -> AgentState:
    """A node that generates a direct conversational response."""
    print("---NODE: Direct Response Node---")
    query = state["messages"][-1].content
    response = llm.invoke(f"You are a helpful assistant. Respond to the following: '{query}'")
    return {"messages": [response]}

# --- 3. Define the Router Function and Conditional Edge ---

class Router(BaseModel):
    """Pydantic model to define the output of our router LLM call."""
    next_node: Literal["use_search_tool", "direct_response"]

# Attach the Pydantic model to the LLM to get structured output
structured_llm = llm.with_structured_output(Router)

def classify_query(state: AgentState) -> Literal["use_search_tool", "direct_response"]:
    """
    This is the router function. It classifies the user's query and returns a string
    that matches one of the keys in our conditional edge mapping.
    """
    print("---NODE: Classifier--- (Deciding the next step)")
    query = state["messages"][-1].content
    prompt = (
        "You are an expert at routing user queries. Classify the user's query as either 'use_search_tool' "
        "for questions that require factual information, or 'direct_response' for greetings and simple conversation."
        f"User query: {query}"
    )
    
    result = structured_llm.invoke(prompt)
    decision = result.next_node

    print(f"---Classifier Decision: '{decision}'---")
    return decision

def classify_node(state: AgentState) -> AgentState:
    """
    This is the router function. It classifies the user's query and returns a string
    that matches one of the keys in our conditional edge mapping.
    """
    print("---NODE: Classifier--- (Deciding the next step)")
    query = state["messages"][-1].content
    prompt = (
        "You are an expert at routing user queries. Classify the user's query as either 'use_search_tool' "
        "for questions that require factual information, or 'direct_response' for greetings and simple conversation."
        f"User query: {query}"
    )
    
    result = structured_llm.invoke(prompt)
    decision = result.next_node

    print(f"---Classifier Decision: '{decision}'---")
    return {
        "__next__": decision,
        "messages": state["messages"] + [AIMessage(content=f"Routing to: {decision}")]
    }


# --- 4. Build the Graph ---

builder = StateGraph(AgentState)

# Add the nodes to the graph
builder.add_node("classifier", classify_node)
builder.add_node("search_node", search_node)
builder.add_node("direct_response_node", direct_response_node)

# Set the entry point
builder.set_entry_point("classifier")

# Add the conditional edge. This is the core of the branching logic.
builder.add_conditional_edges(
    # The source node is the classifier. Its output will determine the next step.
    "classifier",
    # The second argument is the router function itself.
    classify_query,
    # The third argument is a dictionary mapping the router's output to the next node.
    {
        "use_search_tool": "search_node",
        "direct_response": "direct_response_node",
    },
)

# Add normal edges from the leaf nodes to the end
builder.add_edge("search_node", END)
builder.add_edge("direct_response_node", END)

# Compile the graph
graph = builder.compile()

# --- 5. Run the Graph ---

def run_graph(query: str):
    print(f"\n--- Running graph for query: '{query}' ---")
    initial_state = {"messages": [HumanMessage(content=query)]}
    # The `stream` method lets us see the flow of execution
    for event in graph.stream(initial_state, {"recursion_limit": 5}):
        print(event)
        print("---")

def main():
    """Main function to run a few example queries."""
    if "GOOGLE_API_KEY" not in os.environ:
        raise ValueError("Please set the GOOGLE_API_KEY environment variable.")

    # This query should be routed to the search node
    run_graph("What is the capital of France?")
    
    # This query should be routed to the direct response node
    run_graph("Hi there, how's it going?")

if __name__ == "__main__":
    main()
