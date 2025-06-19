"""
This script provides a tutorial on the Agent Supervisor multi-agent pattern in LangGraph.

The Agent Supervisor pattern is a foundational architecture for creating dynamic, conversational
multi-agent systems. It involves a central LLM-based supervisor that acts as a router,
analyzing the current state of the conversation and deciding which specialist agent should
handle the task next.

This pattern is highly effective for workflows where you need to route between different
experts or tools based on user input.

Key Concepts Illustrated:
- **Central Supervisor**: An LLM-based node that directs the flow of the graph.
- **Specialist Agents**: Multiple "worker" agents, each with a specific role or tool.
- **Dynamic Routing**: The supervisor uses function-calling to decide which agent to call
  next, or whether to end the conversation.
- **Command-Based Control**: The graph's flow is controlled using `Command(goto=...)`,
  which allows for dynamic, state-driven transitions between nodes.

Workflow:
1.  A user provides input.
2.  The `supervisor_node` analyzes the input and the conversation history.
3.  It decides which agent is best suited for the task: the `research_agent` for factual
    queries or the `router_agent` for general conversation.
4.  It routes control to the selected agent.
5.  The agent performs its task and returns control to the supervisor.
6.  The supervisor decides the next step, which could be another agent or ending the workflow.

To run this script:
1.  Install the required packages:
    ```bash
    pip install langgraph langchain_google_genai langchain_core tavily-python
    ```
2.  Set the following environment variables:
    - `GOOGLE_API_KEY`: Your API key for Google Gemini.
    - `TAVILY_API_KEY`: Your API key for the Tavily search engine.
3.  Run the script:
    ```bash
    python langgraph_mastery/03_advanced/14_agent_supervisor.py
    ```
"""

import os
from typing import TypedDict, Literal, Annotated

from langchain_core.messages import AnyMessage, HumanMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from tavily import TavilyClient

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import Command

# --- 1. Define the State for the Graph ---

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

# --- 2. Define Tools ---

if "TAVILY_API_KEY" not in os.environ:
    raise ValueError("Please set the TAVILY_API_KEY environment variable.")
tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

def search_tool(query: str) -> str:
    """Performs a web search using Tavily and returns the results."""
    print(f"---SUPERVISED AGENT (Tool): Performing search for '{query}'---")
    results = tavily_client.search(query=query, max_results=3)
    return "\n".join([r["content"] for r in results["results"]])

# --- 3. Define the Worker Agents ---

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0)

def research_agent(state: AgentState) -> AgentState:
    """An agent that uses the search tool to answer research questions."""
    print("---AGENT: Research Agent---")
    query = state["messages"][-1].content
    result = search_tool(query)
    return {"messages": [HumanMessage(content=f"Research results: {result}")]}

def router_agent(state: AgentState) -> AgentState:
    """A conversational agent that can handle general queries."""
    print("---AGENT: Router Agent---")
    prompt = (
        "You are a helpful assistant. Respond to the user's query directly. "
        f"User query: {state['messages'][-1].content}"
    )
    result = llm.invoke(prompt).content
    return {"messages": [HumanMessage(content=result)]}

# --- 4. Define the Supervisor Node ---

class Router(BaseModel):
    """Function-calling model to route work to the correct agent or end the conversation."""
    next: Literal["research_agent", "router_agent", "__end__"] = Field(
        description="The next agent to route to, or '__end__' to finish."
    )

supervisor_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0)
structured_llm = supervisor_llm.with_structured_output(Router)

def supervisor_node(state: AgentState) -> Command[Literal["research_agent", "router_agent", END]]:
    """The central supervisor that decides the next action."""
    print("---SUPERVISOR: Analyzing query and routing...---")
    prompt = (
        "You are a supervisor in a multi-agent system. Based on the user's query, "
        "route them to the appropriate agent. Use 'research_agent' for questions that require "
        "web searches, and 'router_agent' for general conversation. If the query is a greeting or a simple question, use 'router_agent'."
        f"User query: {state['messages'][-1].content}"
    )
    
    route = structured_llm.invoke(prompt)
    print(f"---SUPERVISOR: Decision: route to {route.next}---")
    
    if route.next == "__end__":
        return END
    return Command(goto=route.next)

# --- 5. Build the Graph ---

builder = StateGraph(AgentState)

builder.add_node("supervisor", supervisor_node)
builder.add_node("research_agent", research_agent)
builder.add_node("router_agent", router_agent)

builder.add_edge(START, "supervisor")
builder.add_edge("research_agent", "supervisor")
builder.add_edge("router_agent", "supervisor")

# The supervisor will decide where to go next

graph = builder.compile()

# --- 6. Run the Graph ---

def run_graph(query: str):
    print(f"\n--- Running graph for query: '{query}' ---")
    initial_state = {"messages": [HumanMessage(content=query)]}
    for event in graph.stream(initial_state, {"recursion_limit": 5}):
        print(event)
        print("---")

def main():
    """Main function to run a few example queries."""
    if "GOOGLE_API_KEY" not in os.environ:
        raise ValueError("Please set the GOOGLE_API_KEY environment variable.")

    run_graph("Hello, how are you today?")
    run_graph("What are the latest developments in large language models?")

if __name__ == "__main__":
    main()
