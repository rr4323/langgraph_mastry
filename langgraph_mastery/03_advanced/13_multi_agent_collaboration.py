"""
This script provides a comprehensive tutorial on building a hierarchical multi-agent system
using LangGraph.

Hierarchical multi-agent systems are a powerful architecture for building complex and
scalable AI applications. They involve creating specialized teams of agents, each managed
by its own supervisor, and a top-level supervisor that orchestrates these teams.

This example demonstrates a common pattern:
1.  **A "Research Team" Subgraph**: This is a self-contained graph representing a team of
    agents with a specific purpose (e.g., conducting research).
2.  **Team Supervisor**: The research team is managed by its own supervisor, built using
    the convenient `create_react_agent` prebuilt from LangGraph. This supervisor uses
    its team members (defined as tools) to accomplish its tasks.
3.  **Top-Level Supervisor**: A higher-level graph that acts as the main entry point.
    It delegates tasks to the appropriate team and processes the result.

Key Concepts Illustrated:
- **Modularity**: The research team is a reusable component (a subgraph) that can be
  plugged into any other graph.
- **Hierarchy**: A top-level agent manages a team of sub-agents, creating a clear
  chain of command.
- **Tool-Based Agents**: Worker agents are exposed as tools to their supervisor, which
  simplifies the supervisor's logic. It just needs to decide which tool to call next.
- **State Management**: The top-level graph and the subgraph maintain their own states,
  and data is passed between them.

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
    python langgraph_mastery/03_advanced/13_multi_agent_collaboration.py
    ```
"""

import os
from typing import TypedDict

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from tavily import TavilyClient

from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import create_react_agent

# --- 1. Define Tools for the Research Team ---
# These functions will be the "worker agents" that the research team's supervisor can call.

# Ensure TAVILY_API_KEY is set before initializing the client
if "TAVILY_API_KEY" not in os.environ:
    raise ValueError("Please set the TAVILY_API_KEY environment variable.")
tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

def web_search(query: str) -> str:
    """
    Performs a web search using the Tavily client.
    Args:
        query: The search query.
    Returns:
        A string containing the search results.
    """
    print(f"---RESEARCH TEAM (Tool): Performing web search for '{query}'---")
    results = tavily_client.search(query=query, max_results=2)
    # Tavily search returns a list of dictionaries. We extract the 'content'.
    return "\n".join([r["content"] for r in results["results"]])

# --- 2. Create the Research Team Supervisor (Subgraph) ---
# We use the `create_react_agent` prebuilt to easily create a supervisor
# that can use the tools we defined. This compiled graph is our subgraph.

research_tools = [web_search]
# Use a specific model for the researcher agent
researcher_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0)
research_team_subgraph = create_react_agent(researcher_llm, tools=research_tools)


# --- 3. Define the Top-Level Supervisor Graph ---

class TopLevelState(TypedDict):
    """
    State for the top-level supervisor graph.
    """
    task: str
    research_result: str

def delegate_to_research_team(state: TopLevelState):
    """
    Node that invokes the research team subgraph.
    This is the bridge between the top-level graph and the subgraph.
    """
    print("---TOP-LEVEL SUPERVISOR: Delegating task to Research Team---")
    task = state["task"]
    
    # The input for the subgraph must match its expected state.
    # The `create_react_agent` expects a list of messages.
    subgraph_input = {"messages": [HumanMessage(content=task)]}
    
    # Invoke the subgraph
    subgraph_result = research_team_subgraph.invoke(subgraph_input)
    
    # The result of the prebuilt agent is the final list of messages,
    # where the last message is the agent's final answer.
    final_answer = subgraph_result["messages"][-1].content
    
    return {"research_result": final_answer}

def summarize_result(state: TopLevelState):
    """
    A final node in the top-level graph to summarize the research.
    """
    print("---TOP-LEVEL SUPERVISOR: Summarizing the research result---")
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.1)
    prompt = f"Provide a final, concise summary of the following research report:\n\n{state['research_result']}"
    summary = llm.invoke(prompt).content
    return {"research_result": summary}


# --- 4. Build and Compile the Top-Level Graph ---

builder = StateGraph(TopLevelState)

builder.add_node("delegate_to_research_team", delegate_to_research_team)
builder.add_node("summarize_result", summarize_result)

builder.set_entry_point("delegate_to_research_team")
builder.add_edge("delegate_to_research_team", "summarize_result")
builder.add_edge("summarize_result", END)

top_level_graph = builder.compile()


# --- 5. Run the Hierarchical Multi-Agent System ---

def main():
    """
    Main function to run the graph.
    """
    if "GOOGLE_API_KEY" not in os.environ:
        raise ValueError("Please set the GOOGLE_API_KEY environment variable.")

    print("Starting hierarchical multi-agent system...")
    
    initial_input = {
        "task": "What is the current status of the Artemis program by NASA? What are the next major milestones?"
    }
    
    # Stream the events to see the flow of execution
    print("\n--- Graph Execution Stream ---")
    for event in top_level_graph.stream(initial_input, stream_mode="values"):
        print(event)
        print("---")

    print("\n--- HIERARCHICAL EXECUTION COMPLETE ---")


if __name__ == "__main__":
    main()
