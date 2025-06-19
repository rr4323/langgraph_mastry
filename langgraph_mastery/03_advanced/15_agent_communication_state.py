"""
This script provides a tutorial on advanced multi-agent communication, specifically focusing
on how agents with different, independent state schemas can collaborate.

In complex, real-world applications, different teams or developers might build agents
independently. Each agent may have its own internal state, optimized for its specific
task. This is a powerful and scalable pattern, but it requires a clear strategy for
communication and data transformation between agents.

This tutorial demonstrates the most robust solution for this challenge: using a parent
"orchestrator" graph to manage the interaction between specialist agents with private states.

Key Concepts Illustrated:
- **Private State Schemas**: We define two distinct agents, a `research_agent` and a
  `summary_agent`, each with its own unique `TypedDict` for state management.
- **Orchestrator Graph**: A top-level graph is created to manage the workflow. It does not
  perform the core tasks itself but is responsible for calling the specialist agents.
- **State Transformation**: The orchestrator contains nodes that are responsible for
  invoking the specialist agents. These nodes transform the data from the orchestrator's
  state into the format expected by the specialist and then transform the specialist's
  output back into the orchestrator's state.
- **Modularity and Decoupling**: The research and summary agents are completely decoupled.
  They could be developed, tested, and maintained in isolation, promoting a clean and
  scalable architecture.

Workflow:
1.  The `OrchestratorState` receives an initial query.
2.  The `run_researcher` node is called. It extracts the query, invokes the `research_agent`
    (which runs on its own `ResearchAgentState`), and captures the research data.
3.  The `run_summarizer` node is called. It takes the research data, invokes the
    `summary_agent` (which runs on its own `SummaryAgentState`), and captures the summary.
4.  The final summary is stored in the `OrchestratorState`.

To run this script:
```bash
# Make sure you have a GOOGLE_API_KEY set in your environment variables.
python langgraph_mastery/03_advanced/15_agent_communication_state.py
```
"""

import os
from typing import TypedDict, Annotated

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from langgraph.graph import StateGraph, END, START

# --- 1. Define Private States and Graphs for Specialist Agents ---

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0)

# --- Research Agent ---
class ResearchAgentState(TypedDict):
    """Private state for the research agent."""
    query: str
    research_data: str

def research_node(state: ResearchAgentState) -> ResearchAgentState:
    """A node that simulates doing research."""
    print(f"---AGENT (Researcher): Performing research for '{state['query']}'---")
    # In a real app, this would use a search tool.
    data = f"Extensive research data on '{state['query']}'. Key findings include X, Y, and Z."
    return {"research_data": data}

research_builder = StateGraph(ResearchAgentState)
research_builder.add_node("research", research_node)
research_builder.set_entry_point("research")
research_builder.set_finish_point("research")
research_agent = research_builder.compile()

# --- Summary Agent ---
class SummaryAgentState(TypedDict):
    """Private state for the summary agent."""
    text_to_summarize: str
    summary: str

def summary_node(state: SummaryAgentState) -> SummaryAgentState:
    """A node that summarizes text using an LLM."""
    print("---AGENT (Summarizer): Summarizing text---")
    prompt = f"Please provide a concise summary of the following text:\n\n{state['text_to_summarize']}"
    result = llm.invoke(prompt).content
    return {"summary": result}

summary_builder = StateGraph(SummaryAgentState)
summary_builder.add_node("summarize", summary_node)
summary_builder.set_entry_point("summarize")
summary_builder.set_finish_point("summarize")
summary_agent = summary_builder.compile()


# --- 2. Define the State and Nodes for the Orchestrator Graph ---

class OrchestratorState(TypedDict):
    """The state for the parent graph that orchestrates the agents."""
    initial_query: str
    research_data: str
    final_summary: str

def run_researcher(state: OrchestratorState) -> dict:
    """Node to invoke the research agent and transform state."""
    print("---ORCHESTRATOR: Delegating to Researcher---")
    # Prepare the input for the research agent
    researcher_input = {"query": state["initial_query"]}
    # Invoke the agent
    researcher_output = research_agent.invoke(researcher_input)
    # Transform the output back into the orchestrator's state
    return {"research_data": researcher_output["research_data"]}

def run_summarizer(state: OrchestratorState) -> dict:
    """Node to invoke the summary agent and transform state."""
    print("---ORCHESTRATOR: Delegating to Summarizer---")
    # Prepare the input for the summary agent
    summarizer_input = {"text_to_summarize": state["research_data"]}
    # Invoke the agent
    summarizer_output = summary_agent.invoke(summarizer_input)
    # Transform the output back into the orchestrator's state
    return {"final_summary": summarizer_output["summary"]}


# --- 3. Build and Compile the Orchestrator Graph ---

builder = StateGraph(OrchestratorState)

builder.add_node("run_researcher", run_researcher)
builder.add_node("run_summarizer", run_summarizer)

builder.set_entry_point("run_researcher")
builder.add_edge("run_researcher", "run_summarizer")
builder.add_edge("run_summarizer", END)

orchestrator_graph = builder.compile()


# --- 4. Run the Orchestrator ---

def main():
    """Main function to run the orchestrator graph."""
    if "GOOGLE_API_KEY" not in os.environ:
        raise ValueError("Please set the GOOGLE_API_KEY environment variable.")

    print("Starting orchestrator graph...")
    initial_input = {"initial_query": "The impact of AI on modern software development"}
    
    # Stream the events to see the flow of execution
    for event in orchestrator_graph.stream(initial_input, stream_mode="values"):
        print(event)
        print("---")

    final_result = orchestrator_graph.invoke(initial_input)
    print("\n--- FINAL SUMMARY ---")
    print(final_result["final_summary"])

if __name__ == "__main__":
    main()
