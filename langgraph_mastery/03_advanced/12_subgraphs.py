"""
This script demonstrates the concept of subgraphs in LangGraph.

Subgraphs are a powerful feature for building complex, modular, and hierarchical AI systems.
They allow you to encapsulate a self-contained workflow into its own graph and then use
that entire graph as a single node within a parent graph. This is a key pattern for
creating multi-agent systems where different agents (each represented by a graph) can
collaborate.

This example illustrates a common use case: a "Manager" agent that delegates a task to a
"Researcher" agent.

Key Concepts:
- **Encapsulation**: The Researcher's internal logic (its nodes and edges) is completely
  hidden from the Manager. The Manager only interacts with the Researcher's public
  interface (its input and output schema).
- **State Management**: The script demonstrates the scenario where the parent graph (Manager)
  and the subgraph (Researcher) have different state schemas. This is a flexible pattern
  that allows each component to manage its own internal state.
- **State Transformation**: The Manager has a dedicated node (`invoke_researcher_subgraph`)
  responsible for calling the Researcher subgraph. This node transforms the Manager's
  state into the format expected by the Researcher and then transforms the Researcher's
  output back into the Manager's state.

Workflow:
1.  **Researcher Subgraph**: A simple graph is defined to simulate a research process. It
    takes a research topic, performs a couple of steps, and produces a report.
2.  **Manager Parent Graph**: A higher-level graph that defines a broader task. One of its
    nodes is responsible for invoking the Researcher subgraph.
3.  **Invocation**: The Manager's `invoke_researcher_subgraph` node prepares the input for the
    Researcher, calls `subgraph.invoke()`, and processes the result.

To run this script:
```bash
# Make sure you have a GOOGLE_API_KEY set in your environment variables.
python langgraph_mastery/03_advanced/12_subgraphs.py
```
"""

import os
from typing import TypedDict, Annotated

from langchain_core.messages import AnyMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from typing_extensions import Doc

from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages

# --- 1. Define the State and Nodes for the Researcher Subgraph ---


class ResearcherState(TypedDict):
    """State for the researcher subgraph."""

    research_topic: str
    initial_notes: str
    full_report: str


def research_step_1(state: ResearcherState):
    """First step of the research process."""
    print("---SUBGRAPH: Research Step 1---")
    # In a real app, this would do actual research (e.g., web search)
    notes = f"Initial research notes on {state['research_topic']}. Key points are A, B, and C."
    return {"initial_notes": notes}


def research_step_2(state: ResearcherState):
    """Second step, refining the research into a report."""
    print("---SUBGRAPH: Research Step 2---")
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0)
    prompt = f"Based on these notes: '{state['initial_notes']}', write a one-paragraph summary report."
    report = llm.invoke(prompt).content
    return {"full_report": report}


# --- 2. Build and Compile the Researcher Subgraph ---

researcher_builder = StateGraph(ResearcherState)
researcher_builder.add_node("step_1", research_step_1)
researcher_builder.add_node("step_2", research_step_2)

researcher_builder.add_edge(START, "step_1")
researcher_builder.add_edge("step_1", "step_2")
researcher_builder.add_edge("step_2", END)

# The subgraph is a compiled graph, just like any other.
researcher_subgraph = researcher_builder.compile()


# --- 3. Define the State and Nodes for the Manager Parent Graph ---


class ManagerState(TypedDict):
    """State for the manager parent graph."""

    task_description: str
    final_summary: str


def invoke_researcher_subgraph(state: ManagerState):
    """This node invokes the researcher subgraph and transforms the state."""
    print("---PARENT GRAPH: Delegating to Researcher Subgraph---")
    task = state["task_description"]

    # The input for the subgraph must match its state schema (ResearcherState)
    subgraph_input = {"research_topic": task}

    # Invoke the subgraph
    subgraph_output = researcher_subgraph.invoke(subgraph_input)

    # Transform the subgraph's output back into the parent graph's state
    final_report = subgraph_output["full_report"]
    summary = f"Manager's summary of the research: {final_report}"
    return {"final_summary": summary}


# --- 4. Build and Compile the Manager Parent Graph ---

manager_builder = StateGraph(ManagerState)
# Here, we add the node that *calls* the subgraph.
manager_builder.add_node("delegate_research", invoke_researcher_subgraph)

manager_builder.add_edge(START, "delegate_research")
manager_builder.add_edge("delegate_research", END)

manager_graph = manager_builder.compile()


# --- 5. Run the Parent Graph ---

def main():
    """Main function to run the parent graph."""
    if "GOOGLE_API_KEY" not in os.environ:
        raise ValueError("Please set the GOOGLE_API_KEY environment variable.")

    print("Starting the manager-researcher graph execution...")
    initial_input = {"task_description": "The future of AI in education"}
    final_state = manager_graph.invoke(initial_input)

    print("\n--- FINAL RESULT ---")
    print(final_state["final_summary"])
    print("\nGraph execution finished.")


if __name__ == "__main__":
    main()
