"""
Simple Example Script for Enterprise Knowledge Assistant

This script demonstrates the core concepts of the Enterprise Knowledge Assistant
in a simplified form, connecting the basic LangGraph concepts to the enterprise implementation.
"""

import os
import sys
from typing import Dict, Any, TypedDict, Literal
from dotenv import load_dotenv

# Add the parent directory to the path so we can import from the root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables from .env file
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, END

# Define a simple state for our workflow
class SimpleAssistantState(TypedDict):
    """A simplified state for the Enterprise Knowledge Assistant."""
    query: str
    context: Dict[str, Any]
    response: str
    next: Literal["understand_query", "generate_response", "end"]

def understand_query(state: SimpleAssistantState) -> SimpleAssistantState:
    """Understand the user query."""
    print(f"🧠 Understanding query: {state['query']}")
    
    # Create a chat model
    model = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.3,
        convert_system_message_to_human=True
    )
    
    # Create messages for the model
    messages = [
        SystemMessage(content="""You are a query understanding assistant.
Your job is to analyze user queries and extract key information.
Format your response as a brief analysis of the query."""),
        HumanMessage(content=state["query"])
    ]
    
    # Get the analysis
    response = model.invoke(messages)
    
    # Update the state
    state["context"]["query_analysis"] = response.content
    state["next"] = "generate_response"
    
    return state

def generate_response(state: SimpleAssistantState) -> SimpleAssistantState:
    """Generate a response to the user query."""
    print(f"💬 Generating response based on analysis")
    
    # Create a chat model
    model = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.7,
        convert_system_message_to_human=True
    )
    
    # Get the query analysis
    query_analysis = state["context"].get("query_analysis", "")
    
    # Create messages for the model
    messages = [
        SystemMessage(content=f"""You are an Enterprise Knowledge Assistant.
Use the following query analysis to provide a helpful response:
{query_analysis}

Always be professional and helpful."""),
        HumanMessage(content=state["query"])
    ]
    
    # Generate the response
    response = model.invoke(messages)
    
    # Update the state
    state["response"] = response.content
    state["next"] = "end"
    
    return state

def create_simple_workflow() -> StateGraph:
    """Create a simple workflow for the Enterprise Knowledge Assistant."""
    # Create a new graph
    workflow = StateGraph(SimpleAssistantState)
    
    # Add nodes to the graph
    workflow.add_node("understand_query", understand_query)
    workflow.add_node("generate_response", generate_response)
    
    # Add edges to the graph
    workflow.add_conditional_edges(
        "understand_query",
        lambda state: state["next"],
        {
            "generate_response": "generate_response",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "generate_response",
        lambda state: state["next"],
        {
            "end": END
        }
    )
    
    # Set the entry point
    workflow.set_entry_point("understand_query")
    
    # Compile the graph
    return workflow.compile()

def main():
    """Run the simple example."""
    print("=" * 50)
    print("Enterprise Knowledge Assistant - Simple Example")
    print("=" * 50)
    
    # Create the workflow
    workflow = create_simple_workflow()
    
    # Get user input
    query = input("\nEnter your query: ")
    
    # Create the initial state
    state = {
        "query": query,
        "context": {},
        "response": "",
        "next": "understand_query"
    }
    
    # Process the query
    result = workflow.invoke(state)
    
    # Display the result
    print("\n" + "=" * 50)
    print("Response:")
    print(result["response"])
    print("=" * 50)
    
    # Show the query analysis
    print("\nQuery Analysis:")
    print(result["context"]["query_analysis"])

if __name__ == "__main__":
    main()
