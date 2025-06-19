"""
LangGraph Basics: Understanding Simple Workflows
===============================================

This script demonstrates how to create a simple workflow in LangGraph
using Google's Generative AI model.
"""

from doctest import debug
import os
import sys
from typing import Annotated, Literal, TypedDict
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

# Add the parent directory to the path so we can import from the root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables from .env file
load_dotenv()

# Define our state for the workflow
class WorkflowState(TypedDict):
    """State for our simple workflow."""
    question: str
    research: str
    answer: str
    next: Literal["research", "answer", "end"]

def create_research_agent():
    """Create an agent for researching information."""
    model = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.3,
        convert_system_message_to_human=True,
    )
    
    return model

def create_answer_agent():
    """Create an agent for formulating answers."""
    model = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.7,
        convert_system_message_to_human=True,
    )
    
    return model

def research(state: WorkflowState) -> WorkflowState:
    """Research node in our workflow."""
    print("🔍 Researching information...")
    
    research_agent = create_research_agent()
    
    # Create messages for the research agent
    messages = [
        SystemMessage(content="""You are a research assistant. 
Your job is to gather relevant information about a question.
Provide concise but comprehensive research notes that will help answer the question."""),
        HumanMessage(content=f"Research this question: {state['question']}")
    ]
    
    # Get research information
    response = research_agent.invoke(messages)
    
    # Update the state
    return {
        **state,
        "research": response.content,
        "next": "answer"
    }

def answer(state: WorkflowState) -> WorkflowState:
    """Answer node in our workflow."""
    print("✍️ Formulating an answer...")
    
    answer_agent = create_answer_agent()
    
    # Create messages for the answer agent
    messages = [
        SystemMessage(content="""You are an expert at answering questions.
Use the research provided to formulate a clear, concise, and accurate answer."""),
        HumanMessage(content=f"""
Question: {state['question']}

Research: {state['research']}

Please provide a comprehensive answer based on this research.
""")
    ]
    
    # Get the answer
    response = answer_agent.invoke(messages)
    
    # Update the state
    return {
        **state,
        "answer": response.content,
        "next": "end"
    }

def create_workflow():
    """Create a simple workflow using LangGraph."""
    print("Creating a simple LangGraph workflow...")
    
    # Create a new graph
    workflow = StateGraph(WorkflowState)
    
    # Add nodes to the graph
    workflow.add_node("research_node", research)
    workflow.add_node("answer_node", answer)
    
    # Add edges to the graph
    workflow.add_edge("research_node", "answer_node")
    workflow.add_edge("answer_node", END)
    
    # Set the entry point
    workflow.set_entry_point("research_node")
    
    # Compile the graph
    return workflow.compile()

def run_workflow(workflow, question: str):
    """Run the workflow with a given question."""
    print("\n" + "=" * 50)
    print("Running the LangGraph Workflow")
    print("=" * 50)
    
    # Initialize the state
    initial_state = {
        "question": question,
        "research": "",
        "answer": "",
        "next": "research"
    }
    
    # Run the workflow
    print(f"\nQuestion: {question}")
    result = workflow.invoke(initial_state)
    
    # Print the result
    print("\n" + "=" * 50)
    print("Workflow Result")
    print("=" * 50)
    print(f"\nResearch Notes:\n{result['research']}")
    print("\n" + "-" * 50)
    print(f"Final Answer:\n{result['answer']}")

def main():
    """Main function to create and run a simple workflow."""
    print("=" * 50)
    print("Understanding Simple Workflows in LangGraph")
    print("=" * 50)
    
    # Create a workflow
    workflow = create_workflow()
    
    # Get a question from the user
    question = input("\nEnter a question to process through the workflow: ")
    
    # Run the workflow
    run_workflow(workflow, question)

if __name__ == "__main__":
    main()
