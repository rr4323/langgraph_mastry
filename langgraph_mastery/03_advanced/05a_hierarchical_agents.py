"""
LangGraph Advanced: Hierarchical Agent Architectures
=================================================

This script demonstrates how to build hierarchical agent architectures with
supervisor and worker agents using LangGraph and Google's Generative AI model.
"""

import os
import sys
import time
from typing import Annotated, Dict, List, Literal, TypedDict, Union, Any, Optional
from enum import Enum
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END

# Add the parent directory to the path so we can import from the root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables from .env file
load_dotenv()

# Define our state for the hierarchical agent system
class HierarchicalState(TypedDict):
    """State for our hierarchical agent system."""
    messages: List[Union[HumanMessage, AIMessage, SystemMessage]]
    task: str
    supervisor_plan: str
    worker_assignments: Dict[str, str]
    worker_results: Dict[str, str]
    supervisor_feedback: Dict[str, str]
    final_output: str
    next: Optional[str]

# Define worker types
class WorkerType(str, Enum):
    """Types of worker agents in the system."""
    RESEARCHER = "researcher"
    ANALYST = "analyst"
    WRITER = "writer"
    FACT_CHECKER = "fact_checker"
    EDITOR = "editor"

# Define tools for the workers
@tool
def search_information(query: str) -> str:
    """
    Search for information related to a query.
    
    Args:
        query: The search query
    
    Returns:
        str: Search results
    """
    # Simulate API call delay
    time.sleep(1)
    
    # Simulate search results
    return f"Here are the search results for '{query}':\n\n" + \
           f"1. {query} is a topic of great interest in recent research.\n" + \
           f"2. According to recent studies, {query} has shown significant developments.\n" + \
           f"3. Experts in {query} suggest several approaches to understanding this topic.\n" + \
           f"4. The history of {query} dates back to several decades ago.\n" + \
           f"5. Future trends in {query} indicate promising directions for further exploration."

@tool
def check_facts(statement: str) -> Dict[str, Any]:
    """
    Check the factual accuracy of a statement.
    
    Args:
        statement: The statement to fact-check
    
    Returns:
        Dict: Fact-checking results
    """
    # Simulate API call delay
    time.sleep(1)
    
    # Simulate fact-checking
    import random
    accuracy = random.uniform(0.7, 1.0)
    
    return {
        "statement": statement,
        "accuracy_score": round(accuracy, 2),
        "verified": accuracy > 0.8,
        "confidence": "high" if accuracy > 0.9 else "medium" if accuracy > 0.8 else "low"
    }

# Supervisor node
def supervisor_planning_node(state: HierarchicalState) -> HierarchicalState:
    """Supervisor planning node in the hierarchical system."""
    print("🧠 Supervisor: Creating a plan...")
    
    # Create a supervisor agent
    model = ChatGoogleGenerativeAI(
        model="gemini-pro",
        temperature=0.4,
        convert_system_message_to_human=True
    )
    
    # Extract the task from the state
    task = state["task"]
    
    # Create messages for the supervisor
    messages = [
        SystemMessage(content="""You are a supervisor agent responsible for planning and coordinating tasks.
Your job is to break down a complex task into subtasks and assign them to specialized worker agents.
Available worker types:
1. RESEARCHER - Gathers information and conducts research
2. ANALYST - Analyzes data and information
3. WRITER - Creates well-written content
4. FACT_CHECKER - Verifies factual accuracy
5. EDITOR - Polishes and improves content

Create a detailed plan with specific assignments for each worker type."""),
        HumanMessage(content=f"Task: {task}\n\nCreate a detailed plan to accomplish this task.")
    ]
    
    # Get the plan from the supervisor
    response = model.invoke(messages)
    supervisor_plan = response.content
    
    # Create worker assignments based on the plan
    worker_assignments = {}
    
    # Create a separate agent to extract worker assignments from the plan
    assignment_model = ChatGoogleGenerativeAI(
        model="gemini-pro",
        temperature=0.1,
        convert_system_message_to_human=True
    )
    
    assignment_messages = [
        SystemMessage(content="""Extract specific assignments for each worker type from the supervisor's plan.
Format the output as a JSON-like structure with worker types as keys and their assignments as values.
Only include worker types that have specific assignments."""),
        HumanMessage(content=f"Supervisor's Plan:\n\n{supervisor_plan}")
    ]
    
    assignment_response = assignment_model.invoke(assignment_messages)
    
    # Parse the assignments
    try:
        import re
        import json
        
        # Try to extract JSON-like structure
        match = re.search(r'\{.*\}', assignment_response.content, re.DOTALL)
        if match:
            assignments_str = match.group(0)
            assignments_str = assignments_str.replace("'", "\"")
            worker_assignments = json.loads(assignments_str)
        else:
            # Fallback: manually extract assignments
            for worker_type in WorkerType:
                if worker_type.value.upper() in assignment_response.content:
                    sections = assignment_response.content.split(worker_type.value.upper())
                    if len(sections) > 1:
                        assignment = sections[1].split("\n\n")[0].strip()
                        worker_assignments[worker_type.value] = assignment
    except Exception as e:
        print(f"Error parsing assignments: {e}")
        # Fallback: create default assignments
        worker_assignments = {
            WorkerType.RESEARCHER.value: f"Research information about {task}",
            WorkerType.ANALYST.value: f"Analyze the research findings about {task}",
            WorkerType.WRITER.value: f"Write content about {task} based on research and analysis",
            WorkerType.FACT_CHECKER.value: f"Verify the factual accuracy of the content about {task}",
            WorkerType.EDITOR.value: f"Polish and improve the content about {task}"
        }
    
    # Update the state
    return {
        **state,
        "supervisor_plan": supervisor_plan,
        "worker_assignments": worker_assignments,
        "next": "worker_execution"
    }

def worker_execution_node(state: HierarchicalState) -> HierarchicalState:
    """Worker execution node in the hierarchical system."""
    print("👷 Workers: Executing assigned tasks...")
    
    # Extract worker assignments from the state
    worker_assignments = state["worker_assignments"]
    task = state["task"]
    
    # Initialize worker results
    worker_results = {}
    
    # Process each worker assignment
    for worker_type, assignment in worker_assignments.items():
        print(f"  - {worker_type.upper()}: Working on assignment...")
        
        # Create a worker agent based on the type
        model = ChatGoogleGenerativeAI(
            model="gemini-pro",
            temperature=0.7,
            convert_system_message_to_human=True
        )
        
        # Create messages for the worker
        system_content = f"You are a specialized {worker_type} agent. "
        
        if worker_type == WorkerType.RESEARCHER.value:
            system_content += """Your job is to gather comprehensive information on a topic.
Focus on collecting factual information, key concepts, and important details."""
            
            # Add search tool results
            search_results = search_information(task)
            assignment += f"\n\nHere are some search results to help you:\n{search_results}"
            
        elif worker_type == WorkerType.ANALYST.value:
            system_content += """Your job is to analyze information and data.
Focus on identifying patterns, insights, and implications."""
            
        elif worker_type == WorkerType.WRITER.value:
            system_content += """Your job is to create well-written content.
Focus on clarity, engagement, and effective communication."""
            
        elif worker_type == WorkerType.FACT_CHECKER.value:
            system_content += """Your job is to verify the factual accuracy of content.
Focus on identifying and correcting any inaccuracies."""
            
        elif worker_type == WorkerType.EDITOR.value:
            system_content += """Your job is to polish and improve content.
Focus on grammar, style, structure, and overall quality."""
        
        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=f"Task: {task}\n\nAssignment: {assignment}")
        ]
        
        # Get the worker's result
        response = model.invoke(messages)
        worker_results[worker_type] = response.content
    
    # Update the state
    return {
        **state,
        "worker_results": worker_results,
        "next": "supervisor_review"
    }

def supervisor_review_node(state: HierarchicalState) -> HierarchicalState:
    """Supervisor review node in the hierarchical system."""
    print("🧠 Supervisor: Reviewing worker results...")
    
    # Extract worker results from the state
    worker_results = state["worker_results"]
    task = state["task"]
    
    # Initialize supervisor feedback
    supervisor_feedback = {}
    
    # Review each worker's result
    for worker_type, result in worker_results.items():
        print(f"  - Reviewing {worker_type.upper()}'s work...")
        
        # Create a supervisor agent for review
        model = ChatGoogleGenerativeAI(
            model="gemini-pro",
            temperature=0.4,
            convert_system_message_to_human=True
        )
        
        # Create messages for the supervisor review
        messages = [
            SystemMessage(content="""You are a supervisor agent responsible for reviewing work.
Your job is to provide constructive feedback on the work of specialized agents.
Focus on identifying strengths, weaknesses, and areas for improvement."""),
            HumanMessage(content=f"""
Task: {task}

Worker Type: {worker_type.upper()}

Worker's Result:
{result}

Please review this work and provide constructive feedback.
""")
        ]
        
        # Get the supervisor's feedback
        response = model.invoke(messages)
        supervisor_feedback[worker_type] = response.content
    
    # Update the state
    return {
        **state,
        "supervisor_feedback": supervisor_feedback,
        "next": "integration"
    }

def integration_node(state: HierarchicalState) -> HierarchicalState:
    """Integration node in the hierarchical system."""
    print("🔄 Integrating results and feedback...")
    
    # Extract information from the state
    task = state["task"]
    supervisor_plan = state["supervisor_plan"]
    worker_results = state["worker_results"]
    supervisor_feedback = state["supervisor_feedback"]
    
    # Create an integration agent
    model = ChatGoogleGenerativeAI(
        model="gemini-pro",
        temperature=0.5,
        convert_system_message_to_human=True
    )
    
    # Prepare the worker results and feedback
    results_and_feedback = ""
    for worker_type in WorkerType:
        if worker_type.value in worker_results:
            results_and_feedback += f"\n\n## {worker_type.value.upper()} RESULTS\n"
            results_and_feedback += worker_results[worker_type.value]
            
            if worker_type.value in supervisor_feedback:
                results_and_feedback += f"\n\n### Supervisor Feedback on {worker_type.value.upper()}'s Work\n"
                results_and_feedback += supervisor_feedback[worker_type.value]
    
    # Create messages for the integration
    messages = [
        SystemMessage(content="""You are an integration specialist.
Your job is to combine the results from multiple specialized agents into a cohesive final output.
Take into account the supervisor's feedback and ensure the final output is comprehensive and high-quality."""),
        HumanMessage(content=f"""
Task: {task}

Supervisor's Plan:
{supervisor_plan}

Worker Results and Supervisor Feedback:
{results_and_feedback}

Please integrate all the information into a cohesive final output that accomplishes the task.
""")
    ]
    
    # Get the integrated result
    response = model.invoke(messages)
    final_output = response.content
    
    # Update the state
    return {
        **state,
        "final_output": final_output,
        "next": "end"
    }

def create_hierarchical_system():
    """Create a hierarchical agent system using LangGraph."""
    print("Creating a hierarchical agent system with LangGraph...")
    
    # Create a new graph
    workflow = StateGraph(HierarchicalState)
    
    # Add nodes to the graph
    workflow.add_node("supervisor_planning", supervisor_planning_node)
    workflow.add_node("worker_execution", worker_execution_node)
    workflow.add_node("supervisor_review", supervisor_review_node)
    workflow.add_node("integration", integration_node)
    
    # Add edges
    workflow.add_conditional_edges(
        "supervisor_planning",
        lambda state: state["next"],
        {
            "worker_execution": "worker_execution"
        }
    )
    
    workflow.add_conditional_edges(
        "worker_execution",
        lambda state: state["next"],
        {
            "supervisor_review": "supervisor_review"
        }
    )
    
    workflow.add_conditional_edges(
        "supervisor_review",
        lambda state: state["next"],
        {
            "integration": "integration"
        }
    )
    
    workflow.add_conditional_edges(
        "integration",
        lambda state: state["next"],
        {
            "end": END
        }
    )
    
    # Set the entry point
    workflow.set_entry_point("supervisor_planning")
    
    # Compile the graph
    return workflow.compile()

def run_hierarchical_system(workflow, task: str):
    """Run the hierarchical agent system with a given task."""
    print("\n" + "=" * 50)
    print("Running the Hierarchical Agent System")
    print("=" * 50)
    
    # Initialize the state
    initial_state = {
        "messages": [HumanMessage(content=task)],
        "task": task,
        "supervisor_plan": "",
        "worker_assignments": {},
        "worker_results": {},
        "supervisor_feedback": {},
        "final_output": "",
        "next": None
    }
    
    # Run the workflow
    print(f"\nTask: {task}")
    result = workflow.invoke(initial_state)
    
    # Print the workflow results
    print("\n" + "=" * 50)
    print("Hierarchical System Results")
    print("=" * 50)
    
    print("\nSupervisor's Plan:")
    print("-" * 50)
    print(result["supervisor_plan"][:300] + "..." if len(result["supervisor_plan"]) > 300 else result["supervisor_plan"])
    
    print("\nWorker Assignments:")
    print("-" * 50)
    for worker_type, assignment in result["worker_assignments"].items():
        print(f"{worker_type.upper()}: {assignment[:100]}..." if len(assignment) > 100 else assignment)
    
    print("\nFinal Output:")
    print("-" * 50)
    print(result["final_output"])
    
    return result

def main():
    """Main function to demonstrate hierarchical agent architectures."""
    print("=" * 50)
    print("Hierarchical Agent Architectures in LangGraph")
    print("=" * 50)
    
    # Create a hierarchical agent system
    workflow = create_hierarchical_system()
    
    # Get a task from the user
    task = input("\nEnter a task for the hierarchical agent system (e.g., 'Create a comprehensive guide about climate change'): ")
    
    # Run the hierarchical agent system
    run_hierarchical_system(workflow, task)

if __name__ == "__main__":
    main()
