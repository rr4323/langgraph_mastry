"""
LangGraph Advanced: Complex Workflows
===================================

This script demonstrates how to build complex workflows with conditional branching
and error handling using LangGraph and Google's Generative AI model.
"""

import os
import sys
import time
import random
from typing import Annotated, Dict, List, Literal, TypedDict, Union, Any, Optional
from enum import Enum
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, END
from langchain_community.tools import DuckDuckGoSearchResults, DuckDuckGoSearchRun

# Add the parent directory to the path so we can import from the root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables from .env file
load_dotenv()

# Define enums for workflow states
class TaskStatus(str, Enum):
    """Status of a task in the workflow."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    NEEDS_REVIEW = "needs_review"

class ReviewDecision(str, Enum):
    """Decision made during review."""
    APPROVE = "approve"
    REJECT = "reject"
    REVISE = "revise"

# Define our state for the complex workflow
class ComplexWorkflowState(TypedDict):
    """State for our complex workflow."""
    messages: List[Union[HumanMessage, AIMessage, SystemMessage]]
    task: str
    status: TaskStatus
    research: str
    plan: str
    draft: str
    review_feedback: str
    final_output: str
    error: Optional[str]
    next: Literal["research", "planning", "drafting", "review", "revision", "finalization", "error_handler", "end"]

# Define tools for the workflow
@tool
def search_information(query: str) -> str:
    """
    Search for information related to a query.
    
    Args:
        query: The search query
    
    Returns:
        str: Search results
    """
    try:
        # Detailed results (title + link)
        search_tool = DuckDuckGoSearchResults()
        raw_results = search_tool.run(query, max_results=5)
        if isinstance(raw_results, list):
            formatted = "\n".join([
                f"{idx+1}. {item['title']} - {item['href']}" for idx, item in enumerate(raw_results)
            ])
        else:
            formatted = raw_results
        # Concise summary using the lighter wrapper
        simple_search = DuckDuckGoSearchRun()
        summary = simple_search.run(query)
        return (
            f"Top search results for '{query}':\n\n" + formatted +
            "\n\nQuick summary:\n" + summary
        )
    except Exception as e:
        # Fallback to placeholder text if search fails
        return f"Search unavailable ({e}). Here is a placeholder summary about '{query}'.\n" + \
               f"1. {query} is a topic of wide interest.\n2. Recent studies highlight significant developments in {query}.\n3. Experts propose various approaches to advance {query}."

# Helper to build a ReAct-based researcher agent
def create_researcher_agent():
    """Return a ReAct-style agent that can call the `search_information` tool."""
    model = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.3,
        convert_system_message_to_human=True,
    )
    tools = [search_information]
    agent = create_react_agent(
        model=model,
        tools=tools,
        prompt="""You are a research assistant with access to external search tools.\nYour goal is to gather comprehensive, factual information about a given topic.\nThink step-by-step, decide when you should call a tool, observe the result, and then continue reasoning until you can return a clear, well-structured research summary.""",
    )
    return agent

# Define the workflow nodes
def research_node(state: ComplexWorkflowState) -> ComplexWorkflowState:
    """Research node in our complex workflow."""
    print("🔍 Researching information...")
    
    try:
        # Use the ReAct-style agent so the model can autonomously decide when to call tools
        task = state["task"]
        researcher = create_researcher_agent()
        result = researcher.invoke({"messages": [HumanMessage(content=task)]})
        # The response format mirrors the advanced_tools example
        research = result["messages"][-1].content if isinstance(result, dict) and "messages" in result else result
        return {
            **state,
            "research": research,
            "next": "planning",
            "error": None,
        }
    except Exception as e:
        # Handle any errors
        error_message = f"Error in research node: {str(e)}"
        print(f"❌ {error_message}")
        return {
            **state,
            "error": error_message,
            "next": "error_handler"
        }

def planning_node(state: ComplexWorkflowState) -> ComplexWorkflowState:
    """Planning node in our complex workflow."""
    print("📝 Creating a plan...")
    
    try:
        # Create a planning agent
        model = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.4,
            convert_system_message_to_human=True
        )
        
        # Extract the task and research from the state
        task = state["task"]
        research = state["research"]
        
        # Create messages for the planning
        messages = [
            SystemMessage(content="""You are a planning specialist.
Your job is to create a structured plan based on research information.
The plan should outline the key sections and points to cover."""),
            HumanMessage(content=f"""
Task: {task}

Research Information:
{research}

Create a detailed plan for completing this task. Include:
1. Main sections to cover
2. Key points for each section
3. Logical flow and structure
""")
        ]
        
        # Get the plan
        response = model.invoke(messages)
        plan = response.content
        
        # Update the state
        return {
            **state,
            "plan": plan,
            "next": "drafting",
            "error": None
        }
    except Exception as e:
        # Handle any errors
        error_message = f"Error in planning node: {str(e)}"
        print(f"❌ {error_message}")
        return {
            **state,
            "error": error_message,
            "next": "error_handler"
        }

def drafting_node(state: ComplexWorkflowState) -> ComplexWorkflowState:
    """Drafting node in our complex workflow."""
    print("✍️ Creating a draft...")
    
    try:
        # Create a drafting agent
        model = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.7,
            convert_system_message_to_human=True
        )
        
        # Extract the task, research, and plan from the state
        task = state["task"]
        research = state["research"]
        plan = state["plan"]
        
        # Create messages for the drafting
        messages = [
            SystemMessage(content="""You are a content creation specialist.
Your job is to create a well-written draft based on research and a plan.
The draft should be comprehensive, clear, and engaging."""),
            HumanMessage(content=f"""
Task: {task}

Research Information:
{research}

Plan:
{plan}

Create a comprehensive draft based on this research and plan.
""")
        ]
        
        # Get the draft
        response = model.invoke(messages)
        draft = response.content
        
        # Randomly decide if the draft needs review (70% chance)
        needs_review = random.random() < 0.7
        
        # Update the state
        return {
            **state,
            "draft": draft,
            "status": TaskStatus.NEEDS_REVIEW if needs_review else TaskStatus.COMPLETED,
            "next": "review" if needs_review else "finalization",
            "error": None
        }
    except Exception as e:
        # Handle any errors
        error_message = f"Error in drafting node: {str(e)}"
        print(f"❌ {error_message}")
        return {
            **state,
            "error": error_message,
            "next": "error_handler"
        }

def review_node(state: ComplexWorkflowState) -> ComplexWorkflowState:
    """Review node in our complex workflow."""
    print("👀 Reviewing the draft...")
    
    try:
        # Create a review agent
        model = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.4,
            convert_system_message_to_human=True
        )
        
        # Extract the task and draft from the state
        task = state["task"]
        draft = state["draft"]
        
        # Create messages for the review
        messages = [
            SystemMessage(content="""You are a quality review specialist.
Your job is to review content and provide constructive feedback.
Evaluate the content for accuracy, clarity, completeness, and engagement."""),
            HumanMessage(content=f"""
Task: {task}

Draft to Review:
{draft}

Please review this draft and provide feedback on:
1. Content accuracy and completeness
2. Structure and organization
3. Clarity and readability
4. Suggested improvements

Also provide a review decision: APPROVE, REJECT, or REVISE.
""")
        ]
        
        # Get the review feedback
        response = model.invoke(messages)
        review_feedback = response.content
        
        # Determine the review decision
        decision = ReviewDecision.REVISE  # Default decision
        if "APPROVE" in review_feedback:
            decision = ReviewDecision.APPROVE
        elif "REJECT" in review_feedback:
            decision = ReviewDecision.REJECT
        
        # Determine next step based on the decision
        next_step = "finalization" if decision == ReviewDecision.APPROVE else "revision"
        
        # Update the state
        return {
            **state,
            "review_feedback": review_feedback,
            "next": next_step,
            "error": None
        }
    except Exception as e:
        # Handle any errors
        error_message = f"Error in review node: {str(e)}"
        print(f"❌ {error_message}")
        return {
            **state,
            "error": error_message,
            "next": "error_handler"
        }

def revision_node(state: ComplexWorkflowState) -> ComplexWorkflowState:
    """Revision node in our complex workflow."""
    print("🔄 Revising the draft...")
    
    try:
        # Create a revision agent
        model = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.6,
            convert_system_message_to_human=True
        )
        
        # Extract the task, draft, and review feedback from the state
        task = state["task"]
        draft = state["draft"]
        review_feedback = state["review_feedback"]
        
        # Create messages for the revision
        messages = [
            SystemMessage(content="""You are a content revision specialist.
Your job is to revise content based on review feedback.
Address all the feedback points and improve the content accordingly."""),
            HumanMessage(content=f"""
Task: {task}

Original Draft:
{draft}

Review Feedback:
{review_feedback}

Please revise the draft based on this feedback. Create a complete revised version.
""")
        ]
        
        # Get the revised draft
        response = model.invoke(messages)
        revised_draft = response.content
        
        # Update the state
        return {
            **state,
            "draft": revised_draft,
            "next": "finalization",
            "error": None
        }
    except Exception as e:
        # Handle any errors
        error_message = f"Error in revision node: {str(e)}"
        print(f"❌ {error_message}")
        return {
            **state,
            "error": error_message,
            "next": "error_handler"
        }

def finalization_node(state: ComplexWorkflowState) -> ComplexWorkflowState:
    """Finalization node in our complex workflow."""
    print("✅ Finalizing the output...")
    
    try:
        # Create a finalization agent
        model = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.5,
            convert_system_message_to_human=True
        )
        
        # Extract the task and draft from the state
        task = state["task"]
        draft = state["draft"]
        
        # Create messages for the finalization
        messages = [
            SystemMessage(content="""You are a content finalization specialist.
Your job is to polish and finalize content to ensure it's ready for delivery.
Focus on final improvements in clarity, formatting, and overall quality."""),
            HumanMessage(content=f"""
Task: {task}

Draft to Finalize:
{draft}

Please finalize this content. Make any necessary improvements to:
1. Formatting and structure
2. Language and style
3. Overall presentation

The output should be publication-ready.
""")
        ]
        
        # Get the finalized output
        response = model.invoke(messages)
        final_output = response.content
        
        # Update the state
        return {
            **state,
            "final_output": final_output,
            "status": TaskStatus.COMPLETED,
            "next": "end",
            "error": None
        }
    except Exception as e:
        # Handle any errors
        error_message = f"Error in finalization node: {str(e)}"
        print(f"❌ {error_message}")
        return {
            **state,
            "error": error_message,
            "next": "error_handler"
        }

def error_handler_node(state: ComplexWorkflowState) -> ComplexWorkflowState:
    """Error handler node in our complex workflow."""
    print("🛠️ Handling error...")
    
    # Extract the error from the state
    error = state["error"]
    print(f"Error details: {error}")
    
    # Determine the recovery strategy based on the current state
    if not state["research"]:
        # If research failed, try to recover by skipping research
        print("Recovery strategy: Skipping research and proceeding to planning")
        return {
            **state,
            "research": "Research could not be completed due to an error. Proceeding with planning based on available information.",
            "next": "planning",
            "error": None
        }
    elif not state["plan"]:
        # If planning failed, try to recover by creating a simple plan
        print("Recovery strategy: Creating a simple default plan")
        return {
            **state,
            "plan": f"Simple plan for task: {state['task']}\n\n1. Introduction\n2. Main content\n3. Conclusion",
            "next": "drafting",
            "error": None
        }
    elif not state["draft"]:
        # If drafting failed, mark the task as failed
        print("Recovery strategy: Marking task as failed")
        return {
            **state,
            "status": TaskStatus.FAILED,
            "final_output": f"The task could not be completed due to an error: {error}",
            "next": "end",
            "error": None
        }
    else:
        # For other errors, try to salvage what we have
        print("Recovery strategy: Salvaging existing work")
        return {
            **state,
            "final_output": f"Partial completion of task. Last successful stage: {state['draft']}",
            "status": TaskStatus.COMPLETED,
            "next": "end",
            "error": None
        }

def create_complex_workflow():
    """Create a complex workflow using LangGraph."""
    print("Creating a complex workflow with LangGraph...")
    
    # Create a new graph
    workflow = StateGraph(ComplexWorkflowState)
    
    # Add nodes to the graph
    workflow.add_node("research_node", research_node)
    workflow.add_node("planning_node", planning_node)
    workflow.add_node("drafting_node", drafting_node)
    workflow.add_node("review_node", review_node)
    workflow.add_node("revision_node", revision_node)
    workflow.add_node("finalization_node", finalization_node)
    workflow.add_node("error_handler_node", error_handler_node)
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "research_node",
        lambda state: state["next"],
        {
            "planning": "planning_node",
            "error_handler": "error_handler_node"
        }
    )
    
    workflow.add_conditional_edges(
        "planning_node",
        lambda state: state["next"],
        {
            "drafting": "drafting_node",
            "error_handler": "error_handler_node"
        }
    )
    
    workflow.add_conditional_edges(
        "drafting_node",
        lambda state: state["next"],
        {
            "review": "review_node",
            "finalization": "finalization_node",
            "error_handler": "error_handler_node"
        }
    )
    
    workflow.add_conditional_edges(
        "review_node",
        lambda state: state["next"],
        {
            "revision": "revision_node",
            "finalization": "finalization_node",
            "error_handler": "error_handler_node"
        }
    )
    
    workflow.add_conditional_edges(
        "revision_node",
        lambda state: state["next"],
        {
            "finalization": "finalization_node",
            "error_handler": "error_handler_node"
        }
    )
    
    workflow.add_conditional_edges(
        "finalization_node",
        lambda state: state["next"],
        {
            "end": END,
            "error_handler": "error_handler_node"
        }
    )
    
    workflow.add_conditional_edges(
        "error_handler_node",
        lambda state: state["next"],
        {
            "planning": "planning_node",
            "drafting": "drafting_node",
            "end": END
        }
    )
    
    # Set the entry point
    workflow.set_entry_point("research_node")
    
    # Compile the graph
    return workflow.compile()

def run_complex_workflow(workflow, task: str):
    """Run the complex workflow with a given task."""
    print("\n" + "=" * 50)
    print("Running the Complex Workflow")
    print("=" * 50)
    
    # Initialize the state
    initial_state = {
        "messages": [HumanMessage(content=task)],
        "task": task,
        "status": TaskStatus.PENDING,
        "research": "",
        "plan": "",
        "draft": "",
        "review_feedback": "",
        "final_output": "",
        "error": None,
        "next": "research"
    }
    
    # Run the workflow
    print(f"\nTask: {task}")
    result = workflow.invoke(initial_state)
    
    # Print the workflow results
    print("\n" + "=" * 50)
    print("Workflow Results")
    print("=" * 50)
    
    print(f"\nTask Status: {result['status']}")
    
    if result["research"]:
        print("\nResearch:")
        print("-" * 50)
        print(result["research"][:300] + "..." if len(result["research"]) > 300 else result["research"])
    
    if result["plan"]:
        print("\nPlan:")
        print("-" * 50)
        print(result["plan"][:300] + "..." if len(result["plan"]) > 300 else result["plan"])
    
    if result["review_feedback"]:
        print("\nReview Feedback:")
        print("-" * 50)
        print(result["review_feedback"][:300] + "..." if len(result["review_feedback"]) > 300 else result["review_feedback"])
    
    print("\nFinal Output:")
    print("-" * 50)
    print(result["final_output"])
    
    return result

def main():
    """Main function to demonstrate a complex workflow."""
    print("=" * 50)
    print("Complex Workflows in LangGraph")
    print("=" * 50)
    
    # Create a complex workflow
    workflow = create_complex_workflow()
    
    # Get a task from the user
    task = input("\nEnter a task for the complex workflow (e.g., 'Write an article about artificial intelligence'): ")
    
    # Run the complex workflow
    run_complex_workflow(workflow, task)

if __name__ == "__main__":
    main()
