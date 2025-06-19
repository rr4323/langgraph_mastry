"""
LangGraph Advanced: Human-in-the-Loop Systems
===========================================

This script demonstrates how to create interactive systems with human feedback
using LangGraph and Google's Generative AI model.
"""

import os
import sys
import time
import uuid
from typing import Annotated, Dict, List, Literal, TypedDict, Union, Any, Optional
from enum import Enum
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver

# Add the parent directory to the path so we can import from the root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables from .env file
load_dotenv()

# Define enums for workflow states
class ApprovalStatus(str, Enum):
    """Status of human approval."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_REVISION = "needs_revision"

# Define our state for the human-in-the-loop workflow
class HITLState(TypedDict):
    """State for our human-in-the-loop workflow."""
    messages: List[Union[HumanMessage, AIMessage, SystemMessage]]
    task: str
    draft: str
    human_feedback: str
    final_output: str
    approval_status: ApprovalStatus
    revision_count: int
    next: Literal["drafting", "human_review", "revision", "finalization", "end"]

def drafting_node(state: HITLState) -> HITLState:
    """Drafting node in our human-in-the-loop workflow."""
    print("✍️ Creating a draft...")
    
    # Create a drafting agent
    model = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.7,
        convert_system_message_to_human=True
    )
    
    # Extract the task from the state
    task = state["task"]
    
    # If we have human feedback from a previous revision, include it
    feedback_prompt = ""
    if state["human_feedback"] and state["revision_count"] > 0:
        feedback_prompt = f"""
Previous draft:
{state['draft']}

Human feedback on the previous draft:
{state['human_feedback']}

Please address all the feedback points in your new draft.
"""
    
    # Create messages for the drafting, maintaining conversation history
    system_message = SystemMessage(content="""You are a content creation specialist.
Your job is to create high-quality content based on the given task.
If human feedback is provided, make sure to address all the feedback points.""")
    
    # Prepare the task message with feedback if available
    task_message = HumanMessage(content=f"""Task: {task}
{feedback_prompt}

Create a comprehensive and well-structured draft. Be sure to address all points from the task and any provided feedback.""")
    
    # Build message history
    messages = [system_message]
    
    # Add previous messages if they exist
    if "messages" in state and state["messages"]:
        messages.extend(state["messages"][-4:])  # Keep last 4 messages for context
    
    # Add the current task message
    messages.append(task_message)
    
    # Get the draft
    response = model.invoke(messages)
    draft = response.content
    
    # Update the state
    return {
        **state,
        "draft": draft,
        "approval_status": ApprovalStatus.PENDING,
        "next": "human_review"
    }

def human_review_node(state: HITLState) -> HITLState:
    """Human review node in our human-in-the-loop workflow with enhanced feedback."""
    print("\n" + "=" * 50)
    print("Human Review Required")
    print("=" * 50)
    
    # Display the task and draft for human review
    print(f"\nTask: {state['task']}")
    print(f"Revision: {state['revision_count'] + 1}")
    
    if state['revision_count'] > 0 and state.get('human_feedback'):
        print("\nPrevious Feedback:")
        print("-" * 50)
        print(state['human_feedback'])
        print("-" * 50)
    
    print("\nDraft for Review:")
    print("-" * 50)
    print(state["draft"])
    print("-" * 50)
    
    # Get human feedback with more detailed options
    print("\nPlease review the draft and provide your feedback.")
    print("Options:")
    print("1. Approve (a): Accept the draft as is")
    print("2. Reject (r): Reject the draft completely")
    print("3. Revise (v): Request specific revisions")
    print("4. View full history (h): See all previous messages")
    
    while True:
        decision = input("\nYour decision (a/r/v/h): ").lower().strip()
        
        if decision in ['a', 'approve']:
            approval_status = ApprovalStatus.APPROVED
            human_feedback = "Approved without changes."
            next_step = "finalization"
            break
            
        elif decision in ['r', 'reject']:
            approval_status = ApprovalStatus.REJECTED
            print("\nPlease provide a reason for rejection (or press Enter to cancel):")
            feedback = input("> ").strip()
            if not feedback:
                print("Rejection cancelled. Please choose another option.")
                continue
            human_feedback = f"REJECTED: {feedback}"
            next_step = "end"
            break
            
        elif decision in ['v', 'revise']:
            approval_status = ApprovalStatus.NEEDS_REVISION
            print("\nPlease provide specific feedback for revision (be as detailed as possible):")
            print("You can request changes to content, style, structure, etc.")
            feedback = input("> ").strip()
            if not feedback:
                print("No feedback provided. Please provide specific feedback or choose another option.")
                continue
            human_feedback = f"REVISION REQUESTED: {feedback}"
            next_step = "revision"
            break
            
        elif decision in ['h', 'history']:
            print("\nMessage History:")
            print("-" * 50)
            for i, msg in enumerate(state.get('messages', [])[-5:], 1):  # Show last 5 messages
                print(f"{i}. {msg.type.upper()}: {msg.content[:200]}{'...' if len(msg.content) > 200 else ''}")
            print("-" * 50)
            continue
            
        else:
            print("Invalid option. Please enter 'a' (approve), 'r' (reject), 'v' (revise), or 'h' (history).")
    
    # Update the state with the new feedback
    updated_messages = state.get('messages', []).copy()
    updated_messages.append(HumanMessage(content=f"Human Feedback: {human_feedback}"))
    
    return {
        **state,
        "messages": updated_messages,
        "human_feedback": human_feedback,
        "approval_status": approval_status,
        "next": next_step
    }

def revision_node(state: HITLState) -> HITLState:
    """Revision node in our human-in-the-loop workflow with enhanced state management."""
    print("\n🔄 Processing revision request...")
    
    # Increment the revision count
    revision_count = state["revision_count"] + 1
    
    # Add the human feedback to the message history
    updated_messages = state.get('messages', []).copy()
    if state.get('human_feedback'):
        updated_messages.append(AIMessage(content=f"Revision requested. Feedback: {state['human_feedback']}"))
    
    # Update the state to trigger the drafting node again
    return {
        **state,
        "messages": updated_messages,
        "revision_count": revision_count,
        "next": "drafting"
    }

def finalization_node(state: HITLState) -> HITLState:
    """Finalization node in our human-in-the-loop workflow with enhanced error handling."""
    print("\n✅ Finalizing the output...")
    
    try:
        # Create a finalization agent
        model = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash-latest",
            temperature=0.5,
            convert_system_message_to_human=True
        )
        
        # Extract the task and draft from the state
        task = state["task"]
        draft = state["draft"]
        
        # Create messages for the finalization, including previous context
        system_message = SystemMessage(content="""You are a content finalization specialist.
Your job is to polish and finalize content to ensure it's ready for delivery.
Focus on final improvements in clarity, formatting, and overall quality.""")
        
        # Include relevant context from previous messages
        finalization_messages = [system_message]
        
        # Add previous messages for context (last 3 messages)
        if "messages" in state and state["messages"]:
            finalization_messages.extend(state["messages"][-3:])
        
        # Add the finalization instruction
        finalization_messages.append(HumanMessage(content=f"""
Task: {task}

Draft to Finalize:
{draft}

Human has approved this draft. Please finalize it with any minor improvements in:
1. Formatting and structure
2. Language and style
3. Overall presentation
4. Consistency with previous feedback

The output should be publication-ready and address all previous feedback points."""))
        
        # Get the finalized output with error handling
        response = model.invoke(finalization_messages)
        final_output = response.content
        
        # Update the message history
        updated_messages = state.get('messages', []).copy()
        updated_messages.append(AIMessage(content="Finalized content is ready for review."))
        
        # Update the state
        return {
            **state,
            "messages": updated_messages,
            "final_output": final_output,
            "approval_status": ApprovalStatus.APPROVED,
            "next": "end"
        }
        
    except Exception as e:
        print(f"\n⚠️ Error during finalization: {str(e)}")
        print("Falling back to the last draft as final output.")
        
        return {
            **state,
            "final_output": state["draft"],
            "approval_status": ApprovalStatus.APPROVED,
            "next": "end"
        }

def create_hitl_workflow(checkpointer = None):
    """Create a human-in-the-loop workflow using LangGraph."""
    print("Creating a human-in-the-loop workflow with LangGraph...")
    
    # Create a new graph
    workflow = StateGraph(HITLState)
    
    # Add nodes to the graph
    workflow.add_node("drafting", drafting_node)
    workflow.add_node("human_review", human_review_node)
    workflow.add_node("revision", revision_node)
    workflow.add_node("finalization", finalization_node)
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "drafting",
        lambda state: state["next"],
        {
            "human_review": "human_review"
        }
    )
    
    workflow.add_conditional_edges(
        "human_review",
        lambda state: state["next"],
        {
            "revision": "revision",
            "finalization": "finalization",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "revision",
        lambda state: state["next"],
        {
            "drafting": "drafting"
        }
    )
    
    workflow.add_conditional_edges(
        "finalization",
        lambda state: state["next"],
        {
            "end": END
        }
    )
    
    # Set the entry point
    workflow.set_entry_point("drafting")
    
    # Compile the graph
    return workflow.compile(checkpointer=checkpointer)

def run_hitl_workflow(workflow, task: str):
    """Run the human-in-the-loop workflow with a given task."""
    print("\n" + "=" * 50)
    print("Running the Human-in-the-Loop Workflow")
    print("=" * 50)
    
    # Initialize the state
    initial_state = {
        "messages": [HumanMessage(content=task)],
        "task": task,
        "draft": "",
        "human_feedback": "",
        "final_output": "",
        "approval_status": ApprovalStatus.PENDING,
        "revision_count": 0,
        "next": "drafting"
    }
    
    # Run the workflow
    print(f"\nTask: {task}")
    result = workflow.invoke(initial_state)
    
    # Print the workflow results
    print("\n" + "=" * 50)
    print("Workflow Results")
    print("=" * 50)
    
    print(f"\nTask: {result['task']}")
    print(f"Approval Status: {result['approval_status']}")
    print(f"Revision Count: {result['revision_count']}")
    
    if result["approval_status"] == ApprovalStatus.APPROVED:
        print("\nFinal Output:")
        print("-" * 50)
        print(result["final_output"])
    elif result["approval_status"] == ApprovalStatus.REJECTED:
        print("\nDraft (Rejected):")
        print("-" * 50)
        print(result["draft"])
        print("\nRejection Reason:")
        print(result["human_feedback"])
    
    return result

def create_persistent_hitl_workflow():
    """Create a persistent human-in-the-loop workflow using LangGraph with proper checkpointing."""
    print("Creating a persistent human-in-the-loop workflow with LangGraph...")
    
    # Create an in-memory checkpointer with a longer TTL for persistence
    checkpointer = InMemorySaver()  
    
    try:
        # Create a workflow with the checkpointer
        workflow = create_hitl_workflow(checkpointer)
        print("✅ Persistent workflow created successfully")
        return workflow
    except Exception as e:
        print(f"❌ Error creating persistent workflow: {str(e)}")
        raise

def run_persistent_hitl_workflow():
    """Run a persistent human-in-the-loop workflow with enhanced session management."""
    from datetime import datetime
    
    def print_header():
        print("\n" + "=" * 50)
        print("🚀 Persistent Human-in-the-Loop Workflow")
        print("=" * 50)
    
    def print_footer():
        print("\n" + "=" * 50)
        print("🏁 Workflow Session Ended")
        print("=" * 50)
    
    def get_user_choice(prompt: str, valid_choices: list, display_options: list = None):
        """Helper to get and validate user input with retry logic."""
        while True:
            try:
                if display_options:
                    print("\n" + "\n".join(display_options))
                choice = input(f"\n{prompt} ").strip().lower()
                if choice in valid_choices:
                    return choice
                print(f"❌ Invalid choice. Please choose from: {', '.join(valid_choices)}")
            except KeyboardInterrupt:
                print("\n\nOperation cancelled by user.")
                return None
            except Exception as e:
                print(f"An error occurred: {e}")
    
    def show_workflow_state(state: dict):
        """Display the current state of the workflow."""
        print("\n" + "📋 Workflow State ".ljust(50, "-"))
        print(f"Task: {state.get('task', 'N/A')}")
        print(f"Status: {state.get('approval_status', 'N/A')}")
        print(f"Revision: {state.get('revision_count', 0) + 1}")
        
        if 'draft' in state and state['draft']:
            print("\n📄 Current Draft Preview:")
            print("-" * 20)
            preview = state['draft'][:500]
            if len(state['draft']) > 500:
                preview += "... (truncated)"
            print(preview)
        
        if state.get('human_feedback'):
            print("\n💬 Latest Feedback:")
            print("\"" + state['human_feedback'] + "\"")
    
    try:
        print_header()
        workflow = create_persistent_hitl_workflow()
        
        # Session management
        choice = get_user_choice(
            "Choose an option (1-2):",
            ['1', '2'],
            ["1. Start a new workflow", "2. Resume existing workflow"]
        )
        if choice is None:
            return None
        
        if choice == '1':
            thread_id = str(uuid.uuid4())
            print(f"\n🚀 New workflow ID: {thread_id}")
            
            while True:
                task = input("\n📝 Enter your task (or 'exit' to cancel): ").strip()
                if task.lower() == 'exit':
                    print("\nOperation cancelled.")
                    return None
                if task:
                    break
                print("❌ Task cannot be empty. Please try again.")
            
            initial_state = {
                "messages": [
                    SystemMessage(content=f"Session started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"),
                    HumanMessage(content=task)
                ],
                "task": task,
                "draft": "",
                "human_feedback": "",
                "final_output": "",
                "approval_status": ApprovalStatus.PENDING,
                "revision_count": 0,
                "next": "drafting"
            }
            config = {"configurable": {"thread_id": thread_id}}
            print(f"\n✅ Starting workflow for task: {task}")
            result = workflow.invoke(initial_state, config=config)
            
        else:  # Resume workflow
            thread_id = input("\n🔍 Enter workflow ID to resume: ").strip()
            if not thread_id:
                print("❌ Workflow ID cannot be empty.")
                return None
                
            print(f"\n🔄 Resuming workflow: {thread_id}")
            config = {"configurable": {"thread_id": thread_id}}
            
            try:
                result = workflow.invoke({}, config=config)
                print(f"✅ Successfully loaded workflow. Current status: {result.get('approval_status', 'N/A')}")
            except Exception as e:
                print(f"❌ Failed to load workflow: {str(e)}")
                return None
        
        # Main interaction loop
        while result.get("next") != "end":
            action = get_user_choice(
                "Choose action (1-4):",
                ['1', '2', '3', '4'],
                [
                    "\n🔄 Workflow Paused",
                    "1. Continue execution",
                    "2. View current state",
                    "3. View message history",
                    "4. Abort workflow"
                ]
            )
            
            if action is None:
                return None
                
            if action == '1':
                result = workflow.invoke({}, config=config)
                continue
                
            elif action == '2':
                show_workflow_state(result)
                input("\nPress Enter to continue...")
                continue
                
            elif action == '3':
                print("\n📜 Message History:")
                print("-" * 50)
                for i, msg in enumerate(result.get('messages', [])[-5:], 1):
                    prefix = f"👤" if isinstance(msg, HumanMessage) else "🤖"
                    print(f"{prefix} {msg.content[:200]}{'...' if len(msg.content) > 200 else ''}")
                print("-" * 50)
                input("\nPress Enter to continue...")
                continue
                
            elif action == '4':
                confirm = get_user_choice(
                    "Are you sure you want to abort? (y/n):",
                    ['y', 'n']
                )
                if confirm == 'y':
                    print("\n⚠️ Workflow aborted by user.")
                    return result
        
        # Final output
        print("\n" + "✅ " + "Workflow Completed".center(46) + " ✅")
        print("=" * 50)
        
        if result.get("approval_status") == ApprovalStatus.APPROVED:
            print("\n🎉 Final Output:")
            print("-" * 50)
            print(result.get("final_output", "No output generated."))
        else:
            print("\n❌ Workflow did not complete successfully.")
            if result.get('human_feedback'):
                print("\nFeedback:", result['human_feedback'])
        
        print(f"\n💾 Workflow ID: {thread_id}")
        print(f"📅 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        return result
        
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        return None
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        print_footer()

def main():
    """Main function to demonstrate human-in-the-loop workflows."""
    print("=" * 50)
    print("Human-in-the-Loop Systems in LangGraph")
    print("=" * 50)
    
    # Let the user choose a mode
    print("\nSelect a mode:")
    print("1. Simple human-in-the-loop workflow")
    print("2. Persistent human-in-the-loop workflow (with checkpointing)")
    
    while True:
        try:
            choice = int(input("\nEnter your choice (1-2): "))
            if choice == 1:
                # Create a simple workflow
                workflow = create_hitl_workflow()
                
                # Get a task from the user
                task = input("\nEnter a task (e.g., 'Write a blog post about machine learning'): ")
                
                # Run the workflow
                run_hitl_workflow(workflow, task)
                break
            elif choice == 2:
                # Run a persistent workflow
                run_persistent_hitl_workflow()
                break
            else:
                print("Please enter 1 or 2.")
        except ValueError:
            print("Please enter a valid number.")

if __name__ == "__main__":
    main()
