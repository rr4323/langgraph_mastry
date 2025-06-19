"""
Human-in-the-Loop Integration in LangGraph

This script demonstrates how to implement human-in-the-loop workflows in LangGraph,
allowing for human oversight, intervention, and collaboration with AI agents.
"""

import os
import sys
import json
import uuid
import logging
import asyncio
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Check for required packages
required_packages = ["langchain-google-genai", "langgraph"]
for package in required_packages:
    try:
        __import__(package.replace("-", "_"))
    except ImportError:
        logger.info(f"Installing {package}...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.graph.graph import CompiledGraph

# Define our state models
class HumanFeedback(BaseModel):
    """Feedback from a human reviewer."""
    approved: bool
    comments: Optional[str] = None
    modifications: Optional[Dict[str, Any]] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

class ContentDraft(BaseModel):
    """A draft of content created by an AI."""
    content: str
    version: int = 1
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    feedback: Optional[HumanFeedback] = None

class HITLState(BaseModel):
    """The state for our human-in-the-loop workflow."""
    task_id: str
    prompt: str
    drafts: List[ContentDraft] = Field(default_factory=list)
    current_draft_index: int = 0
    status: str = "draft_in_progress"  # draft_in_progress, awaiting_feedback, completed, rejected
    final_content: Optional[str] = None
    errors: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    human_in_loop: bool = True  # Flag to enable/disable human review

# Helper functions
def get_model(google_api_key: str):
    """Get the language model."""
    return ChatGoogleGenerativeAI(
        model="gemini-pro",
        google_api_key=google_api_key,
        temperature=0.7,
        convert_system_message_to_human=True
    )

# Node functions for our workflow
async def generate_draft(state: Dict[str, Any], model) -> Dict[str, Any]:
    """Generate a content draft."""
    logger.info("Generating content draft")
    state_obj = HITLState.model_validate(state)
    
    try:
        # Get the prompt and any previous drafts with feedback
        prompt = state_obj.prompt
        previous_drafts = []
        
        for i, draft in enumerate(state_obj.drafts):
            if draft.feedback:
                previous_drafts.append(f"Draft {i+1}:\n{draft.content}\n\nFeedback: {draft.feedback.comments}")
        
        # Prepare the messages for the model
        messages = [
            {"role": "system", "content": "You are a professional content creator. Create high-quality content based on the given prompt."}
        ]
        
        if previous_drafts:
            messages.append({
                "role": "user", 
                "content": f"Create content for the following prompt:\n\n{prompt}\n\nPrevious drafts and feedback:\n\n" + "\n\n".join(previous_drafts)
            })
        else:
            messages.append({
                "role": "user", 
                "content": f"Create content for the following prompt:\n\n{prompt}"
            })
        
        # Generate the content
        response = await model.ainvoke(messages)
        draft_content = response.content
        
        # Create a new draft
        new_draft = ContentDraft(
            content=draft_content,
            version=len(state_obj.drafts) + 1
        )
        
        # Add the draft to the state
        state_obj.drafts.append(new_draft)
        state_obj.current_draft_index = len(state_obj.drafts) - 1
        
        # Update the status
        if state_obj.human_in_loop:
            state_obj.status = "awaiting_feedback"
        else:
            # If no human in the loop, auto-approve
            state_obj.status = "completed"
            state_obj.final_content = draft_content
        
    except Exception as e:
        logger.error(f"Error generating draft: {str(e)}", exc_info=True)
        state_obj.errors.append({
            "step": "generate_draft",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        })
    
    return state_obj.model_dump()

async def process_feedback(state: Dict[str, Any], model) -> Dict[str, Any]:
    """Process human feedback on a draft."""
    logger.info("Processing feedback")
    state_obj = HITLState.model_validate(state)
    
    # Get the current draft
    if not state_obj.drafts or state_obj.current_draft_index >= len(state_obj.drafts):
        state_obj.errors.append({
            "step": "process_feedback",
            "error": "No draft available for feedback",
            "timestamp": datetime.now().isoformat()
        })
        return state_obj.model_dump()
    
    current_draft = state_obj.drafts[state_obj.current_draft_index]
    
    # Check if we have feedback
    if not current_draft.feedback:
        logger.warning("No feedback available to process")
        return state_obj.model_dump()
    
    # Process the feedback
    feedback = current_draft.feedback
    
    if feedback.approved:
        # Draft is approved, mark as completed
        state_obj.status = "completed"
        state_obj.final_content = current_draft.content
        logger.info("Draft approved, workflow completed")
    else:
        # Draft needs revision, go back to draft generation
        state_obj.status = "draft_in_progress"
        logger.info("Draft rejected, generating new version")
    
    return state_obj.model_dump()

# Function to determine the next step
def get_next_step(state: Dict[str, Any]) -> str:
    """Determine the next step in the workflow."""
    state_obj = HITLState.model_validate(state)
    
    if state_obj.status == "draft_in_progress":
        return "generate_draft"
    elif state_obj.status == "awaiting_feedback":
        return "await_human"
    elif state_obj.status == "completed":
        return "end"
    else:
        # Handle error states
        return "end"

# Human-in-the-loop functions
def await_human_feedback(state: Dict[str, Any]) -> Dict[str, Any]:
    """Wait for human feedback on the current draft."""
    logger.info("Awaiting human feedback")
    state_obj = HITLState.model_validate(state)
    
    # This is a special node that will pause execution until human feedback is provided
    # The actual implementation would depend on your application's UI/UX
    
    return state_obj.model_dump()

def provide_human_feedback(
    state: Dict[str, Any],
    approved: bool,
    comments: Optional[str] = None,
    modifications: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Provide human feedback on a draft."""
    logger.info(f"Providing human feedback: approved={approved}")
    state_obj = HITLState.model_validate(state)
    
    # Get the current draft
    if not state_obj.drafts or state_obj.current_draft_index >= len(state_obj.drafts):
        state_obj.errors.append({
            "step": "provide_feedback",
            "error": "No draft available for feedback",
            "timestamp": datetime.now().isoformat()
        })
        return state_obj.model_dump()
    
    current_draft = state_obj.drafts[state_obj.current_draft_index]
    
    # Add feedback to the draft
    current_draft.feedback = HumanFeedback(
        approved=approved,
        comments=comments,
        modifications=modifications
    )
    
    # Update the state
    state_obj.drafts[state_obj.current_draft_index] = current_draft
    
    # Move to the next step
    if approved:
        state_obj.status = "completed"
        state_obj.final_content = current_draft.content
    else:
        state_obj.status = "draft_in_progress"
    
    return state_obj.model_dump()

# Create the workflow graph
def create_hitl_graph(model):
    """Create the human-in-the-loop workflow graph."""
    workflow = StateGraph(HITLState)
    
    # Add nodes to the graph
    workflow.add_node("generate_draft", lambda state: generate_draft(state, model))
    workflow.add_node("process_feedback", lambda state: process_feedback(state, model))
    workflow.add_node("await_human", await_human_feedback)
    
    # Add edges
    workflow.add_edge("generate_draft", "await_human")
    workflow.add_edge("await_human", "process_feedback")
    workflow.add_edge("process_feedback", get_next_step)
    
    # Set the entry point
    workflow.set_entry_point(get_next_step)
    
    # Compile the graph
    return workflow.compile()

# Human-in-the-Loop Workflow Manager
class HITLWorkflowManager:
    """Manager for human-in-the-loop workflows."""
    
    def __init__(self, google_api_key: str):
        """Initialize the manager."""
        self.model = get_model(google_api_key)
        self.graph = create_hitl_graph(self.model)
        self.active_workflows: Dict[str, Dict[str, Any]] = {}
    
    async def create_workflow(self, prompt: str, human_in_loop: bool = True) -> str:
        """Create a new workflow."""
        task_id = str(uuid.uuid4())
        
        # Initialize the state
        state = HITLState(
            task_id=task_id,
            prompt=prompt,
            human_in_loop=human_in_loop,
            metadata={
                "created_at": datetime.now().isoformat()
            }
        )
        
        # Start the workflow
        result = await self.graph.ainvoke(state.model_dump())
        
        # Store the workflow state
        self.active_workflows[task_id] = result
        
        return task_id
    
    def get_workflow_state(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get the current state of a workflow."""
        return self.active_workflows.get(task_id)
    
    async def provide_feedback(
        self,
        task_id: str,
        approved: bool,
        comments: Optional[str] = None,
        modifications: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Provide human feedback for a workflow."""
        if task_id not in self.active_workflows:
            raise ValueError(f"Workflow {task_id} not found")
        
        # Get the current state
        state = self.active_workflows[task_id]
        
        # Add the feedback
        state = provide_human_feedback(state, approved, comments, modifications)
        
        # Continue the workflow
        result = await self.graph.ainvoke(state)
        
        # Update the stored state
        self.active_workflows[task_id] = result
        
        return result
    
    def get_current_draft(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get the current draft for a workflow."""
        state = self.get_workflow_state(task_id)
        if not state:
            return None
        
        state_obj = HITLState.model_validate(state)
        if not state_obj.drafts or state_obj.current_draft_index >= len(state_obj.drafts):
            return None
        
        return state_obj.drafts[state_obj.current_draft_index].model_dump()
    
    def is_awaiting_feedback(self, task_id: str) -> bool:
        """Check if a workflow is awaiting feedback."""
        state = self.get_workflow_state(task_id)
        if not state:
            return False
        
        return state.get("status") == "awaiting_feedback"
    
    def is_completed(self, task_id: str) -> bool:
        """Check if a workflow is completed."""
        state = self.get_workflow_state(task_id)
        if not state:
            return False
        
        return state.get("status") == "completed"
    
    def get_final_content(self, task_id: str) -> Optional[str]:
        """Get the final content for a completed workflow."""
        state = self.get_workflow_state(task_id)
        if not state:
            return None
        
        return state.get("final_content")

# Interactive CLI for the human-in-the-loop example
async def run_interactive_cli():
    """Run an interactive CLI for the human-in-the-loop example."""
    print("=" * 80)
    print("Human-in-the-Loop Integration in LangGraph")
    print("=" * 80)
    print("\nThis example demonstrates how to implement human-in-the-loop workflows,")
    print("allowing for human oversight, intervention, and collaboration with AI agents.")
    
    # Get the Google API key from environment
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        print("ERROR: GOOGLE_API_KEY not found in environment variables")
        print("Please set your GOOGLE_API_KEY in the .env file")
        return
    
    # Create the workflow manager
    manager = HITLWorkflowManager(google_api_key)
    
    # Get user input
    print("\nEnter a content creation prompt:")
    prompt = input("> ")
    
    # Ask if human review should be enabled
    print("\nEnable human review? (y/n)")
    human_in_loop = input("> ").lower() == "y"
    
    # Create the workflow
    print("\nCreating workflow...")
    task_id = await manager.create_workflow(prompt, human_in_loop)
    print(f"Workflow created with ID: {task_id}")
    
    # Main interaction loop
    while True:
        # Get the current state
        state = manager.get_workflow_state(task_id)
        state_obj = HITLState.model_validate(state)
        
        # Display the current status
        print("\n" + "=" * 80)
        print(f"Status: {state_obj.status}")
        print("=" * 80)
        
        # If we're awaiting feedback, show the draft and get feedback
        if state_obj.status == "awaiting_feedback":
            current_draft = manager.get_current_draft(task_id)
            if current_draft:
                print(f"\nDraft {current_draft['version']}:")
                print("-" * 40)
                print(current_draft["content"])
                print("-" * 40)
                
                print("\nApprove this draft? (y/n)")
                approved = input("> ").lower() == "y"
                
                if not approved:
                    print("\nProvide feedback comments:")
                    comments = input("> ")
                    
                    # Process the feedback
                    print("\nProcessing feedback...")
                    await manager.provide_feedback(task_id, approved, comments)
                else:
                    # Approve the draft
                    print("\nApproving draft...")
                    await manager.provide_feedback(task_id, approved, "Looks good!")
            else:
                print("No draft available")
                break
        
        # If the workflow is completed, show the final content
        elif state_obj.status == "completed":
            print("\nWorkflow completed!")
            print("\nFinal Content:")
            print("-" * 40)
            print(state_obj.final_content)
            print("-" * 40)
            break
        
        # If there's an error or other status, break the loop
        else:
            print(f"Workflow status: {state_obj.status}")
            if state_obj.errors:
                print("\nErrors:")
                for error in state_obj.errors:
                    print(f"- {error.get('step', 'unknown')}: {error.get('error', 'unknown error')}")
            break
    
    print("\nThank you for using the Human-in-the-Loop example!")

# Main function
if __name__ == "__main__":
    asyncio.run(run_interactive_cli())
