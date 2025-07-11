"""
Human-in-the-Loop Integration in LangGraph
This script demonstrates how to implement human-in-the-loop workflows in LangGraph,
allowing for human oversight, intervention, and collaboration with AI agents.
This updated version leverages LangGraph's built-in checkpointers and interrupts
to simplify state management and create a more robust and idiomatic workflow.
"""

import os
import sys
import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import logging
import asyncio

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Check for required packages
required_packages = ["langchain-google-genai", "langgraph", "pydantic", "python-dotenv"]
for package in required_packages:
    try:
        __import__(package.replace("-", "_"))
    except ImportError:
        logger.info(f"Installing {package}...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.graph import CompiledGraph


# --- State Models (Pydantic) ---
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
    human_in_loop: bool = True


# --- Helper and Node Functions ---
def get_model(google_api_key: str):
    """Get the language model."""
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=google_api_key,
        temperature=0.7,
        convert_system_message_to_human=True
    )


async def generate_draft(state: HITLState, model) -> Dict[str, Any]:
    """
    Generates a content draft. This node's responsibility is now focused
    solely on content generation, not on setting the workflow status.
    """
    logger.info("Generating content draft")
    try:
        prompt = state.prompt
        previous_drafts_feedback = [
            f"Draft {i + 1}:\n{draft.content}\nFeedback: {draft.feedback.comments}"
            for i, draft in enumerate(state.drafts) if draft.feedback
        ]
        system_message = "You are a professional content creator. Create high-quality content based on the given prompt."
        user_message_content = f"Create content for the following prompt:\n{prompt}"

        if previous_drafts_feedback:
            user_message_content += "\n\nPrevious drafts and feedback:\n" + "\n".join(previous_drafts_feedback)

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message_content}
        ]

        response = await model.ainvoke(messages)
        new_draft = ContentDraft(content=response.content, version=len(state.drafts) + 1)
        state.drafts.append(new_draft)
        state.current_draft_index = len(state.drafts) - 1
        state.status = "awaiting_feedback"  # Updated status

    except Exception as e:
        logger.error(f"Error generating draft: {str(e)}", exc_info=True)
        state.errors.append({"step": "generate_draft", "error": str(e), "timestamp": datetime.now().isoformat()})

    return state.model_dump()


# --- Graph Definition ---
def create_hitl_graph(model) -> CompiledGraph:
    """
    Creates the human-in-the-loop workflow graph using interrupts.
    The graph's structure now explicitly handles the flow of control.
    """
    workflow = StateGraph(HITLState)

    async def call_generate_draft(state_dict: dict) -> dict:
        state = HITLState.model_validate(state_dict)
        return await generate_draft(state, model)

    def finalize_draft(state: HITLState) -> dict:
        current_draft = state.drafts[state.current_draft_index]
        state.final_content = current_draft.content
        state.status = "completed"
        return state.model_dump()

    workflow.add_node("generate_draft", call_generate_draft)
    workflow.add_node("human_review", lambda s: s)
    workflow.add_node("finalize", finalize_draft)

    def should_request_review(state_dict: dict) -> str:
        state = HITLState.model_validate(state_dict)
        if state.human_in_loop:
            logger.info("Draft requires human review.")
            return "human_review"
        else:
            logger.info("Auto-approving draft as human-in-loop is disabled.")
            state.final_content = state.drafts[-1].content if state.drafts else ""
            state.status = "completed"
            return "finalize"

    def route_after_feedback(state_dict: dict) -> str:
        state = HITLState.model_validate(state_dict)
        current_draft = state.drafts[state.current_draft_index]
        print(current_draft.feedback)
        if not current_draft.feedback:
            logger.error("No feedback found on current draft. Ending workflow.")
            state.status = "completed"
            return 'finalize'
        if current_draft.feedback.approved:
            logger.info("Draft approved. Completing workflow.")
            state.final_content = current_draft.content
            state.status = "completed"
            return 'finalize'
        else:
            logger.info("Draft rejected. Returning to generate a new version.")
            state.status = "draft_in_progress"
            return "generate_draft"

    workflow.set_entry_point("generate_draft")

    workflow.add_conditional_edges(
        "generate_draft",
        should_request_review,
        {"human_review": "human_review", "finalize": "finalize"}
    )

    workflow.add_conditional_edges(
        "human_review",
        route_after_feedback,
        {"generate_draft": "generate_draft", "finalize": "finalize"}
    )
    workflow.add_edge("finalize", END)

    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer, interrupt_before=["human_review"])


# --- Workflow Manager ---
class HITLWorkflowManager:
    """
    Manages human-in-the-loop workflows using LangGraph's persistence.
    This class is now much simpler as it doesn't manage state directly.
    """
    def __init__(self, google_api_key: str):
        self.model = get_model(google_api_key)
        self.graph = create_hitl_graph(self.model)

    async def create_workflow(self, prompt: str, human_in_loop: bool = True) -> str:
        """Creates and starts a new workflow, returning its unique ID."""
        task_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": task_id}}
        initial_state = HITLState(
            task_id=task_id,
            prompt=prompt,
            human_in_loop=human_in_loop,
            metadata={"created_at": datetime.now().isoformat()}
        )
        await self.graph.ainvoke(initial_state.model_dump(), config)
        return task_id

    def get_workflow_state(self, task_id: str) -> Optional[HITLState]:
        """Gets the current state of a workflow from the checkpointer."""
        config = {"configurable": {"thread_id": task_id}}
        snapshot = self.graph.get_state(config)
        return HITLState.model_validate(snapshot.values) if snapshot else None

    async def provide_feedback(self, task_id: str, approved: bool, comments: Optional[str] = None) -> HITLState:
        """Provides feedback and resumes the workflow."""
        config = {"configurable": {"thread_id": task_id}}
        state = self.get_workflow_state(task_id)
        if not state:
            raise ValueError(f"Workflow {task_id} not found or has no state.")

        current_draft = state.drafts[state.current_draft_index]
        current_draft.feedback = HumanFeedback(approved=approved, comments=comments)
        state.status = "approved" if approved else "rejected"
        self.graph.update_state(config, state.model_dump())
        # Resume the workflow
        print(state.drafts[state.current_draft_index].feedback)
        result = await self.graph.ainvoke(None, config)
        return HITLState.model_validate(result)

    def get_current_draft(self, task_id: str) -> Optional[ContentDraft]:
        state = self.get_workflow_state(task_id)
        if state and state.drafts:
            return state.drafts[state.current_draft_index]
        return None


# --- Interactive CLI (Updated for new manager) ---
async def run_interactive_cli():
    """Run an interactive CLI for the human-in-the-loop example."""
    print("=" * 80)
    print("Human-in-the-Loop Integration in LangGraph (Refactored)")
    print("=" * 80)

    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        print("\nERROR: GOOGLE_API_KEY not found. Please set it in a .env file.")
        return

    manager = HITLWorkflowManager(google_api_key)

    prompt = input("\nEnter a content creation prompt:\n> ")

    human_in_loop_input = input("\nEnable human review? (y/n) [y]: ").lower()
    human_in_loop = human_in_loop_input != 'n'

    print("\nCreating workflow...")
    task_id = await manager.create_workflow(prompt, human_in_loop)
    print(f"Workflow created with ID: {task_id}")

    while True:
        state = manager.get_workflow_state(task_id)
        if not state:
            print("Workflow has ended unexpectedly.")
            break

        print("\n" + "=" * 80)
        print(f"Workflow Status: {state.status.upper()}")
        print("=" * 80)
        print(state.status)
        if state.status == "completed":
            print("\nWorkflow completed!")
            print("\nFinal Content:")
            print("-" * 40)
            print(state.final_content)
            print("-" * 40)
            break

        current_draft = manager.get_current_draft(task_id)
        if current_draft:
            # print(f"\nDraft {current_draft.version}:")
            # print("-" * 40)
            # print(current_draft.content)
            # print("-" * 40)

            approved_input = input("\nApprove this draft? (y/n) [y]: ").lower()
            approved = approved_input != 'n'
            comments = None
            
            if not approved:
                comments = input("\nProvide feedback comments for the next version:\n> ")

            print("\nProcessing feedback...")
            print(approved, comments)
            await manager.provide_feedback(task_id, approved, comments or "Approved")
        else:
            print("Error: Workflow is paused but no draft is available.")
            break

    print("\nThank you for using the Human-in-the-Loop example!")


if __name__ == "__main__":
    asyncio.run(run_interactive_cli())