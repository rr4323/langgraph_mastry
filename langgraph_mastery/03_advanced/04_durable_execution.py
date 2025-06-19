"""
Durable Execution in LangGraph

This script demonstrates how to implement durable execution in LangGraph,
allowing workflows to persist through failures and resume from where they left off.
"""

import os
import sys
import json
import logging
import asyncio
import tempfile
from typing import Dict, List, Any, Optional
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
from langgraph.checkpoint.sqlite import SqliteSaver

# Define our state model
class TaskStep(BaseModel):
    """A step in a task workflow."""
    name: str
    description: str
    completed: bool = False
    result: Optional[str] = None
    error: Optional[str] = None

class WorkflowState(BaseModel):
    """The state for our durable workflow."""
    task_id: str
    query: str
    steps: List[TaskStep] = Field(default_factory=list)
    current_step_index: int = 0
    status: str = "in_progress"  # in_progress, completed, failed
    result: Optional[str] = None
    errors: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    last_checkpoint: Optional[str] = None

# Helper functions
def get_checkpoint_dir():
    """Get the directory for storing checkpoints."""
    checkpoint_dir = os.path.join(tempfile.gettempdir(), "langgraph_checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    return checkpoint_dir

def get_model(google_api_key: str):
    """Get the language model."""
    return ChatGoogleGenerativeAI(
        model="gemini-pro",
        google_api_key=google_api_key,
        temperature=0.4,
        convert_system_message_to_human=True
    )

# Node functions for our workflow
async def research_step(state: Dict[str, Any], model) -> Dict[str, Any]:
    """Research step in the workflow."""
    logger.info("Running research step")
    state_obj = WorkflowState.model_validate(state)
    
    # Get the current step
    if state_obj.current_step_index >= len(state_obj.steps):
        state_obj.errors.append({
            "step": "research",
            "error": "Step index out of bounds",
            "timestamp": datetime.now().isoformat()
        })
        return state_obj.model_dump()
    
    current_step = state_obj.steps[state_obj.current_step_index]
    
    # Check if this step is already completed
    if current_step.completed:
        logger.info("Research step already completed, skipping")
        return state_obj.model_dump()
    
    # Simulate research work
    try:
        logger.info("Performing research...")
        
        # Use the model to generate research results
        messages = [
            {"role": "system", "content": "You are a research assistant. Provide relevant information for the given query."},
            {"role": "user", "content": f"Research query: {state_obj.query}"}
        ]
        
        response = await model.ainvoke(messages)
        research_result = response.content
        
        # Update the step
        current_step.completed = True
        current_step.result = research_result
        
        # Move to the next step
        state_obj.current_step_index += 1
        
    except Exception as e:
        logger.error(f"Error in research step: {str(e)}", exc_info=True)
        current_step.error = str(e)
        state_obj.errors.append({
            "step": "research",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        })
    
    # Record checkpoint
    state_obj.last_checkpoint = "after_research"
    
    return state_obj.model_dump()

async def analysis_step(state: Dict[str, Any], model) -> Dict[str, Any]:
    """Analysis step in the workflow."""
    logger.info("Running analysis step")
    state_obj = WorkflowState.model_validate(state)
    
    # Get the current step
    if state_obj.current_step_index >= len(state_obj.steps):
        state_obj.errors.append({
            "step": "analysis",
            "error": "Step index out of bounds",
            "timestamp": datetime.now().isoformat()
        })
        return state_obj.model_dump()
    
    current_step = state_obj.steps[state_obj.current_step_index]
    
    # Check if this step is already completed
    if current_step.completed:
        logger.info("Analysis step already completed, skipping")
        return state_obj.model_dump()
    
    # Get the research results
    research_step = next((step for step in state_obj.steps if step.name == "research"), None)
    if not research_step or not research_step.completed:
        state_obj.errors.append({
            "step": "analysis",
            "error": "Research step not completed",
            "timestamp": datetime.now().isoformat()
        })
        return state_obj.model_dump()
    
    # Simulate analysis work
    try:
        logger.info("Performing analysis...")
        
        # Use the model to generate analysis
        messages = [
            {"role": "system", "content": "You are an analyst. Analyze the research information and provide insights."},
            {"role": "user", "content": f"Query: {state_obj.query}\n\nResearch information: {research_step.result}"}
        ]
        
        response = await model.ainvoke(messages)
        analysis_result = response.content
        
        # Update the step
        current_step.completed = True
        current_step.result = analysis_result
        
        # Move to the next step
        state_obj.current_step_index += 1
        
    except Exception as e:
        logger.error(f"Error in analysis step: {str(e)}", exc_info=True)
        current_step.error = str(e)
        state_obj.errors.append({
            "step": "analysis",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        })
    
    # Record checkpoint
    state_obj.last_checkpoint = "after_analysis"
    
    return state_obj.model_dump()

async def report_step(state: Dict[str, Any], model) -> Dict[str, Any]:
    """Report generation step in the workflow."""
    logger.info("Running report step")
    state_obj = WorkflowState.model_validate(state)
    
    # Get the current step
    if state_obj.current_step_index >= len(state_obj.steps):
        state_obj.errors.append({
            "step": "report",
            "error": "Step index out of bounds",
            "timestamp": datetime.now().isoformat()
        })
        return state_obj.model_dump()
    
    current_step = state_obj.steps[state_obj.current_step_index]
    
    # Check if this step is already completed
    if current_step.completed:
        logger.info("Report step already completed, skipping")
        return state_obj.model_dump()
    
    # Get the research and analysis results
    research_step = next((step for step in state_obj.steps if step.name == "research"), None)
    analysis_step = next((step for step in state_obj.steps if step.name == "analysis"), None)
    
    if not research_step or not research_step.completed or not analysis_step or not analysis_step.completed:
        state_obj.errors.append({
            "step": "report",
            "error": "Previous steps not completed",
            "timestamp": datetime.now().isoformat()
        })
        return state_obj.model_dump()
    
    # Simulate report generation
    try:
        logger.info("Generating report...")
        
        # Use the model to generate a report
        messages = [
            {"role": "system", "content": "You are a report writer. Create a comprehensive report based on research and analysis."},
            {"role": "user", "content": f"Query: {state_obj.query}\n\nResearch: {research_step.result}\n\nAnalysis: {analysis_step.result}"}
        ]
        
        response = await model.ainvoke(messages)
        report_result = response.content
        
        # Update the step
        current_step.completed = True
        current_step.result = report_result
        
        # Set the final result
        state_obj.result = report_result
        state_obj.status = "completed"
        
        # Move to the next step (which will be END)
        state_obj.current_step_index += 1
        
    except Exception as e:
        logger.error(f"Error in report step: {str(e)}", exc_info=True)
        current_step.error = str(e)
        state_obj.errors.append({
            "step": "report",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        })
    
    # Record checkpoint
    state_obj.last_checkpoint = "after_report"
    
    return state_obj.model_dump()

# Function to check if there are errors
def has_errors(state: Dict[str, Any]) -> bool:
    """Check if the state has errors."""
    state_obj = WorkflowState.model_validate(state)
    return len(state_obj.errors) > 0

# Function to determine the next step
def get_next_step(state: Dict[str, Any]) -> str:
    """Determine the next step in the workflow."""
    state_obj = WorkflowState.model_validate(state)
    
    # If we've completed all steps, we're done
    if state_obj.current_step_index >= len(state_obj.steps):
        return "end"
    
    # Get the current step
    current_step = state_obj.steps[state_obj.current_step_index]
    
    # Return the step name
    return current_step.name

# Create the workflow graph with checkpointing
def create_workflow_graph(model, checkpoint_dir: str):
    """Create the workflow graph with checkpointing."""
    # Create a checkpoint saver
    checkpoint_file = os.path.join(checkpoint_dir, "workflow_checkpoints.db")
    saver = SqliteSaver(checkpoint_file)
    
    # Create the graph
    workflow = StateGraph(WorkflowState, checkpointer=saver)
    
    # Add nodes to the graph
    workflow.add_node("research", lambda state: research_step(state, model))
    workflow.add_node("analysis", lambda state: analysis_step(state, model))
    workflow.add_node("report", lambda state: report_step(state, model))
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "research",
        has_errors,
        {
            True: END,
            False: "analysis"
        }
    )
    
    workflow.add_conditional_edges(
        "analysis",
        has_errors,
        {
            True: END,
            False: "report"
        }
    )
    
    workflow.add_conditional_edges(
        "report",
        has_errors,
        {
            True: END,
            False: END
        }
    )
    
    # Set the entry point based on the current step
    workflow.set_entry_point(get_next_step)
    
    # Compile the graph
    return workflow.compile()

# Function to create a new workflow
def create_new_workflow(task_id: str, query: str) -> WorkflowState:
    """Create a new workflow state."""
    return WorkflowState(
        task_id=task_id,
        query=query,
        steps=[
            TaskStep(name="research", description="Research information related to the query"),
            TaskStep(name="analysis", description="Analyze the research information"),
            TaskStep(name="report", description="Generate a comprehensive report")
        ],
        metadata={
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
    )

# Function to save workflow state to a file
def save_workflow_state(state: Dict[str, Any], file_path: str):
    """Save the workflow state to a file."""
    with open(file_path, "w") as f:
        json.dump(state, f, indent=2)

# Function to load workflow state from a file
def load_workflow_state(file_path: str) -> Dict[str, Any]:
    """Load the workflow state from a file."""
    with open(file_path, "r") as f:
        return json.load(f)

# Main function to run the workflow
async def run_workflow(task_id: str, query: str, resume: bool = False, state_file: Optional[str] = None):
    """Run the workflow with durable execution."""
    logger.info(f"Running workflow for task {task_id}")
    
    # Get the Google API key from environment
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        logger.error("GOOGLE_API_KEY not found in environment variables")
        print("Please set your GOOGLE_API_KEY in the .env file")
        return {"error": "API key not found"}
    
    # Get the model
    model = get_model(google_api_key)
    
    # Get the checkpoint directory
    checkpoint_dir = get_checkpoint_dir()
    
    # Create the workflow graph
    app = create_workflow_graph(model, checkpoint_dir)
    
    # Initialize or resume the workflow
    if resume and state_file and os.path.exists(state_file):
        logger.info(f"Resuming workflow from state file: {state_file}")
        state_dict = load_workflow_state(state_file)
        state = WorkflowState.model_validate(state_dict)
        logger.info(f"Resuming from step: {state.current_step_index}")
    else:
        logger.info("Creating new workflow")
        state = create_new_workflow(task_id, query)
    
    # Run the workflow
    try:
        logger.info("Running the workflow...")
        result = await app.ainvoke(state.model_dump())
        
        # Save the final state
        if state_file:
            save_workflow_state(result, state_file)
        
        return result
    except Exception as e:
        logger.error(f"Error running workflow: {str(e)}", exc_info=True)
        state.errors.append({
            "step": "workflow",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        })
        state.status = "failed"
        
        # Save the error state
        if state_file:
            save_workflow_state(state.model_dump(), state_file)
        
        return state.model_dump()

async def main():
    """Run the durable execution example."""
    print("=" * 80)
    print("Durable Execution in LangGraph")
    print("=" * 80)
    print("\nThis example demonstrates how to implement durable execution in LangGraph,")
    print("allowing workflows to persist through failures and resume from where they left off.")
    
    # Get user input
    query = input("\nEnter your query: ")
    task_id = f"task_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    # Create a state file
    state_file = os.path.join(get_checkpoint_dir(), f"{task_id}.json")
    
    # Run the workflow
    print("\nRunning workflow...")
    result = await run_workflow(task_id, query, resume=False, state_file=state_file)
    
    # Display the result
    print("\n" + "=" * 80)
    print("Workflow Result:")
    if result.get("status") == "completed":
        print("\nTask completed successfully!")
        print("\nFinal Report:")
        print(result.get("result", "No result generated."))
    else:
        print("\nTask did not complete.")
        if "errors" in result and result["errors"]:
            print("\nErrors:")
            for error in result["errors"]:
                print(f"- {error.get('step', 'unknown')}: {error.get('error', 'unknown error')}")
    
    print("\n" + "=" * 80)
    print(f"Workflow state saved to: {state_file}")
    print("\nYou can resume this workflow by running:")
    print(f"python 04_durable_execution.py --resume --state-file {state_file}")
    
    # Simulate a failure and resume
    if result.get("status") == "completed":
        print("\nWould you like to simulate a failure and resume the workflow? (y/n)")
        choice = input().strip().lower()
        if choice == "y":
            # Modify the state to simulate a failure
            state_dict = load_workflow_state(state_file)
            state = WorkflowState.model_validate(state_dict)
            
            # Reset the last step
            if len(state.steps) > 0:
                state.steps[-1].completed = False
                state.current_step_index = len(state.steps) - 1
                state.status = "in_progress"
                state.result = None
                
                # Save the modified state
                save_workflow_state(state.model_dump(), state_file)
                
                print("\nSimulated a failure in the last step. Resuming workflow...")
                result = await run_workflow(task_id, query, resume=True, state_file=state_file)
                
                # Display the resumed result
                print("\n" + "=" * 80)
                print("Resumed Workflow Result:")
                if result.get("status") == "completed":
                    print("\nTask completed successfully after resuming!")
                    print("\nFinal Report:")
                    print(result.get("result", "No result generated."))
                else:
                    print("\nTask did not complete after resuming.")
                    if "errors" in result and result["errors"]:
                        print("\nErrors:")
                        for error in result["errors"]:
                            print(f"- {error.get('step', 'unknown')}: {error.get('error', 'unknown error')}")

if __name__ == "__main__":
    # Check if we're resuming a workflow
    if len(sys.argv) > 1 and "--resume" in sys.argv:
        state_file_index = sys.argv.index("--state-file") + 1 if "--state-file" in sys.argv else -1
        if state_file_index > 0 and state_file_index < len(sys.argv):
            state_file = sys.argv[state_file_index]
            if os.path.exists(state_file):
                state_dict = load_workflow_state(state_file)
                task_id = state_dict.get("task_id", "unknown_task")
                query = state_dict.get("query", "unknown_query")
                print(f"Resuming workflow for task {task_id}...")
                asyncio.run(run_workflow(task_id, query, resume=True, state_file=state_file))
            else:
                print(f"State file not found: {state_file}")
        else:
            print("Please specify a state file with --state-file")
    else:
        asyncio.run(main())
