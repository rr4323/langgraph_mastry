"""
Main workflow graph for the Enterprise Knowledge Assistant.

This module defines the main workflow graph that orchestrates the entire
knowledge assistant process, from query understanding to response generation.
"""

import logging
from typing import Dict, Any, List, Callable

from langgraph.graph import StateGraph, END
from src.models.state import AssistantState, ErrorSeverity
from src.config import get_config
from src.graphs.nodes import (
    query_understanding,
    knowledge_retrieval,
    response_generation,
    error_handling,
    feedback_collection,
    memory_management,
    document_processing
)

logger = logging.getLogger(__name__)
config = get_config()

def route_initial_request(state: AssistantState) -> str:
    """
    Determines the initial path based on the input state.
    Routes to document processing or query understanding.
    """
    logger.info("Routing initial request")
    if state.document_path:
        logger.info("Routing to document processing.")
        return "document_processing"
    if state.query:
        logger.info("Routing to query understanding.")
        return "query_understanding"
    
    logger.error("Invalid input: No document_path or query provided.")
    return "invalid_input"

def create_main_graph() -> StateGraph:
    """Create the main workflow graph for the Enterprise Knowledge Assistant.
    
    Returns:
        StateGraph: The compiled workflow graph.
    """
    logger.info("Creating main workflow graph")
    
    # Create a new graph
    workflow = StateGraph(AssistantState)
    
    # --- Add Nodes ---
    workflow.add_node("router", route_initial_request)
    workflow.add_node("document_processing", document_processing.run)
    workflow.add_node("query_understanding", query_understanding.run)
    workflow.add_node("knowledge_retrieval", knowledge_retrieval.run)
    workflow.add_node("response_generation", response_generation.run)
    workflow.add_node("error_handling", error_handling.run)
    
    # Add optional nodes based on configuration
    if config.enable_human_feedback:
        workflow.add_node("feedback_collection", feedback_collection.run)
    
    if config.enable_persistent_memory:
        workflow.add_node("memory_management", memory_management.run)
    
    # Define the main flow
    workflow.add_edge("query_understanding", "knowledge_retrieval")
    workflow.add_edge("knowledge_retrieval", "response_generation")
    
    # Add conditional edges for error handling
    workflow.add_conditional_edges(
        "query_understanding",
        _has_errors,
        {
            True: "error_handling",
            False: "knowledge_retrieval"
        }
    )
    
    workflow.add_conditional_edges(
        "knowledge_retrieval",
        _has_errors,
        {
            True: "error_handling",
            False: "response_generation"
        }
    )
    
    # Add conditional edges for feedback and memory
    if config.enable_human_feedback:
        workflow.add_edge("response_generation", "feedback_collection")
        
        if config.enable_persistent_memory:
            workflow.add_edge("feedback_collection", "memory_management")
            workflow.add_edge("memory_management", END)
        else:
            workflow.add_edge("feedback_collection", END)
    else:
        if config.enable_persistent_memory:
            workflow.add_edge("response_generation", "memory_management")
            workflow.add_edge("memory_management", END)
        else:
            workflow.add_edge("response_generation", END)
    
    # Error handling can lead back to the main flow or end
    workflow.add_conditional_edges(
        "error_handling",
        _can_recover,
        {
            True: "query_understanding",  # Try again from the beginning
            False: END  # Cannot recover, end the workflow
        }
    )
    
    # --- Define Edges ---
    # The entry point is now the router
    workflow.set_entry_point("router")

    # Add the main conditional edge from the router
    workflow.add_conditional_edges(
        "router",
        route_initial_request,
        {
            "document_processing": "document_processing",
            "query_understanding": "query_understanding",
            # If input is invalid, just end the flow.
            "invalid_input": END
        }
    )

    # The document processing workflow is simple: run the node and end.
    workflow.add_edge("document_processing", END)

    # --- Query Workflow ---
    workflow.add_edge("query_understanding", "knowledge_retrieval")
    workflow.add_edge("knowledge_retrieval", "response_generation")
    
    # Compile the graph
    return workflow.compile()

def _has_errors(state: Dict[str, Any]) -> bool:
    """Check if the state has any errors.
    
    Args:
        state: The current state.
        
    Returns:
        bool: True if there are errors, False otherwise.
    """
    return len(state.get("errors", [])) > 0

def _can_recover(state: Dict[str, Any]) -> bool:
    """Check if we can recover from the errors.
    
    Args:
        state: The current state.
        
    Returns:
        bool: True if we can recover, False otherwise.
    """
    # Check if there are any critical errors
    for error in state.get("errors", []):
        if error.get("severity") == ErrorSeverity.CRITICAL:
            return False
    
    # Check if we've already tried to recover too many times
    recovery_attempts = state.get("metadata", {}).get("recovery_attempts", 0)
    max_recovery_attempts = 3  # Maximum number of recovery attempts
    
    return recovery_attempts < max_recovery_attempts
