"""
Error Handling Node for the Enterprise Knowledge Assistant.

This module is responsible for handling errors that occur during processing,
attempting recovery when possible, and providing appropriate feedback.
"""

import logging
import time
from typing import Dict, Any

from langchain_core.messages import AIMessage

from src.config import get_config
from src.models.state import AssistantState, ErrorSeverity

logger = logging.getLogger(__name__)
config = get_config()

def run(state: Dict[str, Any]) -> Dict[str, Any]:
    """Run the error handling node.
    
    Args:
        state: The current state.
        
    Returns:
        Dict[str, Any]: The updated state.
    """
    logger.info("Running error handling node")
    
    # Convert dict to AssistantState for easier manipulation
    state_obj = AssistantState.model_validate(state)
    
    # Get the errors
    errors = state_obj.errors
    
    if not errors:
        logger.warning("Error handling node called but no errors found")
        return state_obj.model_dump()
    
    # Log the errors
    for error in errors:
        logger.error(f"Handling error: {error.message} (Node: {error.node}, Severity: {error.severity})")
    
    # Check if we can recover
    can_recover = True
    for error in errors:
        if error.severity == ErrorSeverity.CRITICAL:
            can_recover = False
            break
    
    # Update recovery attempts counter
    recovery_attempts = state_obj.metadata.get("recovery_attempts", 0)
    state_obj.metadata["recovery_attempts"] = recovery_attempts + 1
    
    # Check if we've exceeded max recovery attempts
    max_recovery_attempts = 3
    if recovery_attempts >= max_recovery_attempts:
        can_recover = False
        logger.warning(f"Exceeded maximum recovery attempts ({max_recovery_attempts})")
    
    # Set the current node
    state_obj.current_node = "error_handling"
    
    # Generate user-facing error message
    user_message = _generate_user_error_message(errors, can_recover)
    
    # Add the error message to the messages
    state_obj.messages.append(AIMessage(content=user_message))
    
    # Update the state with the error handling results
    state_obj.results["error_handling"] = {
        "error_count": len(errors),
        "can_recover": can_recover,
        "recovery_attempts": recovery_attempts + 1,
        "user_message": user_message,
        "timestamp": time.time()
    }
    
    # Set the final response if we can't recover
    if not can_recover:
        state_obj.results["final_response"] = user_message
    
    # If we can recover, clear the errors to start fresh
    if can_recover:
        state_obj.errors = []
    
    logger.info(f"Error handling completed, can_recover={can_recover}")
    
    # Return the updated state
    return state_obj.model_dump()

def _generate_user_error_message(errors, can_recover: bool) -> str:
    """Generate a user-facing error message.
    
    Args:
        errors: The errors that occurred.
        can_recover: Whether we can recover from the errors.
        
    Returns:
        str: The user-facing error message.
    """
    if not errors:
        return "An unknown error occurred. Please try again."
    
    # Count errors by severity
    severity_counts = {}
    for error in errors:
        severity = error.severity
        severity_counts[severity] = severity_counts.get(severity, 0) + 1
    
    # Generate a user-friendly message
    if can_recover:
        message = "I encountered an issue while processing your request. Let me try again."
    else:
        message = """I'm sorry, but I've encountered a problem that I can't resolve at the moment.

Here's what happened:
"""
        
        if ErrorSeverity.CRITICAL in severity_counts:
            message += "- A critical system error occurred\n"
        if ErrorSeverity.ERROR in severity_counts:
            message += f"- {severity_counts[ErrorSeverity.ERROR]} error(s) occurred while processing your request\n"
        if ErrorSeverity.WARNING in severity_counts:
            message += f"- {severity_counts[ErrorSeverity.WARNING]} warning(s) were generated\n"
            
        message += """
Please try again later or contact support if the issue persists.
You might want to try rephrasing your question or providing more details."""
    
    return message
