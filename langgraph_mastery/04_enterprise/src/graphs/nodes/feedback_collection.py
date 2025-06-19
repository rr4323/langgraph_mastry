"""
Feedback Collection Node for the Enterprise Knowledge Assistant.

This module is responsible for collecting and processing user feedback
on the assistant's responses, enabling continuous improvement.
"""

import logging
import time
from typing import Dict, Any

from src.config import get_config
from src.models.state import AssistantState, ErrorSeverity, UserFeedback

logger = logging.getLogger(__name__)
config = get_config()

def run(state: Dict[str, Any]) -> Dict[str, Any]:
    """Run the feedback collection node.
    
    Args:
        state: The current state.
        
    Returns:
        Dict[str, Any]: The updated state.
    """
    logger.info("Running feedback collection node")
    
    # Convert dict to AssistantState for easier manipulation
    state_obj = AssistantState.model_validate(state)
    
    try:
        # In a real implementation, this would collect feedback from the user
        # For this example, we'll simulate feedback collection
        
        # Check if we have user feedback in the context
        context = state_obj.context
        feedback_data = context.get("feedback", None)
        
        if feedback_data:
            # Process the feedback
            rating = feedback_data.get("rating", 0)
            comments = feedback_data.get("comments", "")
            
            # Create a UserFeedback object
            feedback = UserFeedback(
                rating=rating,
                comments=comments
            )
            
            # Add the feedback to the state
            state_obj.user_feedback = feedback
            
            # Add to results
            state_obj.results["feedback_collection"] = {
                "rating": rating,
                "has_comments": bool(comments),
                "timestamp": time.time()
            }
            
            logger.info(f"Collected user feedback with rating {rating}")
        else:
            # No feedback available
            logger.info("No user feedback available")
            state_obj.results["feedback_collection"] = {
                "feedback_available": False,
                "timestamp": time.time()
            }
        
        # Set the current node
        state_obj.current_node = "feedback_collection"
        
    except Exception as e:
        logger.error(f"Error in feedback collection: {str(e)}", exc_info=True)
        state_obj.add_error(
            message=f"Failed to collect feedback: {str(e)}",
            node="feedback_collection",
            severity=ErrorSeverity.WARNING,  # Not critical, can continue
            details={}
        )
    
    # Return the updated state
    return state_obj.model_dump()

def _analyze_feedback(feedback: UserFeedback) -> Dict[str, Any]:
    """Analyze user feedback to extract insights.
    
    Args:
        feedback: The user feedback.
        
    Returns:
        Dict[str, Any]: Analysis results.
    """
    # In a real implementation, this would perform sentiment analysis,
    # extract key themes, and identify improvement opportunities
    
    analysis = {
        "sentiment": "positive" if feedback.rating >= 4 else "neutral" if feedback.rating >= 3 else "negative",
        "key_themes": [],
        "improvement_areas": []
    }
    
    # Simple keyword extraction for themes and improvement areas
    if feedback.comments:
        comments_lower = feedback.comments.lower()
        
        # Check for positive themes
        if any(word in comments_lower for word in ["helpful", "useful", "good", "great", "excellent"]):
            analysis["key_themes"].append("helpfulness")
        
        if any(word in comments_lower for word in ["clear", "understandable", "concise"]):
            analysis["key_themes"].append("clarity")
        
        # Check for improvement areas
        if any(word in comments_lower for word in ["slow", "faster", "quicker"]):
            analysis["improvement_areas"].append("response_time")
        
        if any(word in comments_lower for word in ["confusing", "unclear", "complicated"]):
            analysis["improvement_areas"].append("clarity")
        
        if any(word in comments_lower for word in ["incorrect", "wrong", "error", "mistake"]):
            analysis["improvement_areas"].append("accuracy")
    
    return analysis
