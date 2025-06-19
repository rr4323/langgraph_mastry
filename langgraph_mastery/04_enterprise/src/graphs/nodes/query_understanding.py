"""
Query Understanding Node for the Enterprise Knowledge Assistant.

This module is responsible for analyzing and understanding user queries,
extracting key information, and preparing the query for knowledge retrieval.
"""

import logging
from typing import Dict, Any
import time

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from src.config import get_config
from src.models.state import AssistantState, ErrorSeverity

logger = logging.getLogger(__name__)
config = get_config()

def run(state: Dict[str, Any]) -> Dict[str, Any]:
    """Run the query understanding node.
    
    Args:
        state: The current state.
        
    Returns:
        Dict[str, Any]: The updated state.
    """
    logger.info("Running query understanding node")
    
    # Convert dict to AssistantState for easier manipulation
    state_obj = AssistantState.model_validate(state)
    query = state_obj.query
    
    try:
        # Create a chat model
        model = ChatGoogleGenerativeAI(
            model=config.ai_model.model_name,
            temperature=0.3,  # Lower temperature for more deterministic results
            convert_system_message_to_human=True
        )
        
        # Create messages for the model
        messages = [
            SystemMessage(content="""You are a query understanding assistant for an Enterprise Knowledge Assistant.
Your job is to analyze user queries and extract key information to help with knowledge retrieval.

For each query, please:
1. Identify the main topic or subject
2. Extract key entities (people, organizations, concepts, etc.)
3. Determine the query type (factual question, how-to, explanation, etc.)
4. Identify any constraints or filters
5. Note any temporal aspects (time periods, deadlines, etc.)

Format your response as a structured analysis that can be used for knowledge retrieval."""),
            HumanMessage(content=query)
        ]
        
        # Get the analysis
        logger.debug(f"Sending query to model: {query}")
        response = model.invoke(messages)
        
        # Add the messages to the state
        state_obj.messages.append(HumanMessage(content=query))
        
        # Extract and structure the analysis
        analysis = response.content
        
        # Update the state with the analysis
        state_obj.results["query_understanding"] = {
            "analysis": analysis,
            "timestamp": time.time()
        }
        
        # Set the current node
        state_obj.current_node = "query_understanding"
        
        logger.info("Query understanding completed successfully")
        
    except Exception as e:
        logger.error(f"Error in query understanding: {str(e)}", exc_info=True)
        state_obj.add_error(
            message=f"Failed to understand query: {str(e)}",
            node="query_understanding",
            severity=ErrorSeverity.ERROR,
            details={"query": query}
        )
    
    # Return the updated state
    return state_obj.model_dump()
