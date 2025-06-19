"""
Response Generation Node for the Enterprise Knowledge Assistant.

This module orchestrates the response generation process by invoking a
collaborative sub-graph where multiple agents work together to produce a
high-quality response.
"""

import logging
import time
from typing import Dict, Any

from src.models.state import AssistantState, ErrorSeverity
from src.graphs.response_generation_subgraph import create_response_generation_graph, ResponseGenerationState
from langchain_core.messages import AIMessage

logger = logging.getLogger(__name__)

def run(state: Dict[str, Any]) -> Dict[str, Any]:
    """Run the response generation node by invoking the collaborative sub-graph."""
    logger.info("---ORCHESTRATING RESPONSE GENERATION SUB-GRAPH---")
    
    state_obj = AssistantState.model_validate(state)
    query = state_obj.query
    
    try:
        # 1. Create the collaborative response generation sub-graph
        response_graph = create_response_generation_graph()

        # 2. Prepare the initial state for the sub-graph
        retrieved_docs = [doc.content for doc in state_obj.documents]
        
        subgraph_state = ResponseGenerationState(
            query=query,
            retrieved_documents=retrieved_docs,
            messages=[]
        )

        # 3. Invoke the sub-graph to get the final, reviewed response
        final_subgraph_state = response_graph.invoke(subgraph_state)
        
        # The final response is the last message in the sub-graph's state
        final_response_content = final_subgraph_state["messages"][-1].content
        
        logger.info("Collaborative response generation completed successfully.")

        # 4. Update the main graph's state
        state_obj.results["final_response"] = final_response_content
        state_obj.messages.append(AIMessage(content=final_response_content))
        state_obj.current_node = "response_generation"
        state_obj.results["response_generation"] = {
            "response": final_response_content,
            "timestamp": time.time()
        }

    except Exception as e:
        logger.error(f"Error during collaborative response generation: {str(e)}", exc_info=True)
        state_obj.add_error(
            message=f"Failed to generate response via sub-graph: {str(e)}",
            node="response_generation",
            severity=ErrorSeverity.CRITICAL,
            details={"query": query}
        )
        # Provide a fallback response
        fallback_response = "I apologize, but I encountered a critical issue while collaborating on a response. Please contact support."
        state_obj.results["final_response"] = fallback_response

    return state_obj.model_dump()
    
    context_parts = []
    
    for i, doc in enumerate(documents, 1):
        # Format the document metadata
        metadata = doc.metadata
        source_info = f"Source: {metadata.source}"
        if metadata.title:
            source_info += f" | Title: {metadata.title}"
        if metadata.author:
            source_info += f" | Author: {metadata.author}"
        if metadata.updated_at:
            source_info += f" | Last Updated: {metadata.updated_at}"
        
        # Add the document to the context
        context_parts.append(f"--- Document {i} ---\n{source_info}\n\n{doc.content}\n")
    
    return "\n".join(context_parts)
