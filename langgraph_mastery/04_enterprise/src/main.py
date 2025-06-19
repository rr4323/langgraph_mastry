"""
Enterprise Knowledge Assistant - Main Entry Point

This module serves as the main entry point for the Enterprise Knowledge Assistant,
initializing the application and providing the core functionality.
"""

import os
import sys
import logging
import argparse
from typing import Dict, Any, List

# Add the parent directory to the path so we can import from the root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import get_config, is_development
from src.utils.logging_utils import setup_logging
from src.models.state import AssistantState
from src.graphs.main_graph import create_main_graph
from src.agents.agent_factory import create_agent

# Setup logging
config = get_config()
setup_logging(config.logging)
logger = logging.getLogger(__name__)

def initialize_app():
    """Initialize the application components.
    
    Returns:
        tuple: The initialized graph and other components.
    """
    logger.info(f"Initializing {config.app_name} in {config.environment} mode")
    
    # Create the main workflow graph
    logger.debug("Creating main workflow graph")
    graph = create_main_graph()
    
    # Additional initialization can be done here
    
    logger.info("Application initialized successfully")
    return graph

def process_query(graph, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """Process a user query through the graph.
    
    Args:
        graph: The workflow graph.
        query: The user query.
        context: Additional context for the query.
        
    Returns:
        Dict[str, Any]: The result of processing the query.
    """
    logger.info(f"Processing query: {query}")
    
    # Initialize the state
    state = AssistantState(
        query=query,
        context=context or {},
        messages=[],
        current_node="",
        results={},
        errors=[],
        metadata={},
    ).model_dump()
    
    # Process the query through the graph
    try:
        result = graph.invoke(state)
        logger.info("Query processed successfully")
        return result
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        return {
            "error": str(e),
            "query": query,
            "success": False
        }

def process_document(graph, document_path: str) -> Dict[str, Any]:
    """Process a document through the graph.

    Args:
        graph: The workflow graph.
        document_path: The path to the document to process.

    Returns:
        Dict[str, Any]: The result of processing the document.
    """
    logger.info(f"Processing document: {document_path}")

    if not os.path.exists(document_path):
        logger.error(f"Document not found at path: {document_path}")
        return {
            "error": f"Document not found: {document_path}",
            "success": False
        }

    state = AssistantState(
        document_path=document_path,
    ).model_dump()

    try:
        result = graph.invoke(state)
        logger.info("Document processed successfully")
        return result
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}", exc_info=True)
        return {
            "error": str(e),
            "document_path": document_path,
            "success": False
        }

def interactive_session(graph):
    """Run an interactive session with the assistant.
    
    Args:
        graph: The workflow graph.
    """
    print(f"\n{'=' * 50}")
    print(f"Welcome to the {config.app_name}!")
    print(f"{'=' * 50}")
    print("\nType 'exit' or 'quit' to end the session.")
    
    context = {}
    
    while True:
        # Get user input
        query = input("\nYou: ")
        
        # Check for exit command
        if query.lower() in ["exit", "quit"]:
            print("\nThank you for using the Enterprise Knowledge Assistant. Goodbye!")
            break
        
        # Process the query
        result = process_query(graph, query, context)
        
        # Update context with conversation history
        if "messages" in result:
            context["conversation_history"] = result["messages"]
        
        # Display the result
        if "error" in result:
            print(f"\nAssistant: I encountered an error: {result['error']}")
        elif "response" in result:
            print(f"\nAssistant: {result['response']}")
        elif "results" in result and "final_response" in result["results"]:
            print(f"\nAssistant: {result['results']['final_response']}")
        else:
            print("\nAssistant: I processed your request, but I don't have a specific response to show.")

def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(description="Enterprise Knowledge Assistant")
    parser.add_argument("--query", type=str, help="A query to process.")
    parser.add_argument("--document", type=str, help="Path to a document to process.")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode.")

    args = parser.parse_args()

    graph = initialize_app()

    if args.query:
        result = process_query(graph, args.query)
        print(result)
    elif args.document:
        result = process_document(graph, args.document)
        print(result)
    elif args.interactive:
        interactive_session(graph)
    else:
        print("No action specified. Use --query, --document, or --interactive.")
        print("Defaulting to interactive session.")
        interactive_session(graph)
    
    logger.info("Application terminated")

if __name__ == "__main__":
    main()
