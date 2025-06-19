"""
Tests for the main workflow graph of the Enterprise Knowledge Assistant.

This module contains tests for the main workflow graph, ensuring that
it correctly processes queries and handles errors.
"""

import os
import sys
import pytest
from unittest.mock import patch, MagicMock

# Add the parent directory to the path so we can import from the root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.graphs.main_graph import create_main_graph
from src.models.state import AssistantState, ErrorSeverity

@pytest.fixture
def mock_google_genai():
    """Mock the Google Generative AI client."""
    with patch("langchain_google_genai.ChatGoogleGenerativeAI") as mock:
        # Configure the mock to return a specific response
        mock_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "This is a mock response from the AI model."
        mock_instance.invoke.return_value = mock_response
        mock.return_value = mock_instance
        yield mock

@pytest.fixture
def workflow_graph():
    """Create a workflow graph for testing."""
    return create_main_graph()

def test_graph_creation():
    """Test that the graph can be created successfully."""
    graph = create_main_graph()
    assert graph is not None

def test_query_processing(workflow_graph, mock_google_genai):
    """Test that a query can be processed through the graph."""
    # Create an initial state
    initial_state = AssistantState(
        query="What is the company policy on data security?",
        context={},
        messages=[],
        current_node="",
        results={},
        errors=[],
        metadata={},
    ).model_dump()
    
    # Process the query through the graph
    result = workflow_graph.invoke(initial_state)
    
    # Verify the result
    assert "final_response" in result.get("results", {})
    assert result.get("current_node") is not None

def test_error_handling(workflow_graph, mock_google_genai):
    """Test that errors are handled correctly."""
    # Configure the mock to raise an exception
    mock_google_genai.return_value.invoke.side_effect = Exception("Test error")
    
    # Create an initial state
    initial_state = AssistantState(
        query="What is the company policy on data security?",
        context={},
        messages=[],
        current_node="",
        results={},
        errors=[],
        metadata={},
    ).model_dump()
    
    # Process the query through the graph
    result = workflow_graph.invoke(initial_state)
    
    # Verify that errors were handled
    assert len(result.get("errors", [])) > 0
    assert "error_handling" in result.get("results", {})

def test_feedback_processing(workflow_graph, mock_google_genai):
    """Test that feedback can be processed."""
    # Create an initial state with feedback
    initial_state = AssistantState(
        query="What is the company policy on data security?",
        context={"feedback": {"rating": 5, "comments": "Very helpful response!"}},
        messages=[],
        current_node="",
        results={},
        errors=[],
        metadata={},
    ).model_dump()
    
    # Process the query through the graph
    result = workflow_graph.invoke(initial_state)
    
    # Verify that feedback was processed
    assert "feedback_collection" in result.get("results", {})
    
    # If memory management is enabled, verify that memory was stored
    if "memory_management" in result.get("results", {}):
        assert result["results"]["memory_management"]["memory_stored"] is not None

def test_recovery_from_non_critical_error():
    """Test that the system can recover from non-critical errors."""
    # Create a graph
    graph = create_main_graph()
    
    # Create an initial state with a non-critical error
    initial_state = AssistantState(
        query="What is the company policy on data security?",
        context={},
        messages=[],
        current_node="",
        results={},
        errors=[{
            "message": "Non-critical error",
            "node": "query_understanding",
            "severity": ErrorSeverity.WARNING,
            "timestamp": 1622548800.0,
            "details": {}
        }],
        metadata={"recovery_attempts": 0},
    ).model_dump()
    
    # Process the state through the error handling node
    with patch("src.graphs.nodes.error_handling._generate_user_error_message", 
               return_value="Error message"):
        result = graph.invoke(initial_state)
    
    # Verify that recovery was attempted
    assert result["metadata"]["recovery_attempts"] > 0

if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
