import pytest
import os
from unittest.mock import patch, MagicMock

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

from src.graphs.main_graph import create_main_graph
from src.models.state import AssistantState, Document

@pytest.fixture(scope="module")
def app_graph():
    """Fixture to create the main application graph."""
    return create_main_graph()

@pytest.fixture(scope="module")
def test_pdf_path():
    """Fixture to create a dummy PDF for testing and return its path."""
    pdf_path = "/tmp/test_document.pdf"
    c = canvas.Canvas(pdf_path, pagesize=letter)
    c.drawString(100, 750, "This is a test document for the Enterprise Knowledge Assistant.")
    c.showPage()
    c.save()
    yield pdf_path
    os.remove(pdf_path)

def test_document_processing_workflow(app_graph, test_pdf_path):
    """Test the end-to-end document processing workflow."""
    # Initial state for document processing
    initial_state = AssistantState(
        document_path=test_pdf_path,
        query=None,
        messages=[],
        documents=[],
        retrieved_information=None,
        user_feedback=None,
        errors=[],
        results={},
        metadata={},
        current_node="",
        document_processed=False
    )

    # Mock the vector store used in document_processing node
    with patch('src.graphs.nodes.document_processing.FAISS') as mock_faiss:
        mock_vector_store = MagicMock()
        mock_faiss.from_documents.return_value = mock_vector_store

        # Run the graph
        final_state = app_graph.invoke(initial_state.model_dump())

    # Assertions
    assert final_state["document_processed"] is True
    assert "document_processing" in final_state["results"]
    assert final_state["results"]["document_processing"]["status"] == "Success"
    mock_faiss.from_documents.assert_called_once()


def test_query_workflow(app_graph):
    """Test the end-to-end query processing workflow with multi-agent collaboration."""
    # Initial state for a query
    initial_state = AssistantState(
        query="What are the company's travel expense policies?",
        messages=[],
        documents=[
            Document(content="Travel expenses must be pre-approved by a manager.", source="policy_doc_1.pdf"),
            Document(content="Flights should be booked in economy class.", source="policy_doc_2.pdf")
        ],
        retrieved_information=None, # This would be populated by knowledge_retrieval
        user_feedback=None,
        errors=[],
        results={},
        metadata={"user_id": "test_user"},
        current_node="",
        document_path=None,
        document_processed=False
    )

    # Mock the database session in memory_management to avoid actual DB writes
    with patch('src.graphs.nodes.memory_management.DBSession') as mock_db_session:
        # Run the graph from the response generation step
        # In a full E2E test, we'd run the whole graph, but here we focus on the new parts.
        final_state = app_graph.invoke(initial_state.model_dump(), config={"recursion_limit": 10})

    # Assertions
    assert "final_response" in final_state["results"]
    assert len(final_state["results"]["final_response"]) > 0
    assert "response_generation" in final_state["results"]
    assert "memory_management" in final_state["results"]
    assert final_state["results"]["memory_management"]["memory_stored"] is True
    mock_db_session.assert_called()
