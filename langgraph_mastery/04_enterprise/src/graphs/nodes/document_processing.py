"""
Node for processing and ingesting documents into the knowledge base.

This module contains the logic for taking a document, processing its content,
and storing it in a vector database for later retrieval.
"""

import logging
from typing import Dict, Any

from src.models.state import AssistantState
from src.utils.vector_store import get_vector_store
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

logger = logging.getLogger(__name__)

def run(state: AssistantState) -> Dict[str, Any]:
    """Run the document processing node.

    Args:
        state (AssistantState): The current state of the graph.

    Returns:
        Dict[str, Any]: The updated state.
    """
    logger.info("Running document processing node")
    document_path = state.get("document_path")

    if not document_path:
        logger.warning("No document path found in state. Skipping document processing.")
        return {}

    try:
        # Load the document
        loader = PyPDFLoader(document_path)
        documents = loader.load()

        # Split the document into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)

        # Get the vector store
        vector_store = get_vector_store()

        # Add the chunks to the vector store
        vector_store.add_documents(chunks)

        logger.info(f"Successfully processed and stored document: {document_path}")
        return {"document_processed": True}

    except Exception as e:
        logger.error(f"Error processing document: {e}", exc_info=True)
        return {"error": {"message": str(e), "severity": "critical"}}
