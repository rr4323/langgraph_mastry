"""
Knowledge Retrieval Node for the Enterprise Knowledge Assistant.

This module is responsible for retrieving relevant information from
the knowledge base based on the user query and query understanding.
"""

import logging
import time
import os
from typing import Dict, Any, List

from src.config import get_config
from src.models.state import AssistantState, ErrorSeverity, Document, DocumentMetadata, RetrievedInformation

logger = logging.getLogger(__name__)
config = get_config()

def run(state: Dict[str, Any]) -> Dict[str, Any]:
    """Run the knowledge retrieval node.
    
    Args:
        state: The current state.
        
    Returns:
        Dict[str, Any]: The updated state.
    """
    logger.info("Running knowledge retrieval node")
    
    # Convert dict to AssistantState for easier manipulation
    state_obj = AssistantState.model_validate(state)
    query = state_obj.query
    
    try:
        # Get the query understanding results
        query_understanding = state_obj.results.get("query_understanding", {})
        analysis = query_understanding.get("analysis", "")
        
        # Simulate knowledge retrieval
        # In a real implementation, this would connect to a vector database,
        # document store, or other knowledge sources
        documents = _simulate_knowledge_retrieval(query, analysis)
        
        # Create a RetrievedInformation object
        retrieved_info = RetrievedInformation(
            documents=documents,
            source_types=["company_policy", "technical_documentation", "faq"],
            query=query
        )
        
        # Update the state
        state_obj.retrieved_information = retrieved_info
        state_obj.documents = documents
        
        # Add to results
        state_obj.results["knowledge_retrieval"] = {
            "document_count": len(documents),
            "source_types": retrieved_info.source_types,
            "timestamp": time.time()
        }
        
        # Set the current node
        state_obj.current_node = "knowledge_retrieval"
        
        logger.info(f"Knowledge retrieval completed successfully, found {len(documents)} documents")
        
    except Exception as e:
        logger.error(f"Error in knowledge retrieval: {str(e)}", exc_info=True)
        state_obj.add_error(
            message=f"Failed to retrieve knowledge: {str(e)}",
            node="knowledge_retrieval",
            severity=ErrorSeverity.ERROR,
            details={"query": query}
        )
    
    # Return the updated state
    return state_obj.model_dump()

def _simulate_knowledge_retrieval(query: str, analysis: str) -> List[Document]:
    """Simulate knowledge retrieval from various sources.
    
    In a real implementation, this would connect to actual knowledge sources.
    
    Args:
        query: The user query.
        analysis: The query analysis from the understanding node.
        
    Returns:
        List[Document]: The retrieved documents.
    """
    # Create some simulated documents based on the query
    documents = []
    
    # Simulate company policy document
    if any(term in query.lower() for term in ["policy", "procedure", "rule", "guideline"]):
        documents.append(Document(
            content="""
# Company Policy on Data Security

## Overview
All employees must adhere to the company's data security policies to protect sensitive information.

## Key Points
1. All sensitive data must be encrypted at rest and in transit
2. Access to customer data requires manager approval
3. Two-factor authentication is mandatory for all systems
4. Regular security training is required for all employees
5. Security incidents must be reported within 24 hours

## Compliance
Failure to comply with these policies may result in disciplinary action.
            """,
            metadata=DocumentMetadata(
                source="company_policy",
                title="Data Security Policy",
                author="Security Team",
                created_at="2024-01-15",
                updated_at="2024-05-20",
                tags=["security", "compliance", "data protection"]
            )
        ))
    
    # Simulate technical documentation
    if any(term in query.lower() for term in ["system", "technical", "software", "hardware", "application"]):
        documents.append(Document(
            content="""
# Enterprise Resource Planning (ERP) System Documentation

## System Architecture
The ERP system consists of multiple modules including Finance, HR, Inventory, and CRM.

## Database Structure
- Primary database: PostgreSQL 14
- Reporting database: Amazon Redshift
- Data warehouse: Snowflake

## Integration Points
- SAP for financial data
- Salesforce for customer information
- Workday for HR data
- Custom APIs for third-party integrations

## Deployment
The system is deployed on AWS using Kubernetes for orchestration.
            """,
            metadata=DocumentMetadata(
                source="technical_documentation",
                title="ERP System Architecture",
                author="IT Department",
                created_at="2023-11-10",
                updated_at="2024-04-05",
                tags=["erp", "architecture", "technical", "system"]
            )
        ))
    
    # Simulate FAQ
    documents.append(Document(
        content=f"""
# Frequently Asked Questions

## {query}
This is a simulated answer to your query about "{query}". In a real implementation, 
this would be retrieved from an actual knowledge base with relevant information.

The answer would include comprehensive information addressing your specific question,
including any relevant policies, procedures, or technical details.

## Related Questions
1. How do I request access to system X?
2. What is the procedure for reporting security incidents?
3. Who should I contact for technical support?
        """,
        metadata=DocumentMetadata(
            source="faq",
            title="Frequently Asked Questions",
            author="Knowledge Base Team",
            created_at="2024-02-20",
            updated_at="2024-05-25",
            tags=["faq", "help", "support"]
        )
    ))
    
    return documents
