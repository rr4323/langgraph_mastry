"""
Search Tool for the Enterprise Knowledge Assistant.

This module provides a search tool for retrieving information from
various knowledge sources, demonstrating how to implement tools in LangGraph.
"""

import logging
import time
import random
from typing import Dict, Any, List, Optional

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from src.config import get_config
from src.models.state import Document, DocumentMetadata

logger = logging.getLogger(__name__)
config = get_config()

class SearchQuery(BaseModel):
    """Search query parameters."""
    query: str = Field(description="The search query")
    filters: Optional[Dict[str, Any]] = Field(
        default=None, 
        description="Optional filters to apply to the search"
    )
    max_results: int = Field(
        default=5, 
        description="Maximum number of results to return"
    )

class SearchResult(BaseModel):
    """Search result."""
    documents: List[Document]
    total_found: int
    query_time_ms: float

class KnowledgeSearchTool(BaseTool):
    """Tool for searching the knowledge base."""
    name = "knowledge_search"
    description = """
    Search for information in the company knowledge base.
    Use this tool when you need to find specific information about company policies,
    procedures, systems, or other internal knowledge.
    """
    args_schema = SearchQuery
    
    def _run(self, query: str, filters: Optional[Dict[str, Any]] = None, 
             max_results: int = 5) -> Dict[str, Any]:
        """Run the search tool.
        
        Args:
            query: The search query.
            filters: Optional filters to apply to the search.
            max_results: Maximum number of results to return.
            
        Returns:
            Dict[str, Any]: The search results.
        """
        logger.info(f"Searching knowledge base for: {query}")
        start_time = time.time()
        
        # In a real implementation, this would connect to a vector database,
        # document store, or other knowledge sources
        documents = self._simulate_search(query, filters, max_results)
        
        # Calculate query time
        query_time_ms = (time.time() - start_time) * 1000
        
        # Create the result
        result = SearchResult(
            documents=documents,
            total_found=len(documents),
            query_time_ms=query_time_ms
        )
        
        logger.info(f"Search completed in {query_time_ms:.2f}ms, found {len(documents)} documents")
        
        return result.model_dump()
    
    def _simulate_search(self, query: str, filters: Optional[Dict[str, Any]], 
                         max_results: int) -> List[Document]:
        """Simulate a search in the knowledge base.
        
        In a real implementation, this would connect to actual knowledge sources.
        
        Args:
            query: The search query.
            filters: Optional filters to apply to the search.
            max_results: Maximum number of results to return.
            
        Returns:
            List[Document]: The retrieved documents.
        """
        # Simulate some delay for realism
        time.sleep(random.uniform(0.2, 0.8))
        
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
        
        # Simulate HR documentation
        if any(term in query.lower() for term in ["hr", "employee", "benefits", "vacation", "pto", "leave"]):
            documents.append(Document(
                content="""
# Employee Benefits Overview

## Health Insurance
The company provides comprehensive health insurance through BlueCross BlueShield.

## Retirement Benefits
401(k) plan with company matching up to 5% of salary.

## Paid Time Off
- 15 days of vacation per year
- 10 days of sick leave
- 10 paid holidays
- 3 personal days

## Additional Benefits
- Flexible work arrangements
- Professional development budget
- Wellness program
- Employee assistance program
                """,
                metadata=DocumentMetadata(
                    source="hr_documentation",
                    title="Employee Benefits Guide",
                    author="HR Department",
                    created_at="2024-02-01",
                    updated_at="2024-05-15",
                    tags=["hr", "benefits", "employees"]
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
        
        # Apply filters if provided
        if filters:
            filtered_documents = []
            for doc in documents:
                include = True
                for key, value in filters.items():
                    if key == "source" and doc.metadata.source != value:
                        include = False
                        break
                    elif key == "tags" and not any(tag in doc.metadata.tags for tag in value):
                        include = False
                        break
                    elif key == "author" and doc.metadata.author != value:
                        include = False
                        break
                
                if include:
                    filtered_documents.append(doc)
            
            documents = filtered_documents
        
        # Limit the number of results
        return documents[:max_results]

def get_search_tool() -> KnowledgeSearchTool:
    """Get the knowledge search tool.
    
    Returns:
        KnowledgeSearchTool: The knowledge search tool.
    """
    return KnowledgeSearchTool()
