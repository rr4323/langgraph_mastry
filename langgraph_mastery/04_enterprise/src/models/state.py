"""
State models for the Enterprise Knowledge Assistant.

This module defines the state models used throughout the application,
providing type safety and structure for the workflow states.
"""

from typing import Dict, List, Any, Optional, Union
from enum import Enum
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

class ErrorSeverity(str, Enum):
    """Severity levels for errors."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class ErrorRecord(BaseModel):
    """Record of an error that occurred during processing."""
    message: str
    node: str = Field(default="unknown")
    timestamp: float = Field(default_factory=lambda: __import__("time").time())
    severity: ErrorSeverity = Field(default=ErrorSeverity.ERROR)
    details: Dict[str, Any] = Field(default_factory=dict)
    
class DocumentMetadata(BaseModel):
    """Metadata for a document."""
    source: str
    title: Optional[str] = None
    author: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    properties: Dict[str, Any] = Field(default_factory=dict)

class Document(BaseModel):
    """A document with content and metadata."""
    content: str
    metadata: DocumentMetadata

class RetrievedInformation(BaseModel):
    """Information retrieved from the knowledge base."""
    documents: List[Document] = Field(default_factory=list)
    source_types: List[str] = Field(default_factory=list)
    query: str
    timestamp: float = Field(default_factory=lambda: __import__("time").time())

class UserFeedback(BaseModel):
    """Feedback provided by a user."""
    rating: int = Field(ge=1, le=5)
    comments: Optional[str] = None
    timestamp: float = Field(default_factory=lambda: __import__("time").time())
    
class AssistantState(BaseModel):
    """The main state for the Enterprise Knowledge Assistant."""
    # Input
    # Input
    query: Optional[str] = None
    document_path: Optional[str] = None
    context: Dict[str, Any] = Field(default_factory=dict)
    
    # Processing state
    messages: List[Union[HumanMessage, AIMessage, SystemMessage]] = Field(default_factory=list)
    current_node: str = Field(default="")
    
    # Results
    results: Dict[str, Any] = Field(default_factory=dict)
    document_processed: bool = False
    
    # Error handling
    errors: List[ErrorRecord] = Field(default_factory=list)
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Optional fields for specific nodes
    retrieved_information: Optional[RetrievedInformation] = None
    documents: List[Document] = Field(default_factory=list)
    user_feedback: Optional[UserFeedback] = None
    
    # Control flow
    next: Optional[str] = None
    
    def add_error(self, message: str, node: str = "unknown", 
                 severity: ErrorSeverity = ErrorSeverity.ERROR,
                 details: Dict[str, Any] = None) -> None:
        """Add an error to the state.
        
        Args:
            message: The error message.
            node: The node where the error occurred.
            severity: The severity of the error.
            details: Additional details about the error.
        """
        self.errors.append(
            ErrorRecord(
                message=message,
                node=node,
                severity=severity,
                details=details or {}
            )
        )
    
    def has_critical_errors(self) -> bool:
        """Check if there are any critical errors.
        
        Returns:
            bool: True if there are critical errors, False otherwise.
        """
        return any(error.severity == ErrorSeverity.CRITICAL for error in self.errors)
    
    def get_last_message(self) -> Optional[Union[HumanMessage, AIMessage, SystemMessage]]:
        """Get the last message in the conversation.
        
        Returns:
            Optional[Union[HumanMessage, AIMessage, SystemMessage]]: The last message, or None if there are no messages.
        """
        return self.messages[-1] if self.messages else None
