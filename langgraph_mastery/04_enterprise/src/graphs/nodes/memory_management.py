"""
Memory Management Node for the Enterprise Knowledge Assistant.

This module is responsible for managing the assistant's memory,
storing conversation history, user preferences, and learning from interactions.
"""

import logging
import time
from typing import Dict, Any, List

from sqlalchemy import create_engine, Column, Integer, Text, Float, JSON
from sqlalchemy.orm import sessionmaker, declarative_base

from src.config import get_config
from src.models.state import AssistantState, ErrorSeverity

logger = logging.getLogger(__name__)
config = get_config()

# --- Database Setup ---
Base = declarative_base()

class ConversationHistory(Base):
    __tablename__ = 'conversation_history'
    id = Column(Integer, primary_key=True)
    timestamp = Column(Float)
    query = Column(Text)
    response = Column(Text)
    feedback = Column(JSON)
    retrieved_document_count = Column(Integer)
    source_types = Column(JSON)
    errors = Column(JSON)
    metadata = Column(JSON)

# Create engine and session
try:
    engine = create_engine(config.database.connection_string)
    Base.metadata.create_all(engine)
    DBSession = sessionmaker(bind=engine)
except Exception as e:
    logger.error(f"Failed to initialize database: {e}", exc_info=True)
    # Fallback to a non-functional session maker if DB fails
    DBSession = lambda: None

def run(state: Dict[str, Any]) -> Dict[str, Any]:
    """Run the memory management node."""
    if not config.enable_persistent_memory:
        logger.debug("Persistent memory is disabled. Skipping memory management.")
        return state

    logger.info("---RUNNING MEMORY MANAGEMENT NODE---")
    state_obj = AssistantState.model_validate(state)

    try:
        memory_entry = _create_memory_entry(state_obj)
        success = _store_memory(memory_entry)
        state_obj.results["memory_management"] = {
            "memory_stored": success,
            "timestamp": time.time()
        }
        state_obj.current_node = "memory_management"
        logger.info("Memory management completed successfully")
    except Exception as e:
        logger.error(f"Error in memory management: {str(e)}", exc_info=True)
        state_obj.add_error(
            message=f"Failed to manage memory: {str(e)}",
            node="memory_management",
            severity=ErrorSeverity.WARNING,
            details={}
        )
    return state_obj.model_dump()

def _create_memory_entry(state: AssistantState) -> Dict[str, Any]:
    """Create a memory entry from the state."""
    query = state.query
    response = state.results.get("final_response", "")
    feedback = None
    if state.user_feedback:
        feedback = {
            "rating": state.user_feedback.rating,
            "comments": state.user_feedback.comments
        }

    return {
        "timestamp": time.time(),
        "query": query,
        "response": response,
        "feedback": feedback,
        "retrieved_document_count": len(state.documents),
        "source_types": state.retrieved_information.source_types if state.retrieved_information else [],
        "errors": [error.model_dump() for error in state.errors],
        "metadata": state.metadata
    }

def _store_memory(memory_entry: Dict[str, Any]) -> bool:
    """Store a memory entry in the database."""
    session = DBSession()
    if not session:
        logger.error("Database session not available. Cannot store memory.")
        return False
    try:
        db_entry = ConversationHistory(**memory_entry)
        session.add(db_entry)
        session.commit()
        logger.info(f"Successfully stored conversation with id {db_entry.id} to the database.")
        return True
    except Exception as e:
        logger.error(f"Failed to store memory entry in database: {e}", exc_info=True)
        session.rollback()
        return False
    finally:
        session.close()

def retrieve_memory(session_id: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Retrieve recent conversation history."""
    session = DBSession()
    if not session:
        logger.error("Database session not available. Cannot retrieve memory.")
        return []
    try:
        # This is a simplified retrieval. A real app might filter by user_id from metadata.
        history = session.query(ConversationHistory).order_by(ConversationHistory.timestamp.desc()).limit(limit).all()
        # Convert from ORM object to dict, being careful of lazy-loading attributes
        return [
            {c.name: getattr(item, c.name) for c in item.__table__.columns}
            for item in history
        ]
    except Exception as e:
        logger.error(f"Failed to retrieve memory from database: {e}", exc_info=True)
        return []
    finally:
        session.close()
