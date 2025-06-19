"""
Agent Factory for the Enterprise Knowledge Assistant.

This module provides factory functions for creating different types of agents
used in the Enterprise Knowledge Assistant.
"""

import logging
from typing import Dict, Any, Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage

from src.config import get_config

logger = logging.getLogger(__name__)
config = get_config()

def create_agent(agent_type: str, **kwargs) -> Any:
    """Create an agent of the specified type.
    
    Args:
        agent_type: The type of agent to create.
        **kwargs: Additional arguments for agent creation.
        
    Returns:
        Any: The created agent.
    """
    logger.info(f"Creating agent of type: {agent_type}")
    
    if agent_type == "query_understanding":
        return create_query_understanding_agent(**kwargs)
    elif agent_type == "knowledge_retrieval":
        return create_knowledge_retrieval_agent(**kwargs)
    elif agent_type == "response_generation":
        return create_response_generation_agent(**kwargs)
    elif agent_type == "drafter":
        return create_drafter_agent(**kwargs)
    elif agent_type == "reviewer":
        return create_reviewer_agent(**kwargs)
    else:
        logger.warning(f"Unknown agent type: {agent_type}")
        return create_default_agent(**kwargs)

def create_query_understanding_agent(**kwargs) -> ChatGoogleGenerativeAI:
    """Create a query understanding agent.
    
    Args:
        **kwargs: Additional arguments for agent creation.
        
    Returns:
        ChatGoogleGenerativeAI: The created agent.
    """
    # Create a chat model with appropriate settings
    model = ChatGoogleGenerativeAI(
        model=config.ai_model.model_name,
        temperature=0.3,  # Lower temperature for more deterministic results
        convert_system_message_to_human=True
    )
    
    return model

def create_knowledge_retrieval_agent(**kwargs) -> ChatGoogleGenerativeAI:
    """Create a knowledge retrieval agent.
    
    Args:
        **kwargs: Additional arguments for agent creation.
        
    Returns:
        ChatGoogleGenerativeAI: The created agent.
    """
    # Create a chat model with appropriate settings
    model = ChatGoogleGenerativeAI(
        model=config.ai_model.model_name,
        temperature=0.2,  # Lower temperature for more deterministic results
        convert_system_message_to_human=True
    )
    
    return model

def create_response_generation_agent(**kwargs) -> ChatGoogleGenerativeAI:
    """Create a response generation agent.
    
    Args:
        **kwargs: Additional arguments for agent creation.
        
    Returns:
        ChatGoogleGenerativeAI: The created agent.
    """
    # Create a chat model with appropriate settings
    model = ChatGoogleGenerativeAI(
        model=config.ai_model.model_name,
        temperature=config.ai_model.temperature,
        convert_system_message_to_human=True
    )
    
    return model

def create_drafter_agent(**kwargs) -> Any:
    """Create a drafting agent for response generation."""
    logger.debug("Creating drafter agent")
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert writer. Your task is to draft a clear, concise, and comprehensive response to the user's query based on the provided context. "
                "Focus on accuracy and readability. Do not add any preamble or sign-off, just provide the draft response."
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    model = ChatGoogleGenerativeAI(
        model=config.ai_model.model_name,
        temperature=0.7, # Higher temperature for more creative drafting
        convert_system_message_to_human=True
    )
    return prompt | model

def create_reviewer_agent(**kwargs) -> Any:
    """Create a reviewing agent for response generation."""
    logger.debug("Creating reviewer agent")
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a meticulous editor and fact-checker. Your task is to review the draft response. "
                "If the draft is accurate, complete, and well-written, respond with only the word 'OK'. "
                "If it needs improvement, provide specific, constructive feedback on what to change. Do not rewrite the draft yourself."
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    model = ChatGoogleGenerativeAI(
        model=config.ai_model.model_name,
        temperature=0.1, # Lower temperature for precise, factual reviews
        convert_system_message_to_human=True
    )
    return prompt | model

def create_default_agent(**kwargs) -> ChatGoogleGenerativeAI:
    """Create a default agent.
    
    Args:
        **kwargs: Additional arguments for agent creation.
        
    Returns:
        ChatGoogleGenerativeAI: The created agent.
    """
    # Create a chat model with default settings
    model = ChatGoogleGenerativeAI(
        model=config.ai_model.model_name,
        temperature=config.ai_model.temperature,
        convert_system_message_to_human=True
    )
    
    return model

def get_system_prompt(agent_type: str) -> str:
    """Get the system prompt for the specified agent type.
    
    Args:
        agent_type: The type of agent.
        
    Returns:
        str: The system prompt.
    """
    if agent_type == "query_understanding":
        return """You are a query understanding assistant for an Enterprise Knowledge Assistant.
Your job is to analyze user queries and extract key information to help with knowledge retrieval.

For each query, please:
1. Identify the main topic or subject
2. Extract key entities (people, organizations, concepts, etc.)
3. Determine the query type (factual question, how-to, explanation, etc.)
4. Identify any constraints or filters
5. Note any temporal aspects (time periods, deadlines, etc.)

Format your response as a structured analysis that can be used for knowledge retrieval."""
    
    elif agent_type == "knowledge_retrieval":
        return """You are a knowledge retrieval assistant for an Enterprise Knowledge Assistant.
Your job is to formulate effective search queries based on the user's question and query analysis.

For each query, please:
1. Identify the most important keywords for retrieval
2. Generate alternative phrasings for key concepts
3. Specify any filters that should be applied (date ranges, document types, etc.)
4. Indicate the relative importance of different query components

Format your response as a structured search strategy that can be used for knowledge retrieval."""
    
    elif agent_type == "response_generation":
        return """You are an Enterprise Knowledge Assistant for a company.
Your task is to provide helpful, accurate responses based on the company knowledge base.

Use ONLY the information provided to answer the user's question.
If the information provided doesn't fully answer the question, acknowledge this limitation
and suggest what additional information might be needed.

Always maintain a professional, helpful tone and format your responses clearly using markdown.
Include relevant citations or references to company policies when applicable."""
    
    else:
        return """You are an AI assistant for an enterprise company.
Your task is to provide helpful, accurate responses to user queries.
Always maintain a professional, helpful tone and format your responses clearly."""
