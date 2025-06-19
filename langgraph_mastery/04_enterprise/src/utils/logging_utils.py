"""
Logging utilities for the Enterprise Knowledge Assistant.

This module provides utilities for setting up and configuring logging
for the Enterprise Knowledge Assistant.
"""

import logging
import os
import sys
from typing import Dict, Any

from python_json_logger import formatter
from src.models.state import LoggingConfig

def setup_logging(config: LoggingConfig) -> None:
    """Set up logging with the specified configuration.
    
    Args:
        config: The logging configuration.
    """
    # Get the root logger
    root_logger = logging.getLogger()
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Set the log level
    level = getattr(logging, config.level.upper(), logging.INFO)
    root_logger.setLevel(level)
    
    # Create a console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    # Create a formatter based on the config
    if config.json_format:
        log_formatter = formatter.JsonFormatter(config.format)
    else:
        log_formatter = logging.Formatter(config.format)

    console_handler.setFormatter(log_formatter)
    
    # Add the console handler to the root logger
    root_logger.addHandler(console_handler)
    
    # If a file path is specified, add a file handler
    if config.file_path:
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(config.file_path), exist_ok=True)
        
        # Create a file handler
        file_handler = logging.FileHandler(config.file_path)
        file_handler.setLevel(level)
        file_handler.setFormatter(log_formatter)
        
        # Add the file handler to the root logger
        root_logger.addHandler(file_handler)
    
    # Log that logging has been set up
    logging.info(f"Logging set up with level {config.level}")

def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name.
    
    Args:
        name: The name of the logger.
        
    Returns:
        logging.Logger: The logger.
    """
    return logging.getLogger(name)

def log_state_transition(logger: logging.Logger, from_node: str, to_node: str, state: Dict[str, Any]) -> None:
    """Log a state transition.
    
    Args:
        logger: The logger to use.
        from_node: The node transitioning from.
        to_node: The node transitioning to.
        state: The current state.
    """
    logger.debug(f"State transition: {from_node} -> {to_node}")
    
    # Log key state information at debug level
    if logger.isEnabledFor(logging.DEBUG):
        query = state.get("query", "")
        current_node = state.get("current_node", "")
        error_count = len(state.get("errors", []))
        
        logger.debug(f"  Query: {query[:50]}{'...' if len(query) > 50 else ''}")
        logger.debug(f"  Current node: {current_node}")
        logger.debug(f"  Error count: {error_count}")

def log_error(logger: logging.Logger, error: Exception, context: Dict[str, Any] = None) -> None:
    """Log an error with context.
    
    Args:
        logger: The logger to use.
        error: The error to log.
        context: Additional context for the error.
    """
    message = f"Error: {str(error)}"
    
    if context:
        context_str = ", ".join(f"{k}={v}" for k, v in context.items())
        message += f" (Context: {context_str})"
    
    logger.error(message, exc_info=True)
