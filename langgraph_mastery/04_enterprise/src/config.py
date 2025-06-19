"""
Configuration for the Enterprise Knowledge Assistant.

This module handles configuration loading from environment variables
and provides default settings for the application.
"""

import os
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables from .env file
load_dotenv()

class LoggingConfig(BaseModel):
    """Configuration for logging."""
    level: str = Field(default="INFO")
    format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_path: Optional[str] = Field(default=None)
    json_format: bool = Field(default=False)

class AIModelConfig(BaseModel):
    """Configuration for AI models."""
    provider: str = Field(default="google")
    model_name: str = Field(default="gemini-pro")
    temperature: float = Field(default=0.7)
    max_tokens: Optional[int] = Field(default=None)
    api_key: str = Field(default_factory=lambda: os.getenv("GOOGLE_API_KEY", ""))

class DatabaseConfig(BaseModel):
    """Configuration for database connections."""
    type: str = Field(default="sqlite")
    connection_string: str = Field(
        default_factory=lambda: os.getenv(
            "DATABASE_URL", 
            "sqlite:///enterprise_assistant.db"
        )
    )
    pool_size: int = Field(default=5)
    max_overflow: int = Field(default=10)

class CacheConfig(BaseModel):
    """Configuration for caching."""
    enabled: bool = Field(default=True)
    type: str = Field(default="memory")
    ttl: int = Field(default=3600)  # Time to live in seconds
    max_size: int = Field(default=1000)

class APIConfig(BaseModel):
    """Configuration for API settings."""
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    debug: bool = Field(default=False)
    cors_origins: list = Field(default=["*"])
    rate_limit: int = Field(default=100)  # Requests per minute

class SecurityConfig(BaseModel):
    """Configuration for security settings."""
    secret_key: str = Field(
        default_factory=lambda: os.getenv("SECRET_KEY", "supersecretkey")
    )
    token_expiration: int = Field(default=86400)  # 24 hours in seconds
    password_hash_algorithm: str = Field(default="bcrypt")
    ssl_enabled: bool = Field(default=False)

class AppConfig(BaseModel):
    """Main application configuration."""
    app_name: str = Field(default="Enterprise Knowledge Assistant")
    environment: str = Field(
        default_factory=lambda: os.getenv("APP_ENV", "development")
    )
    debug: bool = Field(
        default_factory=lambda: os.getenv("DEBUG", "False").lower() == "true"
    )
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    ai_model: AIModelConfig = Field(default_factory=AIModelConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    
    # Knowledge base settings
    knowledge_base_path: str = Field(
        default_factory=lambda: os.getenv(
            "KNOWLEDGE_BASE_PATH", 
            "./data/knowledge_base"
        )
    )
    
    # Feature flags
    enable_human_feedback: bool = Field(default=True)
    enable_persistent_memory: bool = Field(default=True)
    enable_multi_agent: bool = Field(default=True)
    enable_monitoring: bool = Field(default=True)

# Create a global config instance
config = AppConfig()

def get_config() -> AppConfig:
    """Get the application configuration.
    
    Returns:
        AppConfig: The application configuration.
    """
    return config

def override_config(overrides: Dict[str, Any]) -> AppConfig:
    """Override configuration values.
    
    Args:
        overrides: Dictionary of configuration overrides.
        
    Returns:
        AppConfig: The updated application configuration.
    """
    global config
    
    # Convert flat dictionary to nested structure if needed
    nested_overrides = {}
    for key, value in overrides.items():
        if "." in key:
            parts = key.split(".")
            current = nested_overrides
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = value
        else:
            nested_overrides[key] = value
    
    # Update the config
    config_dict = config.model_dump()
    _deep_update(config_dict, nested_overrides)
    config = AppConfig.model_validate(config_dict)
    
    return config

def _deep_update(d: Dict[str, Any], u: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively update a dictionary.
    
    Args:
        d: Dictionary to update.
        u: Dictionary with updates.
        
    Returns:
        Dict[str, Any]: Updated dictionary.
    """
    for k, v in u.items():
        if isinstance(v, dict) and k in d and isinstance(d[k], dict):
            _deep_update(d[k], v)
        else:
            d[k] = v
    return d

def is_production() -> bool:
    """Check if the application is running in production.
    
    Returns:
        bool: True if in production, False otherwise.
    """
    return config.environment.lower() == "production"

def is_development() -> bool:
    """Check if the application is running in development.
    
    Returns:
        bool: True if in development, False otherwise.
    """
    return config.environment.lower() == "development"

def is_test() -> bool:
    """Check if the application is running in test.
    
    Returns:
        bool: True if in test, False otherwise.
    """
    return config.environment.lower() == "test"
