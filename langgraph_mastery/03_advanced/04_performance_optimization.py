"""
LangGraph Advanced: Performance Optimization
=========================================

This script demonstrates techniques for optimizing the performance of LangGraph applications
including parallel execution, caching, and efficient state management.
"""

import os
import sys
import time
import asyncio
import hashlib
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Annotated, Dict, List, Literal, TypedDict, Union, Any, Optional, Tuple, Callable, Awaitable
from enum import Enum
from functools import lru_cache
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage, FunctionMessage
from langchain_core.tools import tool
from langchain.agents import AgentType, initialize_agent, AgentExecutor
from langchain.tools import Tool
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import SystemMessage
from langchain.agents.agent_types import AgentType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("performance_optimization.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add the parent directory to the path so we can import from the root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables from .env file
load_dotenv()

class SessionData(BaseModel):
    """Session data for tracking workflow execution."""
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    start_time: float = Field(default_factory=time.time)
    last_active: float = Field(default_factory=time.time)
    interaction_count: int = 0
    
    def update_activity(self):
        """Update the last activity time."""
        self.last_active = time.time()
    
    def increment_interaction(self):
        """Increment the interaction count."""
        self.interaction_count += 1

# Define our state for the optimized workflow
class OptimizedState(BaseModel):
    """State for our optimized workflow."""
    messages: List[Union[HumanMessage, AIMessage, SystemMessage, FunctionMessage]] = Field(default_factory=list)
    results: Dict[str, Any] = Field(default_factory=dict)
    cache_hits: int = 0
    cache_misses: int = 0
    execution_times: Dict[str, float] = Field(default_factory=dict)
    parallel_tasks: List[str] = Field(default_factory=list)
    session_data: SessionData = Field(default_factory=SessionData)
    next: Optional[str] = None
    agent_scratchpad: List[Dict[str, Any]] = Field(default_factory=list)
    current_tool: Optional[str] = None
    
    def model_dump(self, **kwargs):
        """Convert the model to a dictionary, handling non-serializable fields."""
        data = super().model_dump(**kwargs)
        # Convert messages to dictionaries using model_dump() for Pydantic v2
        data['messages'] = [msg.model_dump() if hasattr(msg, 'model_dump') else str(msg) for msg in self.messages]
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OptimizedState':
        """Create an instance from a dictionary."""
        # Create a copy of the data to avoid modifying the input
        data = data.copy()
        
        # Handle message deserialization if needed
        if 'messages' in data and data['messages'] and isinstance(data['messages'][0], dict):
            messages = []
            for msg in data['messages']:
                msg = msg.copy()  # Don't modify the original
                msg_type = msg.pop('type', '').lower()
                
                # Handle different message types
                if msg_type == 'human':
                    from langchain_core.messages import HumanMessage
                    messages.append(HumanMessage(**msg))
                elif msg_type == 'ai':
                    from langchain_core.messages import AIMessage
                    messages.append(AIMessage(**msg))
                elif msg_type == 'system':
                    from langchain_core.messages import SystemMessage
                    messages.append(SystemMessage(**msg))
                elif msg_type == 'function':
                    from langchain_core.messages import FunctionMessage
                    messages.append(FunctionMessage(**msg))
                else:
                    # Fallback for unknown message types
                    messages.append(str(msg))
            
            data['messages'] = messages
        
        # Ensure all required fields have default values
        defaults = {
            'results': {},
            'cache_hits': 0,
            'cache_misses': 0,
            'execution_times': {},
            'parallel_tasks': [],
            'session_data': SessionData(),
            'next': None,
            'agent_scratchpad': [],
            'current_tool': None
        }
        
        # Apply defaults for any missing fields
        for key, default in defaults.items():
            if key not in data:
                data[key] = default
        
        return cls(**data)

# Cache configuration
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)
CACHE_FILE = CACHE_DIR / "function_cache.json"
CACHE = {}
CACHE_MAX_AGE_DAYS = 7  # Cache entries older than this will be invalidated

class CacheEntry(BaseModel):
    """Represents a cache entry with timestamp and data."""
    timestamp: float
    data: Any

class CacheManager:
    """Manages caching operations with persistence."""
    
    def __init__(self):
        self.cache = {}
        self._load_cache()
    
    def _load_cache(self):
        """Load cache from disk if it exists."""
        global CACHE
        try:
            if CACHE_FILE.exists():
                with open(CACHE_FILE, 'r') as f:
                    data = json.load(f)
                    # Convert dict to CacheEntry objects
                    for key, entry in data.items():
                        if isinstance(entry, dict) and 'timestamp' in entry and 'data' in entry:
                            self.cache[key] = CacheEntry(**entry)
                    CACHE = self.cache
                    logger.info(f"Loaded {len(self.cache)} cache entries from disk")
        except Exception as e:
            logger.error(f"Error loading cache: {e}")
    
    def _save_cache(self):
        """Save cache to disk."""
        try:
            with open(CACHE_FILE, 'w') as f:
                # Convert CacheEntry objects to dict
                cache_data = {k: v.dict() for k, v in self.cache.items()}
                json.dump(cache_data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving cache: {e}")
    
    def get(self, key: str) -> Any:
        """Get a value from the cache if it exists and is not expired."""
        if key not in self.cache:
            return None
            
        entry = self.cache[key]
        age_days = (time.time() - entry.timestamp) / (24 * 3600)
        
        if age_days > CACHE_MAX_AGE_DAYS:
            del self.cache[key]
            self._save_cache()
            return None
            
        return entry.data
    
    def set(self, key: str, value: Any):
        """Set a value in the cache."""
        self.cache[key] = CacheEntry(timestamp=time.time(), data=value)
        self._save_cache()
    
    def clear_expired(self):
        """Clear expired cache entries."""
        initial_count = len(self.cache)
        now = time.time()
        expired_keys = [
            k for k, v in self.cache.items()
            if (now - v.timestamp) / (24 * 3600) > CACHE_MAX_AGE_DAYS
        ]
        
        for key in expired_keys:
            del self.cache[key]
            
        if expired_keys:
            self._save_cache()
            logger.info(f"Cleared {len(expired_keys)} expired cache entries")

# Initialize cache manager
cache_manager = CacheManager()

def get_cache_key(func_name: str, args: Tuple, kwargs: Dict) -> str:
    """Generate a cache key for a function call."""
    try:
        # Convert arguments to a string representation
        args_str = json.dumps(args, sort_keys=True, default=str)
        kwargs_str = json.dumps(kwargs, sort_keys=True, default=str)
        
        # Create a hash of the function name and arguments
        key = hashlib.md5(f"{func_name}:{args_str}:{kwargs_str}".encode()).hexdigest()
        return key
    except Exception as e:
        logger.error(f"Error generating cache key: {e}")
        raise

def cached_execution(func_name: str, func, *args, **kwargs):
    """Execute a function with caching.
    
    Args:
        func_name: Name of the function being cached (for logging)
        func: The function to execute
        *args: Positional arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function
        
    Returns:
        A tuple of (result, is_cache_hit)
    """
    try:
        # Separate state from other kwargs
        state = None
        other_kwargs = {}
        for k, v in kwargs.items():
            if isinstance(v, OptimizedState):
                state = v
            else:
                other_kwargs[k] = v
        
        # Generate a cache key without the state
        cache_key = get_cache_key(func_name, args, other_kwargs)
        
        # Check if the result is in the cache
        cached_result = cache_manager.get(cache_key)
        if cached_result is not None:
            logger.info(f"Cache hit for {func_name}")
            return cached_result, True
        
        # Prepare arguments for the function call
        call_kwargs = other_kwargs.copy()
        if state is not None:
            call_kwargs['state'] = state
        
        # Execute the function with all arguments
        start_time = time.time()
        result = func(*args, **call_kwargs)
        execution_time = time.time() - start_time
        
        # Only cache if execution was fast and the result is not None
        if execution_time < 1.0 and result is not None:
            try:
                cache_manager.set(cache_key, result)
            except Exception as e:
                logger.warning(f"Failed to cache result for {func_name}: {str(e)}")
        
        logger.info(f"Cache miss for {func_name}, execution time: {execution_time:.2f}s")
        
        return result, False
    except Exception as e:
        logger.error(f"Error in cached_execution for {func_name}: {e}")
        # If there's an error with caching, fall back to direct execution
        try:
            return func(*args, **kwargs), False
        except Exception as inner_e:
            logger.error(f"Direct execution also failed for {func_name}: {inner_e}")
            raise

# Define tools with caching
import requests
from duckduckgo_search import DDGS

@tool
def search_information(query: str, max_results: int = 5) -> str:
    """
    Search for information related to a query using DuckDuckGo.
    
    Args:
        query: The search query
        max_results: Maximum number of results to return (default: 5)
    
    Returns:
        str: Formatted search results with titles and snippets
    """
    try:
        logger.info(f"Searching for: {query}")
        
        # Use DuckDuckGo search
        ddgs = DDGS()
        results = ddgs.text(query, max_results=max_results)
        
        if not results:
            return f"No results found for '{query}'"
        
        # Format the results
        formatted_results = [f"{i+1}. {result['title']}\n   {result['body']}\n   URL: {result['href']}" 
                           for i, result in enumerate(results)]
        
        return f"Search results for '{query}':\n\n" + "\n\n".join(formatted_results)
    except Exception as e:
        logger.error(f"Error in search_information: {e}")
        return f"Error performing search: {str(e)}"

from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

# Download VADER lexicon if not already downloaded
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()

@tool
def analyze_sentiment(text: str) -> Dict[str, float]:
    """
    Analyze the sentiment of a text using VADER (Valence Aware Dictionary and sEntiment Reasoner).
    
    Args:
        text: The text to analyze
    
    Returns:
        Dict: Sentiment scores with compound, positive, neutral, and negative values
    """
    try:
        logger.info("Analyzing sentiment...")
        
        # Get sentiment scores
        scores = sia.polarity_scores(text)
        
        # Format the scores
        return {
            "compound": round(scores['compound'], 3),
            "positive": round(scores['pos'], 3),
            "neutral": round(scores['neu'], 3),
            "negative": round(scores['neg'], 3)
        }
    except Exception as e:
        logger.error(f"Error in analyze_sentiment: {e}")
        return {"error": f"Failed to analyze sentiment: {str(e)}"}

import spacy
from collections import Counter

# Load the English language model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logger.warning("spaCy English model not found. Installing...")
    import sys
    import subprocess
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

@tool
def extract_keywords(text: str, top_n: int = 5) -> List[Dict[str, Any]]:
    """
    Extract keywords from a text using spaCy's NLP capabilities.
    
    Args:
        text: The text to analyze
        top_n: Number of top keywords to return (default: 5)
    
    Returns:
        List[Dict]: List of dictionaries containing keywords and their relevance scores
    """
    try:
        logger.info("Extracting keywords...")
        
        # Process the text
        doc = nlp(text)
        
        # Filter for nouns and proper nouns that are not stop words
        keywords = []
        for token in doc:
            if (not token.is_stop and 
                not token.is_punct and 
                token.pos_ in ["NOUN", "PROPN"] and 
                len(token.text) > 2):
                keywords.append(token.lemma_.lower())
        
        # Count keyword frequencies
        keyword_freq = Counter(keywords)
        
        # Get the most common keywords
        most_common = keyword_freq.most_common(top_n)
        
        # Calculate relevance scores (normalized frequency)
        total = sum(freq for _, freq in most_common) or 1  # Avoid division by zero
        result = [
            {"keyword": kw, "frequency": freq, "relevance": round(freq/total, 3)}
            for kw, freq in most_common
        ]
        
        return result
    except Exception as e:
        logger.error(f"Error in extract_keywords: {e}")
        return [{"error": f"Failed to extract keywords: {str(e)}"}]

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

@tool
def summarize_text(text: str, sentences_count: int = 3) -> str:
    """
    Summarize a text using Latent Semantic Analysis (LSA).
    
    Args:
        text: The text to summarize
        sentences_count: Number of sentences in the summary (default: 3)
    
    Returns:
        str: Summarized text
    """
    try:
        logger.info("Generating summary...")
        
        # Initialize the parser and tokenizer
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        
        # Initialize the stemmer
        stemmer = Stemmer("english")
        
        # Initialize the summarizer
        summarizer = LsaSummarizer(stemmer)
        summarizer.stop_words = get_stop_words("english")
        
        # Generate the summary
        summary_sentences = summarizer(parser.document, sentences_count)
        
        # Join the sentences
        summary = " ".join(str(sentence) for sentence in summary_sentences)
        
        return summary.strip()
    except Exception as e:
        logger.error(f"Error in summarize_text: {e}")
        # Fallback to simple text truncation
        sentences = text.split('.')
        return ". ".join(sentences[:min(sentences_count, len(sentences))]) + "."

@tool
def summarize_with_llm(text: str, model_name: str = "gemini-2.0-flash", temperature: float = 0.3) -> str:
    """
    Summarize a text using a language model.
    
    Args:
        text: The text to summarize
        model_name: Name of the model to use (default: 'gemini-2.0-flash')
        temperature: Temperature for generation (default: 0.3)
    
    Returns:
        str: Summarized text
    """
    try:
        logger.info(f"Generating summary using {model_name}...")
        
        # Initialize the model
        model = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            convert_system_message_to_human=True
        )
        
        # Generate the summary
        response = model.invoke([
            SystemMessage(content="You are a text summarization specialist. Summarize the following text in 2-3 sentences while preserving key information."),
            HumanMessage(content=text)
        ])
        
        summary = response.content.strip()
        
        # Ensure the summary ends with a period
        if summary and not summary.endswith(('.', '!', '?')):
            summary += '.'
            
        return summary
    except Exception as e:
        logger.error(f"Error in summarize_with_llm: {e}")
        # Fall back to extractive summarization if LLM fails
        return summarize_text(text, sentences_count=3)

# Optimized versions of the tools with caching
def cached_search_information(query: str, state: OptimizedState) -> Tuple[str, OptimizedState]:
    """Search for information with caching."""
    try:
        start_time = time.time()
        
        # Update session activity
        state.session_data.update_activity()
        state.session_data.increment_interaction()
        
        # Execute with caching
        result, is_cache_hit = cached_execution("search_information", search_information, query)
        
        execution_time = time.time() - start_time
        
        # Update the state
        state.results["search"] = result
        state.execution_times["search"] = execution_time
        
        if is_cache_hit:
            state.cache_hits += 1
        else:
            state.cache_misses += 1
        
        logger.info(f"Search executed in {execution_time:.2f}s (cached: {is_cache_hit})")
        
        return result, state
    except Exception as e:
        logger.error(f"Error in cached_search_information: {e}")
        # Return a default response in case of error
        return f"Error performing search: {str(e)}", state

def cached_analyze_sentiment(text: str, state: OptimizedState) -> Tuple[Dict[str, float], OptimizedState]:
    """Analyze sentiment with caching."""
    start_time = time.time()
    
    result, is_cache_hit = cached_execution("analyze_sentiment", analyze_sentiment, text)
    execution_time = time.time() - start_time
    
    # Update the state
    updated_results = dict(state.results or {})
    updated_results["sentiment"] = result
    
    updated_execution_times = dict(state.execution_times or {})
    updated_execution_times["sentiment"] = execution_time
    
    updates = {
        "results": updated_results,
        "execution_times": updated_execution_times,
        "cache_hits": state.cache_hits + (1 if is_cache_hit else 0),
        "cache_misses": state.cache_misses + (0 if is_cache_hit else 1)
    }
    
    new_state = state.model_copy(update=updates)
    return result, new_state

def cached_extract_keywords(text: str, state: OptimizedState) -> Tuple[List[str], OptimizedState]:
    """Extract keywords with caching."""
    start_time = time.time()
    
    result, is_cache_hit = cached_execution("extract_keywords", extract_keywords, text)
    
    execution_time = time.time() - start_time
    
    # Update the state
    updated_results = dict(state.results or {})
    updated_results["keywords"] = result
    
    updated_execution_times = dict(state.execution_times or {})
    updated_execution_times["keywords"] = execution_time
    
    updates = {
        "results": updated_results,
        "execution_times": updated_execution_times,
        "cache_hits": state.cache_hits + (1 if is_cache_hit else 0),
        "cache_misses": state.cache_misses + (0 if is_cache_hit else 1)
    }
    
    new_state = state.model_copy(update=updates)
    
    return result, new_state

def cached_summarize_text(text: str, state: OptimizedState) -> Tuple[str, OptimizedState]:
    """Summarize text with caching."""
    start_time = time.time()
    
    result, is_cache_hit = cached_execution("summarize_text", summarize_text, text)
    
    execution_time = time.time() - start_time
    
    # Update the state
    updated_results = dict(state.results or {})
    updated_results["summary"] = result
    
    updated_execution_times = dict(state.execution_times or {})
    updated_execution_times["summary"] = execution_time
    
    updates = {
        "results": updated_results,
        "execution_times": updated_execution_times
    }
    
    if is_cache_hit:
        updates["cache_hits"] = state.cache_hits + 1
    else:
        updates["cache_misses"] = state.cache_misses + 1
    
    new_state = state.model_copy(update=updates)
    return result, new_state

# Workflow nodes
def task_preparation_node(state: OptimizedState) -> OptimizedState:
    """Prepare the tasks for parallel execution."""
    print("🔍 Preparing tasks for parallel execution...")
    
    # Extract the user query from the messages
    messages = state.messages or []
    if not messages:
        return state
        
    last_message = messages[-1]
    
    if isinstance(last_message, HumanMessage):
        # Define parallel tasks
        parallel_tasks = ["search", "sentiment", "keywords"]
        
        # Create a new state with updated values
        state_dict = state.model_dump()
        state_dict.update({
            "parallel_tasks": parallel_tasks,
            "next": "parallel_execution"
        })
        return OptimizedState.from_dict(state_dict)
    
    return state

async def parallel_execution_node(state: OptimizedState) -> OptimizedState:
    """Execute tasks in parallel."""
    print("⚡ Executing tasks in parallel...")
    
    # Extract the user query from the messages
    messages = state.messages or []
    if not messages:
        return state
        
    last_message = messages[-1]
    
    if isinstance(last_message, HumanMessage):
        query = last_message.content
        
        # Define the tasks to run in parallel
        tasks = []
        task_types = []
        
        async def run_task(task_func, *args, **kwargs):
            """Helper to run sync functions in a thread pool."""
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, lambda: task_func(*args, **kwargs))
        
        # Create a copy of the state for each task to avoid sharing state between tasks
        if state.parallel_tasks and "search" in state.parallel_tasks:
            task_state = state.model_copy()
            tasks.append(run_task(cached_search_information, query, task_state))
            task_types.append("search")
            
        if state.parallel_tasks and "sentiment" in state.parallel_tasks:
            task_state = state.model_copy()
            tasks.append(run_task(cached_analyze_sentiment, query, task_state))
            task_types.append("sentiment")
            
        if state.parallel_tasks and "keywords" in state.parallel_tasks:
            task_state = state.model_copy()
            tasks.append(run_task(cached_extract_keywords, query, task_state))
            task_types.append("keywords")
        
        if not tasks:
            return state
            
        # Run the tasks in parallel using asyncio.gather
        start_time = time.time()
        task_results = await asyncio.gather(*tasks, return_exceptions=True)
        parallel_execution_time = time.time() - start_time
        # Initialize updated state
        updated_results = dict(state.results or {})
        updated_execution_times = dict(state.execution_times or {})
        total_cache_hits = state.cache_hits
        total_cache_misses = state.cache_misses
        
        # Process results
        for i, result in enumerate(task_results):
            task_type = task_types[i] if i < len(task_types) else f"task_{i}"
            
            if isinstance(result, Exception):
                logger.error(f"Error in {task_type}: {str(result)}")
                updated_results[task_type] = f"Error: {str(result)}"
                continue
                
            # Handle the result which is a tuple of (result, state)
            if isinstance(result, tuple) and len(result) == 2:
                task_result, task_state = result
                
                # Update the results with the task result
                if task_type == "search" and task_result is not None:
                    updated_results["search"] = task_result
                    if hasattr(task_state, 'execution_times') and "search" in task_state.execution_times:
                        updated_execution_times["search"] = task_state.execution_times["search"]
                    
                elif task_type == "sentiment" and task_result is not None:
                    updated_results["sentiment"] = task_result
                    if hasattr(task_state, 'execution_times') and "sentiment" in task_state.execution_times:
                        updated_execution_times["sentiment"] = task_state.execution_times["sentiment"]
                    
                elif task_type == "keywords" and task_result is not None:
                    updated_results["keywords"] = task_result
                    if hasattr(task_state, 'execution_times') and "keywords" in task_state.execution_times:
                        updated_execution_times["keywords"] = task_state.execution_times["keywords"]
                
                # Update cache statistics
                if hasattr(task_state, 'cache_hits') and hasattr(task_state, 'cache_misses'):
                    total_cache_hits = max(total_cache_hits, task_state.cache_hits)
                    total_cache_misses = max(total_cache_misses, task_state.cache_misses)
        
        # Add parallel execution time
        updated_execution_times["parallel"] = parallel_execution_time
        # Create the new state with all updates including parallel execution time
        new_state = state.model_copy(update={
            "results": updated_results,
            "execution_times": updated_execution_times,
            "cache_hits": total_cache_hits,
            "cache_misses": total_cache_misses,
            "next": "summarization"
        })
        return new_state
    
    return state

def summarization_node(state: OptimizedState) -> OptimizedState:
    """Summarize the results."""
    print("📝 Summarizing results...")
    
    # Extract the user query and results from the state
    messages = state.messages or []
    if not messages:
        print("No messages found in state")
        return state
        
    last_message = messages[-1]
    results = state.results or {}
    
    print(f"Current state results keys: {list(results.keys())}")
    print(f"Last message type: {type(last_message).__name__}")
    
    if isinstance(last_message, HumanMessage):
        query = last_message.content
        print(f"Processing query: {query[:100]}..." if len(query) > 100 else f"Processing query: {query}")
        
        # Prepare the text for summarization
        text = f"Query: {query}\n\n"
        
        if "search" in results:
            search_results = results['search']
            # Format search results for better readability
            if isinstance(search_results, str):
                # If it's a string, just use it as is
                formatted_search = search_results
            elif isinstance(search_results, (list, tuple)):
                # If it's a list, join the items with newlines
                formatted_search = '\n'.join(str(item) for item in search_results)
            elif isinstance(search_results, dict):
                # If it's a dict, format key-value pairs
                formatted_search = '\n'.join(f"{k}: {v}" for k, v in search_results.items())
            else:
                # Fallback to string representation
                formatted_search = str(search_results)
            
            # Log the full search results for debugging
            print("\n" + "="*50)
            print("SEARCH RESULTS:")
            print(formatted_search)
            print("="*50 + "\n")
            
            # Add a truncated version to the summary text
            max_search_length = 1000  # Maximum length for the summary
            if len(formatted_search) > max_search_length:
                summary_search = formatted_search[:max_search_length] + "... [truncated]"
            else:
                summary_search = formatted_search
                
            text += f"## Search Results\n{summary_search}\n\n"
        
        # Summarize the text
        print("Calling cached_summarize_text...")
        summary, new_state = cached_summarize_text(text, state)
        print(f"Summary generated: {summary[:200]}..." if summary and len(summary) > 200 else f"Summary generated: {summary}")
        
        # Create a response message
        response = f"# Analysis Results\n\n"
        response += f"## Summary\n{summary}\n\n" if summary else "## No summary generated\n\n"
        
        if "keywords" in results and results["keywords"]:
            keywords = results["keywords"]
            print(f"Found keywords: {keywords}")
            if isinstance(keywords, (list, tuple)):
                response += f"## Keywords\n{', '.join(str(k) for k in keywords)}\n\n"
            else:
                response += f"## Keywords\n{keywords}\n\n"
        
        if "sentiment" in results and results["sentiment"]:
            sentiment = results["sentiment"]
            print(f"Found sentiment: {sentiment}")
            response += f"## Sentiment Analysis\n"
            if isinstance(sentiment, dict):
                for key, value in sentiment.items():
                    if isinstance(value, (int, float)):
                        response += f"- {key.title()}: {value:.4f}\n"
                    else:
                        response += f"- {key.title()}: {value}\n"
        
        # Add performance metrics
        execution_times = new_state.execution_times or {}
        total_time = sum(execution_times.values())
        parallel_time = execution_times.get("parallel", 0)
        
        response += "\n## Performance Metrics\n"
        response += f"- Total execution time: {total_time:.4f}s\n"
        response += f"- Parallel execution time: {parallel_time:.4f}s\n"
        if total_time > 0:
            response += f"- Time saved through parallelization: {(total_time - parallel_time):.4f}s ({(1 - (parallel_time / total_time)) * 100:.1f}%)\n"
        
        response += f"- Cache hits: {new_state.cache_hits}\n"
        response += f"- Cache misses: {new_state.cache_misses}\n"
        if new_state.cache_hits + new_state.cache_misses > 0:
            hit_rate = (new_state.cache_hits / (new_state.cache_hits + new_state.cache_misses)) * 100
            response += f"- Cache hit rate: {hit_rate:.1f}%\n"
        
        print("\n" + "="*50)
        print("FINAL RESPONSE:")
        print(response)
        print("="*50 + "\n")
        
        # Create a new AIMessage with the response
        response_message = AIMessage(content=response)
        
        # Add the response to the messages
        new_messages = messages + [response_message]
        
        # Update the state with the new messages and set next to end
        updated_state = state.model_copy(update={
            "messages": new_messages,
            "next": "end"
        })
        
        print(f"Updated state messages count: {len(updated_state.messages)}")
        if updated_state.messages:
            print(f"Last message type: {type(updated_state.messages[-1]).__name__}")
            print(f"Last message content: {str(updated_state.messages[-1].content)[:200]}...")
        
        return updated_state
    
    print(f"Last message is not a HumanMessage: {type(last_message).__name__}")
    return state

def create_optimized_workflow():
    """Create an optimized workflow using LangGraph."""
    print("Creating an optimized workflow with LangGraph...")
    
    # Create a new graph
    workflow = StateGraph(OptimizedState)
    
    # Add nodes to the graph
    workflow.add_node("task_preparation", task_preparation_node)
    workflow.add_node("parallel_execution", parallel_execution_node)
    workflow.add_node("summarization", summarization_node)
    
    # Add edges
    workflow.add_conditional_edges(
        "task_preparation",
        lambda state: state.next,
        {
            "parallel_execution": "parallel_execution"
        }
    )
    
    workflow.add_conditional_edges(
        "parallel_execution",
        lambda state: state.next,
        {
            "summarization": "summarization"
        }
    )
    
    workflow.add_conditional_edges(
        "summarization",
        lambda state: state.next,
        {
            "end": END
        }
    )
    
    # Set the entry point
    workflow.set_entry_point("task_preparation")
    
    # Compile the graph
    return workflow.compile()

def sequential_tool_node(state: OptimizedState) -> OptimizedState:
    """Execute tools sequentially in a single node."""
    if not state.messages or not isinstance(state.messages[-1], HumanMessage):
        return state
        
    query = state.messages[-1].content
    results = {}
    
    # Execute tools sequentially
    if "search" in state.parallel_tasks:
        search_result, _ = cached_search_information(query, state)
        results["search"] = search_result
        
    if "sentiment" in state.parallel_tasks:
        sentiment_result, _ = cached_analyze_sentiment(query, state)
        results["sentiment"] = sentiment_result
        
    if "keywords" in state.parallel_tasks:
        keywords_result, _ = cached_extract_keywords(query, state)
        results["keywords"] = keywords_result
    
    # Create a response similar to the parallel version
    response = "# Sequential Analysis Results\n\n"
    
    if "search" in results:
        response += f"## Search Results\n{results['search']}\n\n"
    
    if "sentiment" in results:
        sentiment = results["sentiment"]
        response += "## Sentiment Analysis\n"
        if isinstance(sentiment, dict):
            for key, value in sentiment.items():
                if isinstance(value, (int, float)):
                    response += f"- {key.title()}: {value:.4f}\n"
                else:
                    response += f"- {key.title()}: {value}\n"
        response += "\n"
    
    if "keywords" in results:
        keywords = results["keywords"]
        response += "## Keywords\n"
        if isinstance(keywords, (list, tuple)):
            response += ", ".join(str(k) for k in keywords)
        else:
            response += str(keywords)
        response += "\n\n"
    
    # Create a new state with the response
    new_messages = state.messages + [AIMessage(content=response)]
    
    return state.model_copy(update={
        "messages": new_messages,
        "results": results,
        "next": "end"
    })

def create_sequential_workflow():
    """Create a sequential workflow for comparison."""
    print("Creating a sequential workflow for comparison...")
    
    # Create a new graph
    workflow = StateGraph(OptimizedState)
    
    # Add our sequential node
    workflow.add_node("sequential_tools", sequential_tool_node)
    
    # Add edge to end
    workflow.add_edge("sequential_tools", END)
    
    # Set the entry point
    workflow.set_entry_point("sequential_tools")
        
    # Compile the graph
    return workflow.compile()

async def run_optimized_workflow(query: str):
    """Run the optimized workflow with the given query."""
    print(f"🚀 Starting workflow with query: {query}")
    start_time = time.time()
    
    # Create the workflow - this already returns a compiled workflow
    app = create_optimized_workflow()
    
    # Create the initial state
    initial_state = OptimizedState(
        messages=[HumanMessage(content=query)],
        parallel_tasks=["search", "sentiment", "keywords"]
    )
    
    # Run the workflow asynchronously
    result = await app.ainvoke(initial_state)
    total_time = time.time() - start_time
    
    # Print the results
    print("\n" + "="*50)
    print("Workflow Execution Complete")
    print("="*50)
    
    # Print all messages in the result
    if result and hasattr(result, 'messages') and result.messages:
        print("\nResults:")
        for msg in result.messages:
            if hasattr(msg, 'content'):
                content = msg.content
                # Skip empty or system messages
                if content and not (isinstance(msg, SystemMessage) and not content.strip()):
                    print(f"\n{'-'*20}")
                    print(f"{type(msg).__name__}: {content}")
    
    # Print execution time and metrics
    execution_times = getattr(result, 'execution_times', {}) or {}
    total_execution_time = sum(execution_times.values())
    
    print("\n" + "="*50)
    print("Performance Metrics:")
    print("="*50)
    print(f"Total workflow execution time: {total_time:.2f}s")
    print(f"Total task execution time: {total_execution_time:.2f}s")
    print(f"Cache hits: {getattr(result, 'cache_hits', 0)}")
    print(f"Cache misses: {getattr(result, 'cache_misses', 0)}")
    print("="*50)
    
    return result

async def compare_workflows():
    """Compare optimized and sequential workflows."""
    print("Comparing optimized and sequential workflows...")
    
    # Get user input
    query = input("Enter your query: ")
    
    # Create initial state
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "parallel_tasks": ["search", "sentiment", "keywords"],
        "results": {},
        "cache_hits": 0,
        "cache_misses": 0,
        "execution_times": {}
    }
    
    # Run optimized workflow
    print("\n" + "="*50)
    print("Running optimized workflow (with parallel execution)...")
    print("="*50)
    try:
        optimized_workflow = create_optimized_workflow()
        start_time = time.time()
        optimized_result = await optimized_workflow.ainvoke(initial_state)
        optimized_time = time.time() - start_time
        
        # Print optimized workflow results
        if hasattr(optimized_result, 'messages') and optimized_result.messages:
            print("\nOptimized Workflow Results:")
            for msg in optimized_result.messages:
                if hasattr(msg, 'content') and msg.content and not isinstance(msg, SystemMessage):
                    print(f"\n{msg.content}")
    except Exception as e:
        print(f"\nError in optimized workflow: {str(e)}")
        optimized_time = 0
    
    # Run sequential workflow
    print("\n" + "="*50)
    print("Running sequential workflow (without optimizations)...")
    print("="*50)
    try:
        sequential_workflow = create_sequential_workflow()
        start_time = time.time()
        sequential_result = await sequential_workflow.ainvoke(initial_state)
        sequential_time = time.time() - start_time
        
        # Print sequential workflow results
        if hasattr(sequential_result, 'messages') and sequential_result.messages:
            print("\nSequential Workflow Results:")
            for msg in sequential_result.messages:
                if hasattr(msg, 'content') and msg.content and not isinstance(msg, SystemMessage):
                    print(f"\n{msg.content}")
    except Exception as e:
        print(f"\nError in sequential workflow: {str(e)}")
        sequential_time = 0
    
    # Print comparison
    print("\n" + "="*50)
    print("Workflow Comparison")
    print("="*50)
    print(f"Optimized workflow time: {optimized_time:.4f}s")
    print(f"Sequential workflow time: {sequential_time:.4f}s")
    
    if optimized_time > 0 and sequential_time > 0:
        improvement = ((sequential_time - optimized_time) / sequential_time) * 100
        print(f"Improvement: {improvement:.1f}% faster")
    
    print("\nComparison complete!")

class AutonomousAgent:
    """An autonomous agent that manages the workflow and user interactions."""
    
    def __init__(self, model_name: str = "gemini-2.0-flash"):
        """Initialize the autonomous agent."""
        self.model_name = model_name
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0.3,
            convert_system_message_to_human=True
        )
        self.workflow = create_optimized_workflow()
        # Initialize memory with return_messages=True to maintain message format
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="output"
        )
        self._setup_agent()
    
    def _setup_agent(self):
        """Set up the agent with tools and prompt."""
        # Define the tools available to the agent
        self.tools = [
            Tool(
                name="analyze_text",
                func=self._analyze_text,
                description="""Useful for analyzing text. 
                Input should be a JSON string with 'text' and 'analysis_type' keys.
                analysis_type can be 'sentiment', 'keywords', or 'summary'.""",
                return_direct=True
            ),
            Tool(
                name="search_web",
                func=self._search_web,
                description="""Useful for searching the web for information.
                Input should be a search query string.""",
                return_direct=True
            )
        ]
        
        # Initialize the agent with the tools
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5,
            early_stopping_method="generate",
            memory=self.memory
        )
        
        # Create the agent executor
        self.agent_executor = self.agent
    
    def _analyze_text(self, input_str: str) -> str:
        """Analyze text using the workflow."""
        try:
            # Parse the input
            try:
                input_data = json.loads(input_str)
                text = input_data.get("text", "")
                analysis_type = input_data.get("analysis_type", "summary")
                
                if not text:
                    return "Error: No text provided for analysis."
                
                # Create a new state for analysis
                state = OptimizedState()
                
                # Run the appropriate analysis
                if analysis_type == "sentiment":
                    result, _ = cached_analyze_sentiment(text, state)
                    if isinstance(result, dict):
                        return f"Sentiment Analysis:\n{json.dumps(result, indent=2)}"
                    return f"Sentiment Analysis:\n{result}"
                elif analysis_type == "keywords":
                    result, _ = cached_extract_keywords(text, state)
                    if isinstance(result, (dict, list)):
                        return f"Extracted Keywords:\n{json.dumps(result, indent=2)}"
                    return f"Extracted Keywords:\n{result}"
                else:  # summary
                    result, _ = cached_summarize_text(text, state)
                    return f"Summary:\n{result}"
                    
            except json.JSONDecodeError:
                # If input is not JSON, treat it as plain text and return a summary
                state = OptimizedState()
                result, _ = cached_summarize_text(input_str, state)
                return f"Summary:\n{result}"
                
        except Exception as e:
            logger.error(f"Error in _analyze_text: {e}")
            return f"Error analyzing text: {str(e)}"
    
    def _search_web(self, query: str) -> str:
        """Search the web for information."""
        try:
            state = OptimizedState()
            result, _ = cached_search_information(query, state)
            if isinstance(result, (dict, list)):
                return json.dumps(result, indent=2)
            return str(result)
        except Exception as e:
            logger.error(f"Error in _search_web: {e}", exc_info=True)
            return f"Error searching the web: {str(e)}"
    
    async def process_query(self, query: str) -> str:
        """Process a user query using the agent."""
        try:
            # Update session activity
            self.memory.chat_memory.add_user_message(query)
            
            try:
                # Try to run the agent with the query
                response = await self.agent_executor.ainvoke({"input": query})
                
                # Handle different response formats
                if isinstance(response, dict):
                    output = response.get('output', response.get('response', str(response)))
                else:
                    output = str(response)
                    
            except Exception as agent_error:
                logger.error(f"Agent execution error: {agent_error}", exc_info=True)
                output = "I'm sorry, I encountered an error while processing your request. Could you please rephrase or try a different query?"
            
            # Ensure output is a string and not empty
            output = str(output) if output else "I'm not sure how to respond to that."
            
            # Clean up the output
            output = output.strip()
            
            # Add the AI response to memory if it's not already there
            if not self.memory.chat_memory.messages or self.memory.chat_memory.messages[-1].content != output:
                self.memory.chat_memory.add_ai_message(output)
            
            return output
            
        except Exception as e:
            logger.error(f"Error in process_query: {e}", exc_info=True)
            error_msg = "I apologize, but I'm having trouble processing your request right now. Could you please try again?"
            self.memory.chat_memory.add_ai_message(error_msg)
            return error_msg
    
    async def _process_user_input(self, user_input: str) -> str:
        """Process a single user input and return the response."""
        try:
            # Process the query asynchronously
            response = await self.process_query(user_input)
            return response
        except Exception as e:
            error_msg = f"An error occurred: {str(e)}"
            logger.error(f"Error processing user input: {e}", exc_info=True)
            return error_msg
    
    async def _run_cli_async(self):
        """Async implementation of the CLI."""
        print("=" * 50)
        print("Autonomous Text Analysis Agent")
        print("Type 'exit' to quit the application.")
        print("=" * 50)
        
        while True:
            try:
                # Get user input in a thread to avoid blocking the event loop
                user_input = await asyncio.get_event_loop().run_in_executor(
                    None, input, "\nYou: "
                )
                user_input = user_input.strip()
                
                # Check for exit command
                if user_input.lower() in ('exit', 'quit', 'bye'):
                    print("\nGoodbye!")
                    break
                
                if not user_input:
                    continue
                
                # Process the query
                print("\nAssistant: ", end="", flush=True)
                
                # Process the query asynchronously
                response = await self._process_user_input(user_input)
                
                # Print the response
                print(response)
                
            except asyncio.CancelledError:
                print("\n\nOperation cancelled.")
                break
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\nAn error occurred: {str(e)}")
                continue
    
    def run_cli(self):
        """Run the agent in the command line interface."""
        try:
            asyncio.run(self._run_cli_async())
        except KeyboardInterrupt:
            print("\n\nOperation cancelled by user.")
        except Exception as e:
            print(f"\nAn unexpected error occurred: {str(e)}")

async def main_async():
    """Async main function to demonstrate the autonomous agent."""
    print("=" * 50)
    print("Autonomous Text Analysis Agent")
    print("=" * 50)
    
    while True:
        try:
            # Let the user choose a mode
            print("\nSelect a mode:")
            print("1. Interactive chat with autonomous agent")
            print("2. Run optimized workflow")
            print("3. Compare optimized and sequential workflows")
            print("4. Run with different cache configurations")
            print("5. Exit")
            
            # Use run_in_executor to avoid blocking the event loop
            choice = await asyncio.get_event_loop().run_in_executor(
                None, input, "\nEnter your choice (1-5): "
            )
            choice = choice.strip()
            
            if choice == "1":
                # Initialize and run the autonomous agent
                agent = AutonomousAgent()
                await agent._run_cli_async()
                break
                
            elif choice == "2":
                # Create an optimized workflow
                workflow = create_optimized_workflow()
                
                # Get a query from the user
                query = await asyncio.get_event_loop().run_in_executor(
                    None, input, "\nEnter a query for analysis: "
                )
                
                # Run the workflow
                await run_optimized_workflow(query)
                break
                
            elif choice == "3":
                # Compare workflows
                await compare_workflows()
                break
                
            elif choice == "4":
                # Run with different cache configurations
                print("\nRunning with empty cache (first run)...")
                
                # Create an optimized workflow
                workflow = create_optimized_workflow()
                
                # Get a query from the user
                query = await asyncio.get_event_loop().run_in_executor(
                    None, input, "\nEnter a query for analysis: "
                )
                
                # First run (empty cache)
                await run_optimized_workflow(query)
                
                print("\nRunning with populated cache (second run)...")
                
                # Second run (populated cache)
                await run_optimized_workflow(query)
                break
                
            elif choice == "5":
                print("\nGoodbye!")
                return
                
            else:
                print("\nPlease enter a number between 1 and 5.")
                
        except asyncio.CancelledError:
            print("\n\nOperation cancelled.")
            break
        except KeyboardInterrupt:
            print("\n\nOperation cancelled by user.")
            break
        except Exception as e:
            print(f"\nAn error occurred: {str(e)}")
            continue

def main():
    """Main entry point."""
    loop = None
    try:
        # Create and set a new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(main_async())
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {str(e)}")
    finally:
        # Clean up the event loop if it exists
        if loop is not None:
            try:
                # Cancel all running tasks
                pending = asyncio.all_tasks(loop=loop)
                for task in pending:
                    task.cancel()
                
                # Run the loop until all tasks are done
                if pending:
                    loop.run_until_complete(
                        asyncio.gather(*pending, return_exceptions=True)
                    )
                
                # Close the loop
                loop.close()
            except Exception as e:
                print(f"Error during cleanup: {e}")
                if loop.is_running():
                    loop.stop()
            finally:
                asyncio.set_event_loop(None)

if __name__ == "__main__":
    main()
