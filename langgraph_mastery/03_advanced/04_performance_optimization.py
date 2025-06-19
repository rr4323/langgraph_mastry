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
from typing import Annotated, Dict, List, Literal, TypedDict, Union, Any, Optional, Tuple
from enum import Enum
from functools import lru_cache
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# Add the parent directory to the path so we can import from the root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables from .env file
load_dotenv()

# Define our state for the optimized workflow
class OptimizedState(TypedDict):
    """State for our optimized workflow."""
    messages: List[Union[HumanMessage, AIMessage, SystemMessage]]
    results: Dict[str, Any]
    cache_hits: int
    cache_misses: int
    execution_times: Dict[str, float]
    parallel_tasks: List[str]
    next: Optional[str]

# Simple in-memory cache
CACHE = {}

def get_cache_key(func_name: str, args: Tuple, kwargs: Dict) -> str:
    """Generate a cache key for a function call."""
    # Convert arguments to a string representation
    args_str = json.dumps(args, sort_keys=True)
    kwargs_str = json.dumps(kwargs, sort_keys=True)
    
    # Create a hash of the function name and arguments
    key = hashlib.md5(f"{func_name}:{args_str}:{kwargs_str}".encode()).hexdigest()
    
    return key

def cached_execution(func_name: str, func, *args, **kwargs):
    """Execute a function with caching."""
    # Generate a cache key
    cache_key = get_cache_key(func_name, args, kwargs)
    
    # Check if the result is in the cache
    if cache_key in CACHE:
        print(f"🔄 Cache hit for {func_name}")
        return CACHE[cache_key], True
    
    # Execute the function
    start_time = time.time()
    result = func(*args, **kwargs)
    execution_time = time.time() - start_time
    
    # Store the result in the cache
    CACHE[cache_key] = result
    
    print(f"🆕 Cache miss for {func_name}, execution time: {execution_time:.2f}s")
    
    return result, False

# Define tools with caching
@tool
def search_information(query: str) -> str:
    """
    Search for information related to a query.
    
    Args:
        query: The search query
    
    Returns:
        str: Search results
    """
    # Simulate API call delay
    time.sleep(2)
    
    # Simulate search results
    return f"Here are the search results for '{query}':\n\n" + \
           f"1. {query} is a topic of great interest in recent research.\n" + \
           f"2. According to recent studies, {query} has shown significant developments.\n" + \
           f"3. Experts in {query} suggest several approaches to understanding this topic.\n" + \
           f"4. The history of {query} dates back to several decades ago.\n" + \
           f"5. Future trends in {query} indicate promising directions for further exploration."

@tool
def analyze_sentiment(text: str) -> Dict[str, float]:
    """
    Analyze the sentiment of a text.
    
    Args:
        text: The text to analyze
    
    Returns:
        Dict: Sentiment scores
    """
    # Simulate API call delay
    time.sleep(1.5)
    
    # Simulate sentiment analysis
    import random
    positive = random.uniform(0, 1)
    negative = random.uniform(0, 1 - positive)
    neutral = 1 - positive - negative
    
    return {
        "positive": round(positive, 2),
        "negative": round(negative, 2),
        "neutral": round(neutral, 2)
    }

@tool
def extract_keywords(text: str) -> List[str]:
    """
    Extract keywords from a text.
    
    Args:
        text: The text to analyze
    
    Returns:
        List[str]: Extracted keywords
    """
    # Simulate API call delay
    time.sleep(1)
    
    # Simulate keyword extraction
    # In a real application, you would use an NLP library
    words = text.lower().replace(".", "").replace(",", "").replace("!", "").replace("?", "").split()
    
    # Remove common words
    common_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "with", "by", "about", "as"}
    keywords = [word for word in words if word not in common_words and len(word) > 3]
    
    # Get unique keywords
    unique_keywords = list(set(keywords))
    
    # Limit to 5 keywords
    return unique_keywords[:5]

@tool
def summarize_text(text: str) -> str:
    """
    Summarize a text.
    
    Args:
        text: The text to summarize
    
    Returns:
        str: Summarized text
    """
    # Simulate API call delay
    time.sleep(3)
    
    # Use LLM for summarization
    model = ChatGoogleGenerativeAI(
        model="gemini-pro",
        temperature=0.3,
        convert_system_message_to_human=True
    )
    
    response = model.invoke([
        SystemMessage(content="You are a text summarization specialist. Summarize the following text in 2-3 sentences."),
        HumanMessage(content=text)
    ])
    
    return response.content

# Optimized versions of the tools with caching
def cached_search_information(query: str, state: OptimizedState) -> Tuple[str, OptimizedState]:
    """Search for information with caching."""
    start_time = time.time()
    
    result, is_cache_hit = cached_execution("search_information", search_information, query)
    
    execution_time = time.time() - start_time
    
    # Update the state
    new_state = state.copy()
    new_state["results"]["search"] = result
    new_state["execution_times"]["search"] = execution_time
    
    if is_cache_hit:
        new_state["cache_hits"] += 1
    else:
        new_state["cache_misses"] += 1
    
    return result, new_state

def cached_analyze_sentiment(text: str, state: OptimizedState) -> Tuple[Dict[str, float], OptimizedState]:
    """Analyze sentiment with caching."""
    start_time = time.time()
    
    result, is_cache_hit = cached_execution("analyze_sentiment", analyze_sentiment, text)
    
    execution_time = time.time() - start_time
    
    # Update the state
    new_state = state.copy()
    new_state["results"]["sentiment"] = result
    new_state["execution_times"]["sentiment"] = execution_time
    
    if is_cache_hit:
        new_state["cache_hits"] += 1
    else:
        new_state["cache_misses"] += 1
    
    return result, new_state

def cached_extract_keywords(text: str, state: OptimizedState) -> Tuple[List[str], OptimizedState]:
    """Extract keywords with caching."""
    start_time = time.time()
    
    result, is_cache_hit = cached_execution("extract_keywords", extract_keywords, text)
    
    execution_time = time.time() - start_time
    
    # Update the state
    new_state = state.copy()
    new_state["results"]["keywords"] = result
    new_state["execution_times"]["keywords"] = execution_time
    
    if is_cache_hit:
        new_state["cache_hits"] += 1
    else:
        new_state["cache_misses"] += 1
    
    return result, new_state

def cached_summarize_text(text: str, state: OptimizedState) -> Tuple[str, OptimizedState]:
    """Summarize text with caching."""
    start_time = time.time()
    
    result, is_cache_hit = cached_execution("summarize_text", summarize_text, text)
    
    execution_time = time.time() - start_time
    
    # Update the state
    new_state = state.copy()
    new_state["results"]["summary"] = result
    new_state["execution_times"]["summary"] = execution_time
    
    if is_cache_hit:
        new_state["cache_hits"] += 1
    else:
        new_state["cache_misses"] += 1
    
    return result, new_state

# Workflow nodes
def task_preparation_node(state: OptimizedState) -> OptimizedState:
    """Prepare the tasks for parallel execution."""
    print("🔍 Preparing tasks for parallel execution...")
    
    # Extract the user query from the messages
    messages = state["messages"]
    last_message = messages[-1]
    
    if isinstance(last_message, HumanMessage):
        query = last_message.content
        
        # Define parallel tasks
        parallel_tasks = ["search", "sentiment", "keywords"]
        
        # Update the state
        return {
            **state,
            "parallel_tasks": parallel_tasks,
            "next": "parallel_execution"
        }
    
    return state

async def parallel_execution_node(state: OptimizedState) -> OptimizedState:
    """Execute tasks in parallel."""
    print("⚡ Executing tasks in parallel...")
    
    # Extract the user query from the messages
    messages = state["messages"]
    last_message = messages[-1]
    
    if isinstance(last_message, HumanMessage):
        query = last_message.content
        
        # Define the tasks to run in parallel
        tasks = []
        
        if "search" in state["parallel_tasks"]:
            tasks.append(asyncio.to_thread(cached_search_information, query, state))
        
        if "sentiment" in state["parallel_tasks"]:
            tasks.append(asyncio.to_thread(cached_analyze_sentiment, query, state))
        
        if "keywords" in state["parallel_tasks"]:
            tasks.append(asyncio.to_thread(cached_extract_keywords, query, state))
        
        # Run the tasks in parallel
        start_time = time.time()
        results = await asyncio.gather(*tasks)
        parallel_execution_time = time.time() - start_time
        
        # Merge the results and state updates
        new_state = state.copy()
        
        for _, task_state in results:
            # Merge results
            for key, value in task_state["results"].items():
                if key in new_state["results"]:
                    continue
                new_state["results"][key] = value
            
            # Merge execution times
            for key, value in task_state["execution_times"].items():
                if key in new_state["execution_times"]:
                    continue
                new_state["execution_times"][key] = value
            
            # Update cache stats
            new_state["cache_hits"] = task_state["cache_hits"]
            new_state["cache_misses"] = task_state["cache_misses"]
        
        # Add parallel execution time
        new_state["execution_times"]["parallel"] = parallel_execution_time
        
        # Set the next node
        new_state["next"] = "summarization"
        
        return new_state
    
    return state

def summarization_node(state: OptimizedState) -> OptimizedState:
    """Summarize the results."""
    print("📝 Summarizing results...")
    
    # Extract the user query and results
    messages = state["messages"]
    last_message = messages[-1]
    results = state["results"]
    
    if isinstance(last_message, HumanMessage):
        query = last_message.content
        
        # Prepare the text for summarization
        text = f"Query: {query}\n\n"
        
        if "search" in results:
            text += f"Search Results:\n{results['search']}\n\n"
        
        # Summarize the text
        summary, new_state = cached_summarize_text(text, state)
        
        # Create a response message
        response = f"# Analysis Results\n\n"
        response += f"## Summary\n{summary}\n\n"
        
        if "keywords" in results:
            response += f"## Keywords\n{', '.join(results['keywords'])}\n\n"
        
        if "sentiment" in results:
            sentiment = results["sentiment"]
            response += f"## Sentiment Analysis\n"
            response += f"- Positive: {sentiment['positive']}\n"
            response += f"- Negative: {sentiment['negative']}\n"
            response += f"- Neutral: {sentiment['neutral']}\n\n"
        
        # Add performance metrics
        execution_times = new_state["execution_times"]
        total_time = sum(execution_times.values())
        parallel_time = execution_times.get("parallel", 0)
        
        response += f"## Performance Metrics\n"
        response += f"- Total execution time: {total_time:.2f}s\n"
        response += f"- Parallel execution time: {parallel_time:.2f}s\n"
        response += f"- Time saved through parallelization: {(total_time - parallel_time):.2f}s\n"
        response += f"- Cache hits: {new_state['cache_hits']}\n"
        response += f"- Cache misses: {new_state['cache_misses']}\n"
        
        # Add the response to the messages
        new_messages = add_messages(messages, [AIMessage(content=response)])
        
        # Update the state
        return {
            **new_state,
            "messages": new_messages,
            "next": "end"
        }
    
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
        lambda state: state["next"],
        {
            "parallel_execution": "parallel_execution"
        }
    )
    
    workflow.add_conditional_edges(
        "parallel_execution",
        lambda state: state["next"],
        {
            "summarization": "summarization"
        }
    )
    
    workflow.add_conditional_edges(
        "summarization",
        lambda state: state["next"],
        {
            "end": END
        }
    )
    
    # Set the entry point
    workflow.set_entry_point("task_preparation")
    
    # Compile the graph
    return workflow.compile()

def create_sequential_workflow():
    """Create a sequential workflow for comparison."""
    print("Creating a sequential workflow for comparison...")
    
    # Create a new graph
    workflow = StateGraph(OptimizedState)
    
    # Add a tool node
    tools = [search_information, analyze_sentiment, extract_keywords, summarize_text]
    tool_node = ToolNode(tools)
    
    # Add the tool node to the graph
    workflow.add_node("tools", tool_node)
    
    # Add an edge from the tool node back to itself
    workflow.add_edge("tools", "tools")
    
    # Set the entry point
    workflow.set_entry_point("tools")
    
    # Compile the graph
    return workflow.compile()

def run_optimized_workflow(workflow, query: str):
    """Run the optimized workflow with a given query."""
    print("\n" + "=" * 50)
    print("Running the Optimized Workflow")
    print("=" * 50)
    
    # Initialize the state
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "results": {},
        "cache_hits": 0,
        "cache_misses": 0,
        "execution_times": {},
        "parallel_tasks": [],
        "next": None
    }
    
    # Run the workflow
    print(f"\nQuery: {query}")
    start_time = time.time()
    result = workflow.invoke(initial_state)
    total_time = time.time() - start_time
    
    # Print the workflow results
    print("\n" + "=" * 50)
    print("Workflow Results")
    print("=" * 50)
    
    # Print the last message
    last_message = result["messages"][-1]
    if isinstance(last_message, AIMessage):
        print(last_message.content)
    
    print(f"\nTotal workflow execution time: {total_time:.2f}s")
    
    return result

def compare_workflows():
    """Compare optimized and sequential workflows."""
    print("\n" + "=" * 50)
    print("Comparing Optimized and Sequential Workflows")
    print("=" * 50)
    
    # Create the workflows
    optimized_workflow = create_optimized_workflow()
    sequential_workflow = create_sequential_workflow()
    
    # Get a query from the user
    query = input("\nEnter a query for analysis: ")
    
    # Run the optimized workflow
    print("\nRunning optimized workflow (with parallelization and caching)...")
    start_time = time.time()
    optimized_result = run_optimized_workflow(optimized_workflow, query)
    optimized_time = time.time() - start_time
    
    # Clear the cache for fair comparison
    global CACHE
    CACHE = {}
    
    # Run the sequential workflow
    print("\nRunning sequential workflow (without optimizations)...")
    start_time = time.time()
    sequential_result = sequential_workflow.invoke({
        "messages": [HumanMessage(content=query)],
        "results": {},
        "cache_hits": 0,
        "cache_misses": 0,
        "execution_times": {},
        "parallel_tasks": [],
        "next": None
    })
    sequential_time = time.time() - start_time
    
    # Print comparison
    print("\n" + "=" * 50)
    print("Performance Comparison")
    print("=" * 50)
    
    print(f"\nOptimized workflow execution time: {optimized_time:.2f}s")
    print(f"Sequential workflow execution time: {sequential_time:.2f}s")
    print(f"Time saved: {sequential_time - optimized_time:.2f}s ({(sequential_time - optimized_time) / sequential_time * 100:.2f}%)")
    
    return optimized_result, sequential_result

def main():
    """Main function to demonstrate performance optimization."""
    print("=" * 50)
    print("Performance Optimization in LangGraph")
    print("=" * 50)
    
    # Let the user choose a mode
    print("\nSelect a mode:")
    print("1. Run optimized workflow")
    print("2. Compare optimized and sequential workflows")
    print("3. Run with different cache configurations")
    
    while True:
        try:
            choice = int(input("\nEnter your choice (1-3): "))
            
            if choice == 1:
                # Create an optimized workflow
                workflow = create_optimized_workflow()
                
                # Get a query from the user
                query = input("\nEnter a query for analysis: ")
                
                # Run the workflow
                run_optimized_workflow(workflow, query)
                break
                
            elif choice == 2:
                # Compare workflows
                compare_workflows()
                break
                
            elif choice == 3:
                # Run with different cache configurations
                print("\nRunning with empty cache (first run)...")
                
                # Create an optimized workflow
                workflow = create_optimized_workflow()
                
                # Get a query from the user
                query = input("\nEnter a query for analysis: ")
                
                # First run (empty cache)
                run_optimized_workflow(workflow, query)
                
                print("\nRunning with populated cache (second run)...")
                
                # Second run (populated cache)
                run_optimized_workflow(workflow, query)
                break
                
            else:
                print("Please enter 1, 2, or 3.")
        except ValueError:
            print("Please enter a valid number.")

if __name__ == "__main__":
    main()
