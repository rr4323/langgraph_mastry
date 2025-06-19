"""
LangGraph Advanced: Dynamic Graph Construction
===========================================

This script demonstrates how to dynamically construct and modify LangGraph
workflows at runtime based on context and requirements.
"""

import os
import sys
import time
from typing import Dict, List, Literal, TypedDict, Union, Any, Optional, Callable
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END

# Add the parent directory to the path so we can import from the root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables from .env file
load_dotenv()

# Define our state for the dynamic graph system
class DynamicGraphState(TypedDict):
    """State for our dynamic graph system."""
    messages: List[Union[HumanMessage, AIMessage, SystemMessage]]
    query: str
    context: Dict[str, Any]
    workflow_plan: Dict[str, Any]
    current_node: str
    results: Dict[str, Any]
    next: Optional[str]

# Node registry to store available node functions
NODE_REGISTRY = {}

def register_node(name: str):
    """Decorator to register a node function."""
    def decorator(func):
        NODE_REGISTRY[name] = func
        return func
    return decorator

@register_node("query_analysis")
def query_analysis_node(state: DynamicGraphState) -> DynamicGraphState:
    """Analyze the query and determine the workflow plan."""
    print("🔍 Analyzing query...")
    
    # Extract the query from the state
    query = state["query"]
    
    # Create an analysis agent
    model = ChatGoogleGenerativeAI(
        model="gemini-pro",
        temperature=0.3,
        convert_system_message_to_human=True
    )
    
    # Create messages for the analysis
    messages = [
        SystemMessage(content="""You are a workflow planning assistant.
Your job is to analyze a query and determine the optimal workflow to process it.

Available workflow nodes:
- data_retrieval: Retrieves relevant data for the query
- calculation: Performs calculations or data processing
- research: Conducts research on a topic
- summarization: Summarizes information
- creative_generation: Generates creative content
- fact_checking: Verifies factual accuracy
- personalization: Personalizes content for the user
- formatting: Formats the final output

For each node, specify:
1. Whether it should be included (true/false)
2. Its position in the workflow (order number)
3. Any specific parameters it needs

Return your response as a JSON object with the workflow plan."""),
        HumanMessage(content=f"Query: {query}")
    ]
    
    # Get the analysis
    response = model.invoke(messages)
    
    # Extract the workflow plan
    import json
    import re
    
    # Try to find a JSON object in the response
    json_match = re.search(r'```json\n(.*?)\n```', response.content, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        # Try to find anything that looks like JSON
        json_match = re.search(r'({.*})', response.content, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = response.content
    
    try:
        workflow_plan = json.loads(json_str)
    except json.JSONDecodeError:
        # If JSON parsing fails, create a basic workflow plan
        workflow_plan = {
            "nodes": {
                "data_retrieval": {"include": True, "order": 1, "parameters": {}},
                "summarization": {"include": True, "order": 2, "parameters": {}},
                "formatting": {"include": True, "order": 3, "parameters": {}}
            }
        }
    
    # Update the state
    return {
        **state,
        "workflow_plan": workflow_plan,
        "next": "workflow_construction"
    }

@register_node("workflow_construction")
def workflow_construction_node(state: DynamicGraphState) -> DynamicGraphState:
    """Construct the workflow based on the plan."""
    print("🔧 Constructing workflow...")
    
    # The actual workflow construction happens in create_dynamic_graph
    # This node just passes control to the first node in the workflow
    
    # Get the workflow plan
    workflow_plan = state["workflow_plan"]
    
    # Find the first node in the workflow
    first_node = None
    first_order = float('inf')
    
    for node_name, node_config in workflow_plan.get("nodes", {}).items():
        if node_config.get("include", False) and node_config.get("order", 0) < first_order:
            first_node = node_name
            first_order = node_config.get("order", 0)
    
    # If no nodes are included, end the workflow
    if first_node is None:
        return {
            **state,
            "next": "end"
        }
    
    # Update the state
    return {
        **state,
        "current_node": first_node,
        "next": first_node
    }

@register_node("data_retrieval")
def data_retrieval_node(state: DynamicGraphState) -> DynamicGraphState:
    """Retrieve relevant data for the query."""
    print("📊 Retrieving data...")
    
    # Extract the query and parameters from the state
    query = state["query"]
    parameters = state["workflow_plan"]["nodes"].get("data_retrieval", {}).get("parameters", {})
    
    # Create a data retrieval agent
    model = ChatGoogleGenerativeAI(
        model="gemini-pro",
        temperature=0.2,
        convert_system_message_to_human=True
    )
    
    # Create messages for the data retrieval
    messages = [
        SystemMessage(content="""You are a data retrieval assistant.
Your job is to retrieve relevant data for a query.
Since this is a simulation, please generate realistic-looking data that would be relevant."""),
        HumanMessage(content=f"""
Query: {query}
Parameters: {parameters}

Please retrieve relevant data for this query.
""")
    ]
    
    # Get the data
    response = model.invoke(messages)
    
    # Update the results
    results = state["results"]
    results["data_retrieval"] = response.content
    
    # Find the next node
    next_node = find_next_node(state["workflow_plan"], "data_retrieval")
    
    # Update the state
    return {
        **state,
        "results": results,
        "current_node": "data_retrieval",
        "next": next_node if next_node else "end"
    }

@register_node("calculation")
def calculation_node(state: DynamicGraphState) -> DynamicGraphState:
    """Perform calculations or data processing."""
    print("🧮 Performing calculations...")
    
    # Extract the query and parameters from the state
    query = state["query"]
    parameters = state["workflow_plan"]["nodes"].get("calculation", {}).get("parameters", {})
    
    # Get previous results that might be needed
    previous_results = state["results"]
    
    # Create a calculation agent
    model = ChatGoogleGenerativeAI(
        model="gemini-pro",
        temperature=0.1,
        convert_system_message_to_human=True
    )
    
    # Create messages for the calculation
    messages = [
        SystemMessage(content="""You are a calculation assistant.
Your job is to perform calculations or data processing based on a query and previous results."""),
        HumanMessage(content=f"""
Query: {query}
Parameters: {parameters}
Previous Results: {previous_results}

Please perform the necessary calculations or data processing.
""")
    ]
    
    # Get the calculations
    response = model.invoke(messages)
    
    # Update the results
    results = state["results"]
    results["calculation"] = response.content
    
    # Find the next node
    next_node = find_next_node(state["workflow_plan"], "calculation")
    
    # Update the state
    return {
        **state,
        "results": results,
        "current_node": "calculation",
        "next": next_node if next_node else "end"
    }

@register_node("research")
def research_node(state: DynamicGraphState) -> DynamicGraphState:
    """Conduct research on a topic."""
    print("🔬 Conducting research...")
    
    # Extract the query and parameters from the state
    query = state["query"]
    parameters = state["workflow_plan"]["nodes"].get("research", {}).get("parameters", {})
    
    # Get previous results that might be needed
    previous_results = state["results"]
    
    # Create a research agent
    model = ChatGoogleGenerativeAI(
        model="gemini-pro",
        temperature=0.6,
        convert_system_message_to_human=True
    )
    
    # Create messages for the research
    messages = [
        SystemMessage(content="""You are a research assistant.
Your job is to conduct research on a topic based on a query and previous results."""),
        HumanMessage(content=f"""
Query: {query}
Parameters: {parameters}
Previous Results: {previous_results}

Please conduct research on this topic and provide your findings.
""")
    ]
    
    # Get the research
    response = model.invoke(messages)
    
    # Update the results
    results = state["results"]
    results["research"] = response.content
    
    # Find the next node
    next_node = find_next_node(state["workflow_plan"], "research")
    
    # Update the state
    return {
        **state,
        "results": results,
        "current_node": "research",
        "next": next_node if next_node else "end"
    }

@register_node("summarization")
def summarization_node(state: DynamicGraphState) -> DynamicGraphState:
    """Summarize information."""
    print("📝 Summarizing information...")
    
    # Extract the query and parameters from the state
    query = state["query"]
    parameters = state["workflow_plan"]["nodes"].get("summarization", {}).get("parameters", {})
    
    # Get previous results that might be needed
    previous_results = state["results"]
    
    # Create a summarization agent
    model = ChatGoogleGenerativeAI(
        model="gemini-pro",
        temperature=0.3,
        convert_system_message_to_human=True
    )
    
    # Create messages for the summarization
    messages = [
        SystemMessage(content="""You are a summarization assistant.
Your job is to summarize information based on a query and previous results."""),
        HumanMessage(content=f"""
Query: {query}
Parameters: {parameters}
Previous Results: {previous_results}

Please summarize this information concisely.
""")
    ]
    
    # Get the summary
    response = model.invoke(messages)
    
    # Update the results
    results = state["results"]
    results["summarization"] = response.content
    
    # Find the next node
    next_node = find_next_node(state["workflow_plan"], "summarization")
    
    # Update the state
    return {
        **state,
        "results": results,
        "current_node": "summarization",
        "next": next_node if next_node else "end"
    }

@register_node("creative_generation")
def creative_generation_node(state: DynamicGraphState) -> DynamicGraphState:
    """Generate creative content."""
    print("🎨 Generating creative content...")
    
    # Extract the query and parameters from the state
    query = state["query"]
    parameters = state["workflow_plan"]["nodes"].get("creative_generation", {}).get("parameters", {})
    
    # Get previous results that might be needed
    previous_results = state["results"]
    
    # Create a creative generation agent
    model = ChatGoogleGenerativeAI(
        model="gemini-pro",
        temperature=0.9,
        convert_system_message_to_human=True
    )
    
    # Create messages for the creative generation
    messages = [
        SystemMessage(content="""You are a creative content generator.
Your job is to generate creative content based on a query and previous results."""),
        HumanMessage(content=f"""
Query: {query}
Parameters: {parameters}
Previous Results: {previous_results}

Please generate creative content based on this information.
""")
    ]
    
    # Get the creative content
    response = model.invoke(messages)
    
    # Update the results
    results = state["results"]
    results["creative_generation"] = response.content
    
    # Find the next node
    next_node = find_next_node(state["workflow_plan"], "creative_generation")
    
    # Update the state
    return {
        **state,
        "results": results,
        "current_node": "creative_generation",
        "next": next_node if next_node else "end"
    }

@register_node("fact_checking")
def fact_checking_node(state: DynamicGraphState) -> DynamicGraphState:
    """Verify factual accuracy."""
    print("✓ Checking facts...")
    
    # Extract the query and parameters from the state
    query = state["query"]
    parameters = state["workflow_plan"]["nodes"].get("fact_checking", {}).get("parameters", {})
    
    # Get previous results that might be needed
    previous_results = state["results"]
    
    # Create a fact checking agent
    model = ChatGoogleGenerativeAI(
        model="gemini-pro",
        temperature=0.1,
        convert_system_message_to_human=True
    )
    
    # Create messages for the fact checking
    messages = [
        SystemMessage(content="""You are a fact checking assistant.
Your job is to verify the factual accuracy of information based on a query and previous results."""),
        HumanMessage(content=f"""
Query: {query}
Parameters: {parameters}
Previous Results: {previous_results}

Please verify the factual accuracy of this information and provide corrections if needed.
""")
    ]
    
    # Get the fact checking
    response = model.invoke(messages)
    
    # Update the results
    results = state["results"]
    results["fact_checking"] = response.content
    
    # Find the next node
    next_node = find_next_node(state["workflow_plan"], "fact_checking")
    
    # Update the state
    return {
        **state,
        "results": results,
        "current_node": "fact_checking",
        "next": next_node if next_node else "end"
    }

@register_node("personalization")
def personalization_node(state: DynamicGraphState) -> DynamicGraphState:
    """Personalize content for the user."""
    print("👤 Personalizing content...")
    
    # Extract the query and parameters from the state
    query = state["query"]
    parameters = state["workflow_plan"]["nodes"].get("personalization", {}).get("parameters", {})
    
    # Get previous results that might be needed
    previous_results = state["results"]
    
    # Create a personalization agent
    model = ChatGoogleGenerativeAI(
        model="gemini-pro",
        temperature=0.7,
        convert_system_message_to_human=True
    )
    
    # Create messages for the personalization
    messages = [
        SystemMessage(content="""You are a personalization assistant.
Your job is to personalize content for the user based on a query and previous results."""),
        HumanMessage(content=f"""
Query: {query}
Parameters: {parameters}
Previous Results: {previous_results}

Please personalize this content for the user.
""")
    ]
    
    # Get the personalized content
    response = model.invoke(messages)
    
    # Update the results
    results = state["results"]
    results["personalization"] = response.content
    
    # Find the next node
    next_node = find_next_node(state["workflow_plan"], "personalization")
    
    # Update the state
    return {
        **state,
        "results": results,
        "current_node": "personalization",
        "next": next_node if next_node else "end"
    }

@register_node("formatting")
def formatting_node(state: DynamicGraphState) -> DynamicGraphState:
    """Format the final output."""
    print("📄 Formatting output...")
    
    # Extract the query and parameters from the state
    query = state["query"]
    parameters = state["workflow_plan"]["nodes"].get("formatting", {}).get("parameters", {})
    
    # Get previous results that might be needed
    previous_results = state["results"]
    
    # Create a formatting agent
    model = ChatGoogleGenerativeAI(
        model="gemini-pro",
        temperature=0.4,
        convert_system_message_to_human=True
    )
    
    # Create messages for the formatting
    messages = [
        SystemMessage(content="""You are a formatting assistant.
Your job is to format the final output based on a query and previous results."""),
        HumanMessage(content=f"""
Query: {query}
Parameters: {parameters}
Previous Results: {previous_results}

Please format the final output in a clear, well-structured way.
""")
    ]
    
    # Get the formatted output
    response = model.invoke(messages)
    
    # Update the results
    results = state["results"]
    results["formatting"] = response.content
    
    # Find the next node
    next_node = find_next_node(state["workflow_plan"], "formatting")
    
    # Update the state
    return {
        **state,
        "results": results,
        "current_node": "formatting",
        "next": next_node if next_node else "end"
    }

def find_next_node(workflow_plan: Dict[str, Any], current_node: str) -> Optional[str]:
    """Find the next node in the workflow based on the order."""
    current_order = workflow_plan["nodes"].get(current_node, {}).get("order", 0)
    
    next_node = None
    next_order = float('inf')
    
    for node_name, node_config in workflow_plan.get("nodes", {}).items():
        if node_config.get("include", False) and node_config.get("order", 0) > current_order and node_config.get("order", 0) < next_order:
            next_node = node_name
            next_order = node_config.get("order", 0)
    
    return next_node

def create_dynamic_graph(query: str) -> StateGraph:
    """Create a dynamic graph based on the query."""
    print(f"Creating a dynamic graph for query: {query}")
    
    # Create a new graph
    workflow = StateGraph(DynamicGraphState)
    
    # Add the initial nodes
    workflow.add_node("query_analysis", query_analysis_node)
    workflow.add_node("workflow_construction", workflow_construction_node)
    
    # Add the task nodes from the registry
    for node_name, node_func in NODE_REGISTRY.items():
        if node_name not in ["query_analysis", "workflow_construction"]:
            workflow.add_node(node_name, node_func)
    
    # Add edges for the initial nodes
    workflow.add_conditional_edges(
        "query_analysis",
        lambda state: state["next"],
        {
            "workflow_construction": "workflow_construction"
        }
    )
    
    workflow.add_conditional_edges(
        "workflow_construction",
        lambda state: state["next"],
        {node_name: node_name for node_name in NODE_REGISTRY.keys() if node_name not in ["query_analysis", "workflow_construction"]}
    )
    
    # Add edges for the task nodes
    for node_name in NODE_REGISTRY.keys():
        if node_name not in ["query_analysis", "workflow_construction"]:
            edges = {other_node: other_node for other_node in NODE_REGISTRY.keys() if other_node not in ["query_analysis", "workflow_construction"]}
            edges["end"] = END
            
            workflow.add_conditional_edges(
                node_name,
                lambda state: state["next"],
                edges
            )
    
    # Set the entry point
    workflow.set_entry_point("query_analysis")
    
    # Compile the graph
    return workflow.compile()

def run_dynamic_graph(workflow, query: str):
    """Run the dynamic graph with a given query."""
    print("\n" + "=" * 50)
    print("Running the Dynamic Graph")
    print("=" * 50)
    
    # Initialize the state
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "query": query,
        "context": {},
        "workflow_plan": {},
        "current_node": "",
        "results": {},
        "next": None
    }
    
    # Run the workflow
    print(f"\nQuery: {query}")
    result = workflow.invoke(initial_state)
    
    # Print the workflow results
    print("\n" + "=" * 50)
    print("Dynamic Graph Results")
    print("=" * 50)
    
    print("\nWorkflow Plan:")
    print("-" * 50)
    import json
    print(json.dumps(result["workflow_plan"], indent=2))
    
    print("\nResults:")
    print("-" * 50)
    
    # Print the final result (from formatting if available, otherwise the last result)
    if "formatting" in result["results"]:
        print(result["results"]["formatting"])
    else:
        # Find the last node that was executed
        last_node = None
        last_order = -1
        
        for node_name, node_config in result["workflow_plan"].get("nodes", {}).items():
            if node_config.get("include", False) and node_name in result["results"] and node_config.get("order", 0) > last_order:
                last_node = node_name
                last_order = node_config.get("order", 0)
        
        if last_node and last_node in result["results"]:
            print(result["results"][last_node])
        else:
            print("No results available.")
    
    return result

def main():
    """Main function to demonstrate dynamic graph construction."""
    print("=" * 50)
    print("Dynamic Graph Construction in LangGraph")
    print("=" * 50)
    
    # Get a query from the user
    print("\nEnter a query to process with a dynamically constructed graph.")
    print("Example queries:")
    print("1. What is the current state of renewable energy adoption worldwide?")
    print("2. Write a creative short story about a time traveler.")
    print("3. Calculate the compound interest on a $10,000 investment at 5% for 10 years.")
    print("4. Summarize the key features of quantum computing.")
    
    query = input("\nEnter your query: ")
    
    # Create and run the dynamic graph
    workflow = create_dynamic_graph(query)
    run_dynamic_graph(workflow, query)

if __name__ == "__main__":
    main()
