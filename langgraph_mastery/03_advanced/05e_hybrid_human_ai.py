"""
LangGraph Advanced: Hybrid Human-AI Systems
=========================================

This script demonstrates how to build hybrid human-AI systems with 
collaborative decision-making using LangGraph and Google's Generative AI model.
"""

import os
import sys
import time
import json
import uuid
from typing import Dict, List, Literal, TypedDict, Union, Any, Optional, Callable
from enum import Enum
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

# Define decision types
class DecisionType(str, Enum):
    """Types of decisions in the system."""
    SIMPLE = "simple"
    COMPLEX = "complex"
    CRITICAL = "critical"
    CREATIVE = "creative"

# Define confidence levels
class ConfidenceLevel(str, Enum):
    """Confidence levels for AI decisions."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

# Define our state for the hybrid system
class HybridSystemState(TypedDict):
    """State for our hybrid human-AI system."""
    messages: List[Union[HumanMessage, AIMessage, SystemMessage]]
    query: str
    context: Dict[str, Any]
    ai_analysis: str
    ai_recommendation: str
    ai_confidence: ConfidenceLevel
    decision_type: DecisionType
    human_input: Optional[str]
    final_decision: str
    explanation: str
    next: Optional[str]

# Workflow nodes
def query_classification_node(state: HybridSystemState) -> HybridSystemState:
    """Classify the query to determine the decision type and required approach."""
    print("🔍 Classifying query...")
    
    # Extract the query from the state
    query = state["query"]
    
    # Create a classification agent
    model = ChatGoogleGenerativeAI(
        model="gemini-pro",
        temperature=0.3,
        convert_system_message_to_human=True
    )
    
    # Create messages for the classification
    messages = [
        SystemMessage(content="""You are a query classification assistant.
Your job is to analyze a query and classify it based on decision type and complexity.

Decision Types:
- SIMPLE: Straightforward decisions with clear parameters
- COMPLEX: Decisions involving multiple factors and trade-offs
- CRITICAL: High-stakes decisions with significant consequences
- CREATIVE: Decisions requiring innovation and original thinking

For each query, determine:
1. The decision type
2. The key factors to consider
3. Whether human input is likely needed

Return your response in a structured format."""),
        HumanMessage(content=f"Query: {query}")
    ]
    
    # Get the classification
    response = model.invoke(messages)
    
    # Extract the decision type
    decision_type = DecisionType.COMPLEX  # Default
    
    for dt in DecisionType:
        if dt.value.upper() in response.content.upper():
            decision_type = dt
            break
    
    # Update the state
    return {
        **state,
        "context": {"classification_response": response.content},
        "decision_type": decision_type,
        "next": "ai_analysis"
    }

def ai_analysis_node(state: HybridSystemState) -> HybridSystemState:
    """Perform AI analysis of the query and context."""
    print("🧠 Performing AI analysis...")
    
    # Extract the query and decision type from the state
    query = state["query"]
    decision_type = state["decision_type"]
    context = state["context"]
    
    # Create an analysis agent
    model = ChatGoogleGenerativeAI(
        model="gemini-pro",
        temperature=0.5,
        convert_system_message_to_human=True
    )
    
    # Create messages for the analysis
    messages = [
        SystemMessage(content=f"""You are an analysis assistant for {decision_type.value} decisions.
Your job is to analyze a query thoroughly and provide insights.

For this {decision_type.value} decision, please provide:
1. A comprehensive analysis of the situation
2. Key factors to consider
3. Potential options
4. Pros and cons of each option

Be thorough and consider multiple perspectives."""),
        HumanMessage(content=f"Query: {query}")
    ]
    
    # Get the analysis
    response = model.invoke(messages)
    
    # Update the state
    return {
        **state,
        "ai_analysis": response.content,
        "next": "ai_recommendation"
    }

def ai_recommendation_node(state: HybridSystemState) -> HybridSystemState:
    """Generate AI recommendation based on analysis."""
    print("💡 Generating AI recommendation...")
    
    # Extract the query, decision type, and analysis from the state
    query = state["query"]
    decision_type = state["decision_type"]
    ai_analysis = state["ai_analysis"]
    
    # Create a recommendation agent
    model = ChatGoogleGenerativeAI(
        model="gemini-pro",
        temperature=0.4,
        convert_system_message_to_human=True
    )
    
    # Create messages for the recommendation
    messages = [
        SystemMessage(content=f"""You are a recommendation assistant for {decision_type.value} decisions.
Your job is to provide a clear recommendation based on analysis.

Based on the analysis, please provide:
1. A clear recommendation
2. Justification for your recommendation
3. Your confidence level (LOW, MEDIUM, HIGH, or VERY_HIGH)
4. Any caveats or conditions

Be decisive but honest about uncertainty."""),
        HumanMessage(content=f"""
Query: {query}

Analysis:
{ai_analysis}

Please provide your recommendation and confidence level.
""")
    ]
    
    # Get the recommendation
    response = model.invoke(messages)
    
    # Extract the confidence level
    confidence_level = ConfidenceLevel.MEDIUM  # Default
    
    for cl in ConfidenceLevel:
        if cl.value.upper() in response.content.upper():
            confidence_level = cl
            break
    
    # Update the state
    return {
        **state,
        "ai_recommendation": response.content,
        "ai_confidence": confidence_level,
        "next": "decision_routing"
    }

def decision_routing_node(state: HybridSystemState) -> HybridSystemState:
    """Route the decision to either AI or human based on decision type and confidence."""
    print("🔀 Routing decision...")
    
    # Extract the decision type and confidence from the state
    decision_type = state["decision_type"]
    confidence = state["ai_confidence"]
    
    # Decision routing logic
    if decision_type == DecisionType.SIMPLE and confidence in [ConfidenceLevel.HIGH, ConfidenceLevel.VERY_HIGH]:
        # Simple decisions with high confidence can be made by AI
        print("  Routing to AI decision (simple decision with high confidence)")
        return {
            **state,
            "next": "ai_decision"
        }
    elif decision_type == DecisionType.CRITICAL:
        # Critical decisions always require human input
        print("  Routing to human decision (critical decision)")
        return {
            **state,
            "next": "human_decision"
        }
    elif confidence == ConfidenceLevel.LOW:
        # Low confidence decisions require human input
        print("  Routing to human decision (low confidence)")
        return {
            **state,
            "next": "human_decision"
        }
    elif decision_type == DecisionType.CREATIVE and confidence != ConfidenceLevel.VERY_HIGH:
        # Creative decisions usually benefit from human input
        print("  Routing to human decision (creative decision)")
        return {
            **state,
            "next": "human_decision"
        }
    else:
        # Default to AI for other cases
        print("  Routing to AI decision (default case)")
        return {
            **state,
            "next": "ai_decision"
        }

def ai_decision_node(state: HybridSystemState) -> HybridSystemState:
    """Make a decision using AI only."""
    print("🤖 Making AI decision...")
    
    # Extract the query, analysis, and recommendation from the state
    query = state["query"]
    ai_analysis = state["ai_analysis"]
    ai_recommendation = state["ai_recommendation"]
    
    # Create a decision agent
    model = ChatGoogleGenerativeAI(
        model="gemini-pro",
        temperature=0.3,
        convert_system_message_to_human=True
    )
    
    # Create messages for the decision
    messages = [
        SystemMessage(content="""You are a decision-making assistant.
Your job is to make a final decision based on analysis and recommendations.

Please provide:
1. A clear, decisive final decision
2. A concise explanation of the reasoning
3. Any implementation considerations

Be authoritative but acknowledge the basis of your decision."""),
        HumanMessage(content=f"""
Query: {query}

Analysis:
{ai_analysis}

Recommendation:
{ai_recommendation}

Please make a final decision.
""")
    ]
    
    # Get the decision
    response = model.invoke(messages)
    
    # Split the response into decision and explanation
    lines = response.content.split("\n")
    final_decision = lines[0] if lines else response.content
    explanation = "\n".join(lines[1:]) if len(lines) > 1 else "Based on the analysis and recommendation."
    
    # Update the state
    return {
        **state,
        "final_decision": final_decision,
        "explanation": explanation,
        "next": "decision_implementation"
    }

def human_decision_node(state: HybridSystemState) -> HybridSystemState:
    """Collect human input for the decision."""
    print("\n" + "=" * 50)
    print("Human Decision Required")
    print("=" * 50)
    
    # Extract the query, analysis, and recommendation from the state
    query = state["query"]
    decision_type = state["decision_type"]
    ai_analysis = state["ai_analysis"]
    ai_recommendation = state["ai_recommendation"]
    
    # Display the information
    print(f"\nQuery: {query}")
    print(f"\nDecision Type: {decision_type.value.upper()}")
    
    print("\nAI Analysis:")
    print("-" * 50)
    print(ai_analysis)
    
    print("\nAI Recommendation:")
    print("-" * 50)
    print(ai_recommendation)
    
    # Collect human input
    print("\nPlease provide your decision:")
    human_input = input("> ")
    
    # Create an explanation agent
    model = ChatGoogleGenerativeAI(
        model="gemini-pro",
        temperature=0.4,
        convert_system_message_to_human=True
    )
    
    # Create messages for the explanation
    messages = [
        SystemMessage(content="""You are a decision explanation assistant.
Your job is to create a clear explanation that combines AI analysis with human decision.

Please provide a coherent explanation that:
1. Acknowledges the AI analysis and recommendation
2. Incorporates the human decision
3. Provides a unified rationale

Be concise but comprehensive."""),
        HumanMessage(content=f"""
Query: {query}

AI Analysis:
{ai_analysis}

AI Recommendation:
{ai_recommendation}

Human Decision:
{human_input}

Please create an explanation for this decision.
""")
    ]
    
    # Get the explanation
    response = model.invoke(messages)
    
    # Update the state
    return {
        **state,
        "human_input": human_input,
        "final_decision": human_input,
        "explanation": response.content,
        "next": "decision_implementation"
    }

def decision_implementation_node(state: HybridSystemState) -> HybridSystemState:
    """Implement the decision and provide next steps."""
    print("📋 Implementing decision...")
    
    # Extract the query, final decision, and explanation from the state
    query = state["query"]
    final_decision = state["final_decision"]
    explanation = state["explanation"]
    decision_type = state["decision_type"]
    human_input = state["human_input"]
    
    # Create an implementation agent
    model = ChatGoogleGenerativeAI(
        model="gemini-pro",
        temperature=0.5,
        convert_system_message_to_human=True
    )
    
    # Create messages for the implementation
    messages = [
        SystemMessage(content="""You are a decision implementation assistant.
Your job is to provide clear implementation steps for a decision.

Please provide:
1. Concrete next steps to implement the decision
2. Potential challenges to anticipate
3. Metrics to track for success
4. Timeline considerations

Be practical and actionable."""),
        HumanMessage(content=f"""
Query: {query}

Final Decision: {final_decision}

Explanation: {explanation}

Decision Type: {decision_type.value}

Human Input: {"None" if human_input is None else human_input}

Please provide implementation steps for this decision.
""")
    ]
    
    # Get the implementation
    response = model.invoke(messages)
    
    # Add the implementation to the messages
    new_messages = state["messages"] + [
        AIMessage(content=f"""
# Decision: {final_decision}

## Explanation
{explanation}

## Implementation Plan
{response.content}
""")
    ]
    
    # Update the state
    return {
        **state,
        "messages": new_messages,
        "next": "end"
    }

def create_hybrid_system():
    """Create a hybrid human-AI system using LangGraph."""
    print("Creating a hybrid human-AI system with LangGraph...")
    
    # Create a new graph
    workflow = StateGraph(HybridSystemState)
    
    # Add nodes to the graph
    workflow.add_node("query_classification", query_classification_node)
    workflow.add_node("ai_analysis", ai_analysis_node)
    workflow.add_node("ai_recommendation", ai_recommendation_node)
    workflow.add_node("decision_routing", decision_routing_node)
    workflow.add_node("ai_decision", ai_decision_node)
    workflow.add_node("human_decision", human_decision_node)
    workflow.add_node("decision_implementation", decision_implementation_node)
    
    # Add edges
    workflow.add_conditional_edges(
        "query_classification",
        lambda state: state["next"],
        {
            "ai_analysis": "ai_analysis"
        }
    )
    
    workflow.add_conditional_edges(
        "ai_analysis",
        lambda state: state["next"],
        {
            "ai_recommendation": "ai_recommendation"
        }
    )
    
    workflow.add_conditional_edges(
        "ai_recommendation",
        lambda state: state["next"],
        {
            "decision_routing": "decision_routing"
        }
    )
    
    workflow.add_conditional_edges(
        "decision_routing",
        lambda state: state["next"],
        {
            "ai_decision": "ai_decision",
            "human_decision": "human_decision"
        }
    )
    
    workflow.add_conditional_edges(
        "ai_decision",
        lambda state: state["next"],
        {
            "decision_implementation": "decision_implementation"
        }
    )
    
    workflow.add_conditional_edges(
        "human_decision",
        lambda state: state["next"],
        {
            "decision_implementation": "decision_implementation"
        }
    )
    
    workflow.add_conditional_edges(
        "decision_implementation",
        lambda state: state["next"],
        {
            "end": END
        }
    )
    
    # Set the entry point
    workflow.set_entry_point("query_classification")
    
    # Compile the graph
    return workflow.compile()

def run_hybrid_system(workflow, query: str):
    """Run the hybrid human-AI system with a given query."""
    print("\n" + "=" * 50)
    print("Running the Hybrid Human-AI System")
    print("=" * 50)
    
    # Initialize the state
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "query": query,
        "context": {},
        "ai_analysis": "",
        "ai_recommendation": "",
        "ai_confidence": ConfidenceLevel.MEDIUM,
        "decision_type": DecisionType.COMPLEX,
        "human_input": None,
        "final_decision": "",
        "explanation": "",
        "next": None
    }
    
    # Run the workflow
    print(f"\nQuery: {query}")
    result = workflow.invoke(initial_state)
    
    # Print the workflow results
    print("\n" + "=" * 50)
    print("Hybrid Human-AI System Results")
    print("=" * 50)
    
    print("\nDecision Type:", result["decision_type"])
    print("AI Confidence:", result["ai_confidence"])
    print("Human Input:", "None" if result["human_input"] is None else "Provided")
    
    print("\nFinal Decision:")
    print("-" * 50)
    print(result["final_decision"])
    
    print("\nExplanation:")
    print("-" * 50)
    print(result["explanation"])
    
    # Print the last message which contains the implementation plan
    if result["messages"] and isinstance(result["messages"][-1], AIMessage):
        print("\nImplementation Plan:")
        print("-" * 50)
        print(result["messages"][-1].content)
    
    return result

def main():
    """Main function to demonstrate hybrid human-AI systems."""
    print("=" * 50)
    print("Hybrid Human-AI Systems in LangGraph")
    print("=" * 50)
    
    # Let the user choose a decision scenario
    print("\nSelect a decision scenario or enter your own:")
    print("1. Should we migrate our database from MySQL to MongoDB?")
    print("2. What marketing strategy should we use for our new product launch?")
    print("3. How should we allocate our $50,000 technology budget for next quarter?")
    print("4. What creative direction should we take for our company rebrand?")
    print("5. Enter your own decision query")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-5): ")
            
            if choice == "1":
                query = "Should we migrate our database from MySQL to MongoDB?"
                break
            elif choice == "2":
                query = "What marketing strategy should we use for our new product launch?"
                break
            elif choice == "3":
                query = "How should we allocate our $50,000 technology budget for next quarter?"
                break
            elif choice == "4":
                query = "What creative direction should we take for our company rebrand?"
                break
            elif choice == "5":
                query = input("\nEnter your decision query: ")
                break
            else:
                print("Please enter a number between 1 and 5.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Create and run the hybrid system
    workflow = create_hybrid_system()
    run_hybrid_system(workflow, query)

if __name__ == "__main__":
    main()
