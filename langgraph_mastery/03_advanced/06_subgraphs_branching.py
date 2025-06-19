"""
Subgraphs and Branching in LangGraph

This script demonstrates how to create modular subgraphs and implement
complex branching logic in LangGraph for advanced workflow patterns.
"""

import os
import sys
import logging
from typing import Dict, List, Any, Optional, Literal, Union, Annotated
from enum import Enum
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Check for required packages
required_packages = ["langchain-google-genai", "langgraph"]
for package in required_packages:
    try:
        __import__(package.replace("-", "_"))
    except ImportError:
        logger.info(f"Installing {package}...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.graph.graph import CompiledGraph

# Define our state models
class QueryCategory(str, Enum):
    """Categories for user queries."""
    FACTUAL = "factual"
    ANALYTICAL = "analytical"
    CREATIVE = "creative"
    TECHNICAL = "technical"
    UNKNOWN = "unknown"

class QueryComplexity(str, Enum):
    """Complexity levels for queries."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    UNKNOWN = "unknown"

class QueryClassification(BaseModel):
    """Classification of a user query."""
    category: QueryCategory = QueryCategory.UNKNOWN
    complexity: QueryComplexity = QueryComplexity.UNKNOWN
    requires_research: bool = False
    requires_reasoning: bool = False
    requires_code: bool = False
    confidence: float = 0.0

class Message(BaseModel):
    """A message in the conversation."""
    role: str
    content: str

class WorkflowState(BaseModel):
    """The state for our workflow with subgraphs."""
    query: str
    messages: List[Message] = Field(default_factory=list)
    classification: Optional[QueryClassification] = None
    research_results: Optional[str] = None
    analysis_results: Optional[str] = None
    code_snippets: List[str] = Field(default_factory=list)
    final_response: Optional[str] = None
    errors: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    current_subgraph: Optional[str] = None
    subgraph_states: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

# Helper functions
def get_model(google_api_key: str, temperature: float = 0.4):
    """Get the language model."""
    return ChatGoogleGenerativeAI(
        model="gemini-pro",
        google_api_key=google_api_key,
        temperature=temperature,
        convert_system_message_to_human=True
    )

# Subgraph 1: Query Classification
async def classify_query(state: Dict[str, Any], model) -> Dict[str, Any]:
    """Classify the user query."""
    logger.info("Classifying query")
    state_obj = WorkflowState.model_validate(state)
    
    try:
        # Prepare the messages for the model
        messages = [
            {"role": "system", "content": """You are a query classification system. 
Analyze the user query and classify it according to the following criteria:
1. Category: factual, analytical, creative, or technical
2. Complexity: simple, moderate, or complex
3. Requirements: Does it require research, reasoning, and/or code?

Provide your classification in a structured format."""},
            {"role": "user", "content": f"Classify this query: {state_obj.query}"}
        ]
        
        # Invoke the model
        response = await model.ainvoke(messages)
        classification_text = response.content
        
        # Parse the classification
        category = QueryCategory.UNKNOWN
        complexity = QueryComplexity.UNKNOWN
        requires_research = False
        requires_reasoning = False
        requires_code = False
        confidence = 0.5
        
        # Simple parsing logic - in a real system, you'd use more robust parsing
        if "category" in classification_text.lower():
            if "factual" in classification_text.lower():
                category = QueryCategory.FACTUAL
            elif "analytical" in classification_text.lower():
                category = QueryCategory.ANALYTICAL
            elif "creative" in classification_text.lower():
                category = QueryCategory.CREATIVE
            elif "technical" in classification_text.lower():
                category = QueryCategory.TECHNICAL
        
        if "complexity" in classification_text.lower():
            if "simple" in classification_text.lower():
                complexity = QueryComplexity.SIMPLE
            elif "moderate" in classification_text.lower():
                complexity = QueryComplexity.MODERATE
            elif "complex" in classification_text.lower():
                complexity = QueryComplexity.COMPLEX
        
        if "research" in classification_text.lower() and "yes" in classification_text.lower():
            requires_research = True
        
        if "reasoning" in classification_text.lower() and "yes" in classification_text.lower():
            requires_reasoning = True
        
        if "code" in classification_text.lower() and "yes" in classification_text.lower():
            requires_code = True
        
        if "confidence" in classification_text.lower():
            # Try to extract a confidence value
            try:
                confidence_text = classification_text.lower().split("confidence")[1].split("\n")[0]
                confidence_value = float(''.join(filter(lambda x: x.isdigit() or x == '.', confidence_text)))
                if 0 <= confidence_value <= 1:
                    confidence = confidence_value
                elif 0 <= confidence_value <= 100:
                    confidence = confidence_value / 100
            except:
                pass
        
        # Create the classification object
        classification = QueryClassification(
            category=category,
            complexity=complexity,
            requires_research=requires_research,
            requires_reasoning=requires_reasoning,
            requires_code=requires_code,
            confidence=confidence
        )
        
        # Update the state
        state_obj.classification = classification
        state_obj.messages.append(Message(
            role="system",
            content=f"Query classified as {category.value}, {complexity.value} complexity."
        ))
        
    except Exception as e:
        logger.error(f"Error in classify_query: {str(e)}", exc_info=True)
        state_obj.errors.append({
            "step": "classify_query",
            "error": str(e)
        })
    
    return state_obj.model_dump()

# Subgraph 2: Research
async def research_query(state: Dict[str, Any], model) -> Dict[str, Any]:
    """Research information related to the query."""
    logger.info("Researching query")
    state_obj = WorkflowState.model_validate(state)
    
    try:
        # Save the current state to the subgraph states
        state_obj.subgraph_states["research"] = {
            "input_query": state_obj.query,
            "classification": state_obj.classification.model_dump() if state_obj.classification else None
        }
        
        # Prepare the messages for the model
        messages = [
            {"role": "system", "content": "You are a research assistant. Find relevant information for the given query."},
            {"role": "user", "content": f"Research this query: {state_obj.query}"}
        ]
        
        # Invoke the model
        response = await model.ainvoke(messages)
        research_results = response.content
        
        # Update the state
        state_obj.research_results = research_results
        state_obj.messages.append(Message(
            role="researcher",
            content=f"Research completed: {research_results[:100]}..."
        ))
        
        # Update the subgraph state
        state_obj.subgraph_states["research"]["output"] = {
            "research_results": research_results
        }
        
    except Exception as e:
        logger.error(f"Error in research_query: {str(e)}", exc_info=True)
        state_obj.errors.append({
            "step": "research_query",
            "error": str(e)
        })
    
    return state_obj.model_dump()

# Subgraph 3: Analysis
async def analyze_query(state: Dict[str, Any], model) -> Dict[str, Any]:
    """Analyze the query and research results."""
    logger.info("Analyzing query")
    state_obj = WorkflowState.model_validate(state)
    
    try:
        # Save the current state to the subgraph states
        state_obj.subgraph_states["analysis"] = {
            "input_query": state_obj.query,
            "input_research": state_obj.research_results
        }
        
        # Prepare the messages for the model
        messages = [
            {"role": "system", "content": "You are an analyst. Analyze the query and research information to provide insights."},
            {"role": "user", "content": f"Query: {state_obj.query}\n\nResearch: {state_obj.research_results}"}
        ]
        
        # Invoke the model
        response = await model.ainvoke(messages)
        analysis_results = response.content
        
        # Update the state
        state_obj.analysis_results = analysis_results
        state_obj.messages.append(Message(
            role="analyst",
            content=f"Analysis completed: {analysis_results[:100]}..."
        ))
        
        # Update the subgraph state
        state_obj.subgraph_states["analysis"]["output"] = {
            "analysis_results": analysis_results
        }
        
    except Exception as e:
        logger.error(f"Error in analyze_query: {str(e)}", exc_info=True)
        state_obj.errors.append({
            "step": "analyze_query",
            "error": str(e)
        })
    
    return state_obj.model_dump()

# Subgraph 4: Code Generation
async def generate_code(state: Dict[str, Any], model) -> Dict[str, Any]:
    """Generate code snippets for the query."""
    logger.info("Generating code")
    state_obj = WorkflowState.model_validate(state)
    
    try:
        # Save the current state to the subgraph states
        state_obj.subgraph_states["code_generation"] = {
            "input_query": state_obj.query,
            "input_analysis": state_obj.analysis_results
        }
        
        # Prepare the messages for the model
        messages = [
            {"role": "system", "content": "You are a code generation assistant. Create code snippets that address the query."},
            {"role": "user", "content": f"Query: {state_obj.query}\n\nAnalysis: {state_obj.analysis_results}"}
        ]
        
        # Invoke the model
        response = await model.ainvoke(messages)
        code_snippet = response.content
        
        # Update the state
        state_obj.code_snippets.append(code_snippet)
        state_obj.messages.append(Message(
            role="coder",
            content=f"Code generated: {code_snippet[:100]}..."
        ))
        
        # Update the subgraph state
        state_obj.subgraph_states["code_generation"]["output"] = {
            "code_snippets": state_obj.code_snippets
        }
        
    except Exception as e:
        logger.error(f"Error in generate_code: {str(e)}", exc_info=True)
        state_obj.errors.append({
            "step": "generate_code",
            "error": str(e)
        })
    
    return state_obj.model_dump()

# Subgraph 5: Response Generation
async def generate_response(state: Dict[str, Any], model) -> Dict[str, Any]:
    """Generate the final response."""
    logger.info("Generating response")
    state_obj = WorkflowState.model_validate(state)
    
    try:
        # Prepare the context based on what we have
        context = f"Query: {state_obj.query}\n\n"
        
        if state_obj.research_results:
            context += f"Research: {state_obj.research_results}\n\n"
        
        if state_obj.analysis_results:
            context += f"Analysis: {state_obj.analysis_results}\n\n"
        
        if state_obj.code_snippets:
            context += "Code Snippets:\n"
            for i, snippet in enumerate(state_obj.code_snippets, 1):
                context += f"Snippet {i}:\n{snippet}\n\n"
        
        # Prepare the messages for the model
        messages = [
            {"role": "system", "content": "You are a response generator. Create a comprehensive response that addresses the query using the provided context."},
            {"role": "user", "content": context}
        ]
        
        # Invoke the model
        response = await model.ainvoke(messages)
        final_response = response.content
        
        # Update the state
        state_obj.final_response = final_response
        state_obj.messages.append(Message(
            role="assistant",
            content=final_response
        ))
        
    except Exception as e:
        logger.error(f"Error in generate_response: {str(e)}", exc_info=True)
        state_obj.errors.append({
            "step": "generate_response",
            "error": str(e)
        })
    
    return state_obj.model_dump()

# Create the subgraphs
def create_classification_subgraph(model):
    """Create the query classification subgraph."""
    subgraph = StateGraph(WorkflowState)
    
    # Add nodes
    subgraph.add_node("classify", lambda state: classify_query(state, model))
    
    # Add edges
    subgraph.add_edge("classify", END)
    
    # Set entry point
    subgraph.set_entry_point("classify")
    
    return subgraph.compile()

def create_research_subgraph(model):
    """Create the research subgraph."""
    subgraph = StateGraph(WorkflowState)
    
    # Add nodes
    subgraph.add_node("research", lambda state: research_query(state, model))
    
    # Add edges
    subgraph.add_edge("research", END)
    
    # Set entry point
    subgraph.set_entry_point("research")
    
    return subgraph.compile()

def create_analysis_subgraph(model):
    """Create the analysis subgraph."""
    subgraph = StateGraph(WorkflowState)
    
    # Add nodes
    subgraph.add_node("analyze", lambda state: analyze_query(state, model))
    
    # Add edges
    subgraph.add_edge("analyze", END)
    
    # Set entry point
    subgraph.set_entry_point("analyze")
    
    return subgraph.compile()

def create_code_generation_subgraph(model):
    """Create the code generation subgraph."""
    subgraph = StateGraph(WorkflowState)
    
    # Add nodes
    subgraph.add_node("generate_code", lambda state: generate_code(state, model))
    
    # Add edges
    subgraph.add_edge("generate_code", END)
    
    # Set entry point
    subgraph.set_entry_point("generate_code")
    
    return subgraph.compile()

def create_response_generation_subgraph(model):
    """Create the response generation subgraph."""
    subgraph = StateGraph(WorkflowState)
    
    # Add nodes
    subgraph.add_node("generate_response", lambda state: generate_response(state, model))
    
    # Add edges
    subgraph.add_edge("generate_response", END)
    
    # Set entry point
    subgraph.set_entry_point("generate_response")
    
    return subgraph.compile()

# Main workflow with branching logic
def create_main_workflow(model):
    """Create the main workflow with subgraphs and branching logic."""
    # Create the subgraphs
    classification_subgraph = create_classification_subgraph(model)
    research_subgraph = create_research_subgraph(model)
    analysis_subgraph = create_analysis_subgraph(model)
    code_generation_subgraph = create_code_generation_subgraph(model)
    response_generation_subgraph = create_response_generation_subgraph(model)
    
    # Create the main graph
    main_graph = StateGraph(WorkflowState)
    
    # Add the subgraphs as nodes
    main_graph.add_node("classification", classification_subgraph)
    main_graph.add_node("research", research_subgraph)
    main_graph.add_node("analysis", analysis_subgraph)
    main_graph.add_node("code_generation", code_generation_subgraph)
    main_graph.add_node("response_generation", response_generation_subgraph)
    
    # Define the routing logic based on query classification
    def route_after_classification(state: Dict[str, Any]) -> str:
        """Route to the appropriate next step based on classification."""
        state_obj = WorkflowState.model_validate(state)
        
        if not state_obj.classification:
            return "response_generation"  # Default if classification failed
        
        # Update the current subgraph
        state_obj.current_subgraph = "classification"
        
        # Determine the next steps based on classification
        next_steps = []
        
        # Always do research for factual and analytical queries
        if state_obj.classification.category in [QueryCategory.FACTUAL, QueryCategory.ANALYTICAL] or state_obj.classification.requires_research:
            next_steps.append("research")
        
        # Always do analysis for analytical queries or if reasoning is required
        if state_obj.classification.category == QueryCategory.ANALYTICAL or state_obj.classification.requires_reasoning:
            next_steps.append("analysis")
        
        # Generate code for technical queries or if code is required
        if state_obj.classification.category == QueryCategory.TECHNICAL or state_obj.classification.requires_code:
            next_steps.append("code_generation")
        
        # If no specific steps are needed, go straight to response generation
        if not next_steps:
            return "response_generation"
        
        # Return the first step in the sequence
        return next_steps[0]
    
    def route_after_research(state: Dict[str, Any]) -> str:
        """Route after the research step."""
        state_obj = WorkflowState.model_validate(state)
        
        # Update the current subgraph
        state_obj.current_subgraph = "research"
        
        if not state_obj.classification:
            return "response_generation"
        
        # Check if analysis is needed
        if state_obj.classification.category == QueryCategory.ANALYTICAL or state_obj.classification.requires_reasoning:
            return "analysis"
        
        # Check if code generation is needed
        if state_obj.classification.category == QueryCategory.TECHNICAL or state_obj.classification.requires_code:
            return "code_generation"
        
        # Default to response generation
        return "response_generation"
    
    def route_after_analysis(state: Dict[str, Any]) -> str:
        """Route after the analysis step."""
        state_obj = WorkflowState.model_validate(state)
        
        # Update the current subgraph
        state_obj.current_subgraph = "analysis"
        
        # Check if code generation is needed
        if state_obj.classification and (state_obj.classification.category == QueryCategory.TECHNICAL or state_obj.classification.requires_code):
            return "code_generation"
        
        # Default to response generation
        return "response_generation"
    
    def route_after_code_generation(state: Dict[str, Any]) -> str:
        """Route after the code generation step."""
        state_obj = WorkflowState.model_validate(state)
        
        # Update the current subgraph
        state_obj.current_subgraph = "code_generation"
        
        # Always go to response generation after code generation
        return "response_generation"
    
    # Add the edges with conditional routing
    main_graph.add_edge("classification", route_after_classification)
    main_graph.add_edge("research", route_after_research)
    main_graph.add_edge("analysis", route_after_analysis)
    main_graph.add_edge("code_generation", route_after_code_generation)
    main_graph.add_edge("response_generation", END)
    
    # Set the entry point
    main_graph.set_entry_point("classification")
    
    return main_graph.compile()

# Main function to run the workflow
async def run_workflow(query: str):
    """Run the workflow with subgraphs and branching."""
    logger.info(f"Running workflow for query: {query}")
    
    # Get the Google API key from environment
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        logger.error("GOOGLE_API_KEY not found in environment variables")
        print("Please set your GOOGLE_API_KEY in the .env file")
        return {"error": "API key not found"}
    
    # Get the model
    model = get_model(google_api_key)
    
    # Create the workflow
    workflow = create_main_workflow(model)
    
    # Initialize the state
    initial_state = WorkflowState(query=query)
    
    # Run the workflow
    try:
        logger.info("Running the workflow...")
        result = await workflow.ainvoke(initial_state.model_dump())
        return result
    except Exception as e:
        logger.error(f"Error running workflow: {str(e)}", exc_info=True)
        initial_state.errors.append({
            "step": "workflow",
            "error": str(e)
        })
        return initial_state.model_dump()

# Interactive CLI
async def main():
    """Run the subgraphs and branching example."""
    print("=" * 80)
    print("Subgraphs and Branching in LangGraph")
    print("=" * 80)
    print("\nThis example demonstrates how to create modular subgraphs and implement")
    print("complex branching logic in LangGraph for advanced workflow patterns.")
    
    # Get user input
    query = input("\nEnter your query: ")
    
    print("\nProcessing with subgraphs and branching workflow...")
    result = await run_workflow(query)
    
    # Display the result
    print("\n" + "=" * 80)
    print("Workflow Result:")
    print("=" * 80)
    
    # Show the classification
    if "classification" in result and result["classification"]:
        classification = result["classification"]
        print(f"\nQuery Classification:")
        print(f"- Category: {classification['category']}")
        print(f"- Complexity: {classification['complexity']}")
        print(f"- Requires Research: {classification['requires_research']}")
        print(f"- Requires Reasoning: {classification['requires_reasoning']}")
        print(f"- Requires Code: {classification['requires_code']}")
        print(f"- Confidence: {classification['confidence']:.2f}")
    
    # Show the execution path
    print("\nExecution Path:")
    if "subgraph_states" in result:
        for subgraph in result["subgraph_states"]:
            print(f"- {subgraph}")
    
    # Show the final response
    if "final_response" in result and result["final_response"]:
        print("\nFinal Response:")
        print("-" * 40)
        print(result["final_response"])
        print("-" * 40)
    else:
        print("\nNo final response generated.")
    
    # Show any errors
    if "errors" in result and result["errors"]:
        print("\nErrors:")
        for error in result["errors"]:
            print(f"- {error.get('step', 'unknown')}: {error.get('error', 'unknown error')}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
