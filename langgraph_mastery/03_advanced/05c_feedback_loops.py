"""
LangGraph Advanced: Feedback Loops
================================

This script demonstrates how to implement self-improving systems with feedback
mechanisms using LangGraph and Google's Generative AI model.
"""

import os
import sys
import time
import json
import uuid
from typing import Annotated, Dict, List, Literal, TypedDict, Union, Any, Optional
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

# Define quality metrics
class QualityMetric(str, Enum):
    """Quality metrics for evaluating responses."""
    RELEVANCE = "relevance"
    ACCURACY = "accuracy"
    COMPLETENESS = "completeness"
    CLARITY = "clarity"
    HELPFULNESS = "helpfulness"

# Define feedback model
class Feedback(BaseModel):
    """Model for feedback on responses."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    metrics: Dict[QualityMetric, float]
    comments: str
    timestamp: float = Field(default_factory=time.time)
    
    def get_average_score(self) -> float:
        """Get the average score across all metrics."""
        return sum(self.metrics.values()) / len(self.metrics) if self.metrics else 0

# Define our state for the feedback loop system
class FeedbackLoopState(TypedDict):
    """State for our feedback loop system."""
    messages: List[Union[HumanMessage, AIMessage, SystemMessage]]
    query: str
    response: str
    feedback: Optional[Feedback]
    improvement_plan: str
    improved_response: str
    learning_points: List[str]
    performance_history: List[Dict[str, Any]]
    next: Optional[str]

# File paths for persistence
FEEDBACK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "feedback")
PERFORMANCE_FILE = os.path.join(FEEDBACK_DIR, "performance_history.json")

# Create the feedback directory if it doesn't exist
os.makedirs(FEEDBACK_DIR, exist_ok=True)

def load_performance_history() -> List[Dict[str, Any]]:
    """Load the performance history from disk."""
    if os.path.exists(PERFORMANCE_FILE):
        with open(PERFORMANCE_FILE, "r") as f:
            return json.load(f)
    else:
        return []

def save_performance_history(history: List[Dict[str, Any]]):
    """Save the performance history to disk."""
    with open(PERFORMANCE_FILE, "w") as f:
        json.dump(history, f, indent=2)

def save_feedback(feedback: Feedback):
    """Save feedback to disk."""
    feedback_file = os.path.join(FEEDBACK_DIR, f"feedback_{feedback.id}.json")
    with open(feedback_file, "w") as f:
        json.dump(feedback.model_dump(), f, indent=2)

# Workflow nodes
def response_generation_node(state: FeedbackLoopState) -> FeedbackLoopState:
    """Response generation node in the feedback loop system."""
    print("✍️ Generating initial response...")
    
    # Extract the query from the state
    query = state["query"]
    
    # Load performance history to inform the response
    performance_history = load_performance_history()
    
    # Create a response agent
    model = ChatGoogleGenerativeAI(
        model="gemini-pro",
        temperature=0.7,
        convert_system_message_to_human=True
    )
    
    # Prepare system message with performance insights
    system_content = "You are a helpful assistant that generates high-quality responses."
    
    if performance_history:
        # Calculate average scores for each metric
        metric_scores = {}
        for metric in QualityMetric:
            scores = [entry["feedback"]["metrics"].get(metric, 0) for entry in performance_history if "feedback" in entry]
            if scores:
                metric_scores[metric] = sum(scores) / len(scores)
        
        # Add performance insights to the system message
        system_content += "\n\nBased on past performance, please focus on improving these aspects:"
        
        # Sort metrics by score (ascending)
        sorted_metrics = sorted(metric_scores.items(), key=lambda x: x[1])
        
        for metric, score in sorted_metrics[:3]:  # Focus on the 3 lowest-scoring metrics
            system_content += f"\n- {metric.capitalize()}: Previous average score {score:.2f}/5.0"
    
    # Create messages for the response
    messages = [
        SystemMessage(content=system_content),
        HumanMessage(content=query)
    ]
    
    # Get the response
    response = model.invoke(messages)
    
    # Update the state
    return {
        **state,
        "response": response.content,
        "next": "feedback_collection"
    }

def feedback_collection_node(state: FeedbackLoopState) -> FeedbackLoopState:
    """Feedback collection node in the feedback loop system."""
    print("\n" + "=" * 50)
    print("Feedback Collection")
    print("=" * 50)
    
    # Display the query and response
    print(f"\nQuery: {state['query']}")
    print("\nResponse:")
    print("-" * 50)
    print(state["response"])
    print("-" * 50)
    
    # Collect feedback
    print("\nPlease provide feedback on the response:")
    
    # Collect scores for each metric
    metrics = {}
    for metric in QualityMetric:
        while True:
            try:
                score = float(input(f"{metric.capitalize()} (1-5): "))
                if 1 <= score <= 5:
                    metrics[metric] = score
                    break
                else:
                    print("Please enter a score between 1 and 5.")
            except ValueError:
                print("Please enter a valid number.")
    
    # Collect comments
    comments = input("\nAdditional comments: ")
    
    # Create a feedback object
    feedback = Feedback(
        metrics=metrics,
        comments=comments
    )
    
    # Save the feedback
    save_feedback(feedback)
    
    # Update the state
    return {
        **state,
        "feedback": feedback.model_dump(),
        "next": "feedback_analysis"
    }

def feedback_analysis_node(state: FeedbackLoopState) -> FeedbackLoopState:
    """Feedback analysis node in the feedback loop system."""
    print("🔍 Analyzing feedback...")
    
    # Extract the query, response, and feedback from the state
    query = state["query"]
    response = state["response"]
    feedback = Feedback(**state["feedback"])
    
    # Create a feedback analysis agent
    model = ChatGoogleGenerativeAI(
        model="gemini-pro",
        temperature=0.4,
        convert_system_message_to_human=True
    )
    
    # Create messages for the analysis
    messages = [
        SystemMessage(content="""You are a feedback analysis specialist.
Your job is to analyze feedback on a response and identify areas for improvement."""),
        HumanMessage(content=f"""
Query: {query}

Response:
{response}

Feedback:
- Relevance: {feedback.metrics.get(QualityMetric.RELEVANCE, 'N/A')}/5
- Accuracy: {feedback.metrics.get(QualityMetric.ACCURACY, 'N/A')}/5
- Completeness: {feedback.metrics.get(QualityMetric.COMPLETENESS, 'N/A')}/5
- Clarity: {feedback.metrics.get(QualityMetric.CLARITY, 'N/A')}/5
- Helpfulness: {feedback.metrics.get(QualityMetric.HELPFULNESS, 'N/A')}/5

Comments: {feedback.comments}

Please analyze this feedback and identify:
1. Strengths of the response
2. Areas for improvement
3. Specific recommendations for addressing each area of improvement
""")
    ]
    
    # Get the analysis
    analysis_response = model.invoke(messages)
    improvement_plan = analysis_response.content
    
    # Update the state
    return {
        **state,
        "improvement_plan": improvement_plan,
        "next": "response_improvement"
    }

def response_improvement_node(state: FeedbackLoopState) -> FeedbackLoopState:
    """Response improvement node in the feedback loop system."""
    print("🔄 Improving the response based on feedback...")
    
    # Extract the query, response, and improvement plan from the state
    query = state["query"]
    response = state["response"]
    improvement_plan = state["improvement_plan"]
    feedback = Feedback(**state["feedback"])
    
    # Create a response improvement agent
    model = ChatGoogleGenerativeAI(
        model="gemini-pro",
        temperature=0.6,
        convert_system_message_to_human=True
    )
    
    # Create messages for the improvement
    messages = [
        SystemMessage(content="""You are a response improvement specialist.
Your job is to improve a response based on feedback and an improvement plan."""),
        HumanMessage(content=f"""
Query: {query}

Original Response:
{response}

Feedback Scores:
- Relevance: {feedback.metrics.get(QualityMetric.RELEVANCE, 'N/A')}/5
- Accuracy: {feedback.metrics.get(QualityMetric.ACCURACY, 'N/A')}/5
- Completeness: {feedback.metrics.get(QualityMetric.COMPLETENESS, 'N/A')}/5
- Clarity: {feedback.metrics.get(QualityMetric.CLARITY, 'N/A')}/5
- Helpfulness: {feedback.metrics.get(QualityMetric.HELPFULNESS, 'N/A')}/5

Improvement Plan:
{improvement_plan}

Please create an improved version of the response that addresses the feedback and follows the improvement plan.
""")
    ]
    
    # Get the improved response
    improved_response = model.invoke(messages)
    
    # Update the state
    return {
        **state,
        "improved_response": improved_response.content,
        "next": "learning_extraction"
    }

def learning_extraction_node(state: FeedbackLoopState) -> FeedbackLoopState:
    """Learning extraction node in the feedback loop system."""
    print("📚 Extracting learning points...")
    
    # Extract the query, original response, improved response, and feedback from the state
    query = state["query"]
    response = state["response"]
    improved_response = state["improved_response"]
    feedback = Feedback(**state["feedback"])
    improvement_plan = state["improvement_plan"]
    
    # Create a learning extraction agent
    model = ChatGoogleGenerativeAI(
        model="gemini-pro",
        temperature=0.3,
        convert_system_message_to_human=True
    )
    
    # Create messages for the learning extraction
    messages = [
        SystemMessage(content="""You are a learning extraction specialist.
Your job is to extract clear, actionable learning points from a feedback and improvement process."""),
        HumanMessage(content=f"""
Query: {query}

Original Response:
{response}

Feedback:
- Relevance: {feedback.metrics.get(QualityMetric.RELEVANCE, 'N/A')}/5
- Accuracy: {feedback.metrics.get(QualityMetric.ACCURACY, 'N/A')}/5
- Completeness: {feedback.metrics.get(QualityMetric.COMPLETENESS, 'N/A')}/5
- Clarity: {feedback.metrics.get(QualityMetric.CLARITY, 'N/A')}/5
- Helpfulness: {feedback.metrics.get(QualityMetric.HELPFULNESS, 'N/A')}/5
- Comments: {feedback.comments}

Improvement Plan:
{improvement_plan}

Improved Response:
{improved_response}

Please extract 3-5 clear, actionable learning points from this process. Each learning point should be:
1. Specific and concrete
2. Generalizable to similar queries
3. Actionable for future responses
""")
    ]
    
    # Get the learning points
    learning_response = model.invoke(messages)
    
    # Extract the learning points
    learning_points = []
    for line in learning_response.content.split("\n"):
        line = line.strip()
        if line and (line.startswith("-") or line.startswith("*") or line.startswith("#") or line[0].isdigit() and line[1] in [".", ")"]):
            learning_points.append(line.lstrip("- *#0123456789.) "))
    
    # If no learning points were extracted, use the whole response
    if not learning_points:
        learning_points = [learning_response.content]
    
    # Update the performance history
    performance_history = state["performance_history"]
    performance_entry = {
        "timestamp": time.time(),
        "query": query,
        "original_response": response,
        "feedback": feedback.model_dump(),
        "improvement_plan": improvement_plan,
        "improved_response": improved_response,
        "learning_points": learning_points
    }
    performance_history.append(performance_entry)
    
    # Save the updated performance history
    save_performance_history(performance_history)
    
    # Update the state
    return {
        **state,
        "learning_points": learning_points,
        "performance_history": performance_history,
        "next": "end"
    }

def create_feedback_loop_system():
    """Create a feedback loop system using LangGraph."""
    print("Creating a feedback loop system with LangGraph...")
    
    # Create a new graph
    workflow = StateGraph(FeedbackLoopState)
    
    # Add nodes to the graph
    workflow.add_node("response_generation", response_generation_node)
    workflow.add_node("feedback_collection", feedback_collection_node)
    workflow.add_node("feedback_analysis", feedback_analysis_node)
    workflow.add_node("response_improvement", response_improvement_node)
    workflow.add_node("learning_extraction", learning_extraction_node)
    
    # Add edges
    workflow.add_conditional_edges(
        "response_generation",
        lambda state: state["next"],
        {
            "feedback_collection": "feedback_collection"
        }
    )
    
    workflow.add_conditional_edges(
        "feedback_collection",
        lambda state: state["next"],
        {
            "feedback_analysis": "feedback_analysis"
        }
    )
    
    workflow.add_conditional_edges(
        "feedback_analysis",
        lambda state: state["next"],
        {
            "response_improvement": "response_improvement"
        }
    )
    
    workflow.add_conditional_edges(
        "response_improvement",
        lambda state: state["next"],
        {
            "learning_extraction": "learning_extraction"
        }
    )
    
    workflow.add_conditional_edges(
        "learning_extraction",
        lambda state: state["next"],
        {
            "end": END
        }
    )
    
    # Set the entry point
    workflow.set_entry_point("response_generation")
    
    # Compile the graph
    return workflow.compile()

def run_feedback_loop_system(workflow, query: str):
    """Run the feedback loop system with a given query."""
    print("\n" + "=" * 50)
    print("Running the Feedback Loop System")
    print("=" * 50)
    
    # Load the performance history
    performance_history = load_performance_history()
    
    # Initialize the state
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "query": query,
        "response": "",
        "feedback": None,
        "improvement_plan": "",
        "improved_response": "",
        "learning_points": [],
        "performance_history": performance_history,
        "next": None
    }
    
    # Run the workflow
    print(f"\nQuery: {query}")
    result = workflow.invoke(initial_state)
    
    # Print the workflow results
    print("\n" + "=" * 50)
    print("Feedback Loop System Results")
    print("=" * 50)
    
    print("\nOriginal Response:")
    print("-" * 50)
    print(result["response"])
    
    print("\nFeedback:")
    print("-" * 50)
    feedback = Feedback(**result["feedback"])
    for metric, score in feedback.metrics.items():
        print(f"{metric.capitalize()}: {score}/5")
    print(f"Comments: {feedback.comments}")
    
    print("\nImprovement Plan:")
    print("-" * 50)
    print(result["improvement_plan"])
    
    print("\nImproved Response:")
    print("-" * 50)
    print(result["improved_response"])
    
    print("\nLearning Points:")
    print("-" * 50)
    for i, point in enumerate(result["learning_points"], 1):
        print(f"{i}. {point}")
    
    return result

def view_performance_history():
    """View the performance history of the system."""
    print("\n" + "=" * 50)
    print("Performance History")
    print("=" * 50)
    
    # Load the performance history
    performance_history = load_performance_history()
    
    if not performance_history:
        print("\nNo performance history available.")
        return
    
    # Calculate average scores over time
    print("\nAverage Scores Over Time:")
    print("-" * 50)
    
    # Group entries by day
    entries_by_day = {}
    for entry in performance_history:
        timestamp = entry["timestamp"]
        day = time.strftime("%Y-%m-%d", time.localtime(timestamp))
        
        if day not in entries_by_day:
            entries_by_day[day] = []
        
        entries_by_day[day].append(entry)
    
    # Calculate average scores for each day
    for day, entries in sorted(entries_by_day.items()):
        print(f"\nDate: {day}")
        
        # Calculate average scores for each metric
        metric_scores = {}
        for metric in QualityMetric:
            scores = [entry["feedback"]["metrics"].get(metric, 0) for entry in entries if "feedback" in entry]
            if scores:
                metric_scores[metric] = sum(scores) / len(scores)
        
        # Print the average scores
        for metric, score in metric_scores.items():
            print(f"{metric.capitalize()}: {score:.2f}/5")
    
    # Print the most recent learning points
    print("\nRecent Learning Points:")
    print("-" * 50)
    
    recent_entries = sorted(performance_history, key=lambda x: x["timestamp"], reverse=True)[:5]
    for entry in recent_entries:
        print(f"\nQuery: {entry['query']}")
        for i, point in enumerate(entry.get("learning_points", []), 1):
            print(f"{i}. {point}")
        print()

def main():
    """Main function to demonstrate feedback loops."""
    print("=" * 50)
    print("Feedback Loops in LangGraph")
    print("=" * 50)
    
    # Let the user choose a mode
    print("\nSelect a mode:")
    print("1. Run the feedback loop system")
    print("2. View performance history")
    
    while True:
        try:
            choice = int(input("\nEnter your choice (1-2): "))
            
            if choice == 1:
                # Create a feedback loop system
                workflow = create_feedback_loop_system()
                
                # Get a query from the user
                query = input("\nEnter a query: ")
                
                # Run the feedback loop system
                run_feedback_loop_system(workflow, query)
                break
                
            elif choice == 2:
                # View the performance history
                view_performance_history()
                break
                
            else:
                print("Please enter 1 or 2.")
        except ValueError:
            print("Please enter a valid number.")

if __name__ == "__main__":
    main()
