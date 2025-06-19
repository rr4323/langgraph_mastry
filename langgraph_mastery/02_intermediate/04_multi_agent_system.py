"""
LangGraph Intermediate: Multi-Agent System
========================================

This script demonstrates how to build a system with multiple specialized agents
using LangGraph and Google's Generative AI model.
"""

import os
import sys
from typing import Annotated, Dict, List, Literal, TypedDict, Union, Any
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent

# Add the parent directory to the path so we can import from the root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables from .env file
load_dotenv()

# Define our state for the multi-agent system
class MultiAgentState(TypedDict):
    """State for our multi-agent system."""
    messages: List[Union[HumanMessage, AIMessage, SystemMessage]]
    next: Literal["researcher", "analyst", "writer", "end"]
    research_notes: str
    analysis: str
    final_response: str

# Define a Pydantic model for the final structured output
class ArticleResponse(BaseModel):
    """Structured response for the final article."""
    title: str = Field(description="The title of the article")
    introduction: str = Field(description="The introduction paragraph")
    key_points: List[str] = Field(description="List of key points covered in the article")
    conclusion: str = Field(description="The conclusion paragraph")
    
    def __str__(self):
        """String representation of the article."""
        key_points_str = "\n".join([f"- {point}" for point in self.key_points])
        return f"""# {self.title}

## Introduction
{self.introduction}

## Key Points
{key_points_str}

## Conclusion
{self.conclusion}
"""

def create_researcher_agent():
    """Create an agent specialized in research."""
    model = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.3,  # Lower temperature for more factual responses
        convert_system_message_to_human=True
    )
    
    # Create the researcher agent
    agent = create_react_agent(
        model=model,
        tools=[],  # No tools for simplicity, but you could add search tools here
        prompt="""You are a specialized research agent.
Your job is to gather comprehensive information on a topic.
Focus on collecting factual information, key concepts, and important details.
Organize your research in a clear, structured format.
Be thorough but concise.
"""
    )
    
    return agent

def create_analyst_agent():
    """Create an agent specialized in analysis."""
    model = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.4,
        convert_system_message_to_human=True
    )
    
    # Create the analyst agent
    agent = create_react_agent(
        model=model,
        tools=[],
        prompt="""You are a specialized analysis agent.
Your job is to analyze research information and extract insights.
Identify patterns, implications, and connections between concepts.
Evaluate the significance of the information.
Provide a thoughtful analysis that goes beyond summarizing.
"""
    )
    
    return agent

def create_writer_agent():
    """Create an agent specialized in writing."""
    model = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.7,  # Higher temperature for more creative writing
        convert_system_message_to_human=True
    )
    
    # Create the writer agent
    agent = create_react_agent(
        model=model,
        tools=[],
        prompt="""You are a specialized writing agent.
Your job is to create well-structured, engaging content.
Use the research and analysis provided to craft a comprehensive article.
Write in a clear, concise, and engaging style.
Include an introduction, key points, and a conclusion.
"""
    )
    
    return agent

def researcher_node(state: MultiAgentState) -> MultiAgentState:
    """Researcher node in our multi-agent system."""
    print("🔍 Researcher agent is gathering information...")
    
    researcher = create_researcher_agent()
    
    # Extract the user's query from the messages
    user_message = next((msg.content for msg in state["messages"] if isinstance(msg, HumanMessage)), "")
    
    # Create a message for the researcher
    messages = [
        SystemMessage(content="""You are a specialized research agent.
Your job is to gather comprehensive information on a topic.
Focus on collecting factual information, key concepts, and important details.
Organize your research in a clear, structured format."""),
        HumanMessage(content=f"Research this topic thoroughly: {user_message}")
    ]
    
    # Get research information
    response = researcher.invoke({"messages": messages})
    research_notes = response["messages"][-1].content
    
    # Update the state
    return {
        **state,
        "research_notes": research_notes,
        "next": "analyst"
    }

def analyst_node(state: MultiAgentState) -> MultiAgentState:
    """Analyst node in our multi-agent system."""
    print("🧠 Analyst agent is analyzing the research...")
    
    analyst = create_analyst_agent()
    
    # Create a message for the analyst
    messages = [
        SystemMessage(content="""You are a specialized analysis agent.
Your job is to analyze research information and extract insights.
Identify patterns, implications, and connections between concepts.
Evaluate the significance of the information."""),
        HumanMessage(content=f"""
Analyze the following research information:

{state['research_notes']}

Provide a thoughtful analysis that identifies key insights, patterns, and implications.
""")
    ]
    
    # Get analysis
    response = analyst.invoke({"messages": messages})
    analysis = response["messages"][-1].content
    
    # Update the state
    return {
        **state,
        "analysis": analysis,
        "next": "writer"
    }

def writer_node(state: MultiAgentState) -> MultiAgentState:
    """Writer node in our multi-agent system."""
    print("✍️ Writer agent is crafting the final response...")
    
    writer = create_writer_agent()
    
    # Extract the user's query from the messages
    user_message = next((msg.content for msg in state["messages"] if isinstance(msg, HumanMessage)), "")
    
    # Create a message for the writer
    messages = [
        SystemMessage(content="""You are a specialized writing agent.
Your job is to create well-structured, engaging content.
Use the research and analysis provided to craft a comprehensive article.
Write in a clear, concise, and engaging style.
Include an introduction, key points, and a conclusion."""),
        HumanMessage(content=f"""
Create a well-structured article on the following topic:
{user_message}

Here is the research information:
{state['research_notes']}

Here is the analysis:
{state['analysis']}

Craft a comprehensive article with a title, introduction, key points, and conclusion.
""")
    ]
    
    # Get the final response
    response = writer.invoke({"messages": messages})
    final_response = response["messages"][-1].content
    
    # Update the state
    return {
        **state,
        "final_response": final_response,
        "next": "end"
    }

def create_multi_agent_system():
    """Create a multi-agent system using LangGraph."""
    print("Creating a multi-agent system with LangGraph...")
    
    # Create a new graph
    workflow = StateGraph(MultiAgentState)
    
    # Add nodes to the graph
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("analyst", analyst_node)
    workflow.add_node("writer", writer_node)
    
    # Add edges to the graph
    workflow.add_conditional_edges(
        "researcher",
        lambda state: state["next"],
        {
            "analyst": "analyst",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "analyst",
        lambda state: state["next"],
        {
            "writer": "writer",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "writer",
        lambda state: state["next"],
        {
            "end": END
        }
    )
    
    # Set the entry point
    workflow.set_entry_point("researcher")
    
    # Compile the graph
    return workflow.compile()

def format_article_as_structured_output(article_text: str) -> ArticleResponse:
    """Format the article text as a structured output."""
    # Initialize the chat model with Google's Gemini Pro
    model = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.2,  # Low temperature for consistent formatting
        convert_system_message_to_human=True
    )
    
    # Create a message to extract structured information
    messages = [
        SystemMessage(content="""You are a content formatter.
Extract the structured information from the given article.
Format it according to the specified structure."""),
        HumanMessage(content=f"""
Extract the following information from this article:
1. Title
2. Introduction paragraph
3. Key points (as a list)
4. Conclusion paragraph

Here's the article:
{article_text}

Format your response as a JSON object with the following structure:
{{
  "title": "The title",
  "introduction": "The introduction paragraph",
  "key_points": ["Point 1", "Point 2", "Point 3"],
  "conclusion": "The conclusion paragraph"
}}
""")
    ]
    
    # Get the structured information
    response = model.invoke(messages)
    
    # Parse the response as JSON
    import json
    import re
    
    # Extract JSON from the response
    json_match = re.search(r'```json\n(.*?)\n```', response.content, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        json_str = response.content
    
    # Clean up the JSON string
    json_str = re.sub(r'```json|```', '', json_str).strip()
    
    try:
        article_data = json.loads(json_str)
        return ArticleResponse(**article_data)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error parsing JSON: {e}")
        # Fallback with default values
        return ArticleResponse(
            title="Article Title",
            introduction="Introduction paragraph",
            key_points=["Key point 1", "Key point 2", "Key point 3"],
            conclusion="Conclusion paragraph"
        )

def run_multi_agent_system(workflow, query: str):
    """Run the multi-agent system with a given query."""
    print("\n" + "=" * 50)
    print("Running the Multi-Agent System")
    print("=" * 50)
    
    # Initialize the state
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "next": "researcher",
        "research_notes": "",
        "analysis": "",
        "final_response": ""
    }
    
    # Run the workflow
    print(f"\nQuery: {query}")
    result = workflow.invoke(initial_state)
    
    # Print the intermediate results
    print("\n" + "=" * 50)
    print("Intermediate Results")
    print("=" * 50)
    print("\nResearch Notes:")
    print("-" * 50)
    print(result["research_notes"])
    
    print("\nAnalysis:")
    print("-" * 50)
    print(result["analysis"])
    
    # Format the final response as a structured output
    article = format_article_as_structured_output(result["final_response"])
    
    # Print the final result
    print("\n" + "=" * 50)
    print("Final Article")
    print("=" * 50)
    print(article)
    
    return article

def main():
    """Main function to demonstrate a multi-agent system."""
    print("=" * 50)
    print("Multi-Agent System in LangGraph")
    print("=" * 50)
    
    # Create a multi-agent system
    workflow = create_multi_agent_system()
    
    # Get a topic from the user
    topic = input("\nEnter a topic for the multi-agent system to research and write about: ")
    
    # Run the multi-agent system
    run_multi_agent_system(workflow, topic)

if __name__ == "__main__":
    main()
