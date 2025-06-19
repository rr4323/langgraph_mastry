"""
Multi-Agent System with MCP Communication for Advanced LangGraph

This script demonstrates how to create a multi-agent system where agents
communicate through the Model Context Protocol (MCP).
"""

import os
import sys
import re
import asyncio
import time
import functools
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Check for required packages
required_packages = ["langchain-mcp-adapters", "langchain-google-genai", "langgraph"]
for package in required_packages:
    try:
        __import__(package.replace("-", "_"))
    except ImportError:
        logger.info(f"Installing {package}...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END

# Define the state model for our multi-agent system
class AgentMessage(BaseModel):
    """A message in the conversation between agents."""
    role: str
    content: str
    timestamp: float = Field(default_factory=time.time)

class MultiAgentState(BaseModel):
    """The state for our multi-agent system."""
    query: str
    conversation: List[AgentMessage] = Field(default_factory=list)
    current_agent: str = "coordinator"
    current_instructions: Optional[str] = None
    final_response: Optional[str] = None
    context: Dict[str, Any] = Field(default_factory=dict)
    errors: List[Dict[str, Any]] = Field(default_factory=list)

# Agent creation functions
def create_coordinator_agent(google_api_key: str):
    """Create the coordinator agent."""
    model = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=google_api_key,
        temperature=0.2,
        convert_system_message_to_human=True
    )
    
    coordinator_prompt = """You are the Coordinator Agent in a multi-agent system.
Your role is to:
1. Understand the user's query
2. Decide which specialist agent should handle the query next
3. Provide clear instructions to the specialist agent
4. Review responses from specialist agents
5. Determine when the task is complete

The available specialist agents are:
- researcher: Good at finding and retrieving information
- analyst: Good at analyzing data and extracting insights
- writer: Good at crafting well-written responses

For each turn, you should:
1. Review the conversation history
2. Decide which agent should act next
3. Provide clear instructions to that agent
4. If the task is complete, indicate that the final response should be sent to the user

Your response should be in this format:
NEXT_AGENT: [researcher|analyst|writer|final]
INSTRUCTIONS: [Clear instructions for the next agent or final response]
"""
    
    return create_react_agent(
        model=model,
        tools=[],
        prompt=coordinator_prompt
    )

def create_researcher_agent(google_api_key: str, mcp_tools: List):
    """Create the researcher agent."""
    model = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=google_api_key,
        temperature=0.3,
        convert_system_message_to_human=True
    )
    
    researcher_prompt = """You are the Researcher Agent in a multi-agent system.
Your role is to find and retrieve relevant information based on the instructions from the Coordinator.
You have access to specialized tools to search knowledge bases and extract information.
Use these tools to provide accurate and comprehensive information.

When you receive instructions:
1. Understand what information needs to be retrieved
2. Use your tools to search for relevant information
3. Compile the retrieved information in a clear and structured format
4. Provide your findings to be used by other agents

Focus on being thorough and accurate in your research.
"""
    
    return create_react_agent(
        model=model,
        tools=mcp_tools,
        prompt=researcher_prompt
    )

def create_analyst_agent(google_api_key: str, mcp_tools: List):
    """Create the analyst agent."""
    model = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=google_api_key,
        temperature=0.4,
        convert_system_message_to_human=True
    )
    
    analyst_prompt = """You are the Analyst Agent in a multi-agent system.
Your role is to analyze information and extract insights based on the instructions from the Coordinator.
You have access to specialized tools for sentiment analysis and entity extraction.
Use these tools to provide deep analysis and meaningful insights.

When you receive instructions:
1. Understand what analysis needs to be performed
2. Use your tools to analyze the provided information
3. Identify patterns, trends, and key insights
4. Provide your analysis in a clear and structured format

Focus on being insightful and providing value-added analysis.
"""
    
    return create_react_agent(
        model=model,
        tools=mcp_tools,
        prompt=analyst_prompt
    )

def create_writer_agent(google_api_key: str):
    """Create the writer agent."""
    model = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=google_api_key,
        temperature=0.6,
        convert_system_message_to_human=True
    )
    
    writer_prompt = """You are the Writer Agent in a multi-agent system.
Your role is to craft well-written responses based on the information and analysis provided by other agents.
You should create clear, concise, and engaging content that effectively communicates the requested information.

When you receive instructions:
1. Understand what content needs to be created
2. Review the information and analysis provided by other agents
3. Organize the content in a logical and coherent structure
4. Write the content in a style appropriate for the audience and purpose
5. Ensure the content is accurate, complete, and addresses the original query

Focus on clarity, coherence, and effective communication.
"""
    
    return create_react_agent(
        model=model,
        tools=[],
        prompt=writer_prompt
    )

# Node functions for the graph
async def coordinator_node(state: Dict[str, Any], agent, google_api_key: str) -> Dict[str, Any]:
    """Process the state through the coordinator agent."""
    logger.info("--- Running Coordinator Node ---")
    state_obj = MultiAgentState.model_validate(state)
    
    # Prepare the messages for the agent
    messages = []
    
    # Add the initial query if this is the first turn
    if len(state_obj.conversation) == 0:
        messages.append({"role": "user", "content": f"User query: {state_obj.query}"})
    else:
        # Add the conversation history
        messages.append({"role": "user", "content": f"User query: {state_obj.query}\n\nConversation history:"})
        for msg in state_obj.conversation:
            messages.append({"role": "user", "content": f"{msg.role}: {msg.content}"})
    
    # Invoke the agent
    try:
        response = await agent.ainvoke({"messages": messages})
        assistant_message = next((msg.content for msg in response["messages"] if msg.type == "ai"), "")

        logger.info(f"Coordinator raw response: {assistant_message}")
        
        # Parse the response to determine the next agent
        next_agent = "coordinator"  # Default to self
        instructions = ""
        
        for line in assistant_message.split("\n"):
            if line.startswith("NEXT_AGENT:"):
                agent_name = line.split(":", 1)[1].strip().lower()
                if agent_name in ["researcher", "analyst", "writer", "final"]:
                    next_agent = agent_name
            elif line.startswith("INSTRUCTIONS:"):
                instructions = line.split(":", 1)[1].strip()

        # Add the coordinator's message to the conversation
        state_obj.conversation.append(AgentMessage(
            role="coordinator",
            content=assistant_message
        ))
        
        # Update the current agent and instructions
        state_obj.current_instructions = instructions
        if next_agent == "final":
            state_obj.final_response = instructions
            next_agent = "end"
        
        state_obj.current_agent = next_agent
        
    except Exception as e:
        logger.error(f"Error in coordinator node: {str(e)}", exc_info=True)
        state_obj.errors.append({
            "node": "coordinator",
            "error": str(e),
            "timestamp": time.time()
        })
    
    return state_obj.model_dump()

async def researcher_node(state: Dict[str, Any], agent, google_api_key: str) -> Dict[str, Any]:
    """Process the state through the researcher agent."""
    logger.info("--- Running Researcher Node ---")
    state_obj = MultiAgentState.model_validate(state)
    
    # Get instructions from the state
    instructions = state_obj.current_instructions or ""
    logger.info(f"Researcher received instructions: {instructions}")
    
    # Prepare the messages for the agent
    messages = [{"role": "user", "content": f"Instructions: {instructions}\n\nUser query: {state_obj.query}"}]
    
    # Invoke the agent
    try:
        response = await agent.ainvoke({"messages": messages})
        assistant_message = next((msg.content for msg in response["messages"] if msg.type == "ai"), "")
        
        # Add the researcher's message to the conversation
        state_obj.conversation.append(AgentMessage(
            role="researcher",
            content=assistant_message
        ))
        
        # Return to the coordinator
        state_obj.current_agent = "coordinator"
        
    except Exception as e:
        logger.error(f"Error in researcher node: {str(e)}", exc_info=True)
        state_obj.errors.append({
            "node": "researcher",
            "error": str(e),
            "timestamp": time.time()
        })
        state_obj.current_agent = "coordinator"
    
    return state_obj.model_dump()

async def analyst_node(state: Dict[str, Any], agent, google_api_key: str) -> Dict[str, Any]:
    """Process the state through the analyst agent."""
    logger.info("Running analyst node")
    state_obj = MultiAgentState.model_validate(state)
    
    # Get instructions from the state
    instructions = state_obj.current_instructions or ""
    
    # Get researcher findings if available
    researcher_messages = [msg for msg in state_obj.conversation if msg.role == "researcher"]
    researcher_findings = ""
    if researcher_messages:
        researcher_findings = f"\n\nResearcher findings:\n{researcher_messages[-1].content}"
    
    # Prepare the messages for the agent
    messages = [{"role": "user", "content": f"Instructions: {instructions}\n\nUser query: {state_obj.query}{researcher_findings}"}]
    
    # Invoke the agent
    try:
        response = await agent.ainvoke({"messages": messages})
        assistant_message = next((msg.content for msg in response["messages"] if msg.type == "ai"), "")
        
        # Add the analyst's message to the conversation
        state_obj.conversation.append(AgentMessage(
            role="analyst",
            content=assistant_message
        ))
        
        # Return to the coordinator
        state_obj.current_agent = "coordinator"
        
    except Exception as e:
        logger.error(f"Error in analyst node: {str(e)}", exc_info=True)
        state_obj.errors.append({
            "node": "analyst",
            "error": str(e),
            "timestamp": time.time()
        })
        state_obj.current_agent = "coordinator"
    
    return state_obj.model_dump()

async def writer_node(state: Dict[str, Any], agent, google_api_key: str) -> Dict[str, Any]:
    """Process the state through the writer agent."""
    logger.info("--- WRITER ---")
    state_obj = MultiAgentState.model_validate(state)
    
    # Get instructions from the state
    instructions = state_obj.current_instructions or ""
    
    # Get researcher and analyst findings if available
    researcher_messages = [msg for msg in state_obj.conversation if msg.role == "researcher"]
    analyst_messages = [msg for msg in state_obj.conversation if msg.role == "analyst"]
    
    additional_context = ""
    if researcher_messages:
        additional_context += f"\n\nResearcher findings:\n{researcher_messages[-1].content}"
    if analyst_messages:
        additional_context += f"\n\nAnalyst insights:\n{analyst_messages[-1].content}"
    
    # Prepare the messages for the agent
    messages = [{"role": "user", "content": f"Instructions: {instructions}\n\nUser query: {state_obj.query}{additional_context}"}]
    
    # Invoke the agent
    try:
        response = await agent.ainvoke({"messages": messages})
        assistant_message = next((msg.content for msg in response["messages"] if msg.type == "ai"), "")
        
        # Add the writer's message to the conversation
        state_obj.conversation.append(AgentMessage(
            role="writer",
            content=assistant_message
        ))
        
        # Return to the coordinator
        state_obj.current_agent = "coordinator"
        
    except Exception as e:
        logger.error(f"Error in writer node: {e}")
        state_obj.errors.append({
            "node": "writer",
            "error": str(e),
            "timestamp": time.time()
        })
        state_obj.current_agent = "coordinator"
    
    return state_obj.model_dump()

# Main function to run the multi-agent system
async def run_multi_agent_system(query: str):
    """Run the multi-agent system with MCP communication."""
    logger.info(f"Running multi-agent system for query: {query}")
    
    # Get the Google API key from environment
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        logger.error("GOOGLE_API_KEY not found in environment variables")
        print("Please set your GOOGLE_API_KEY in the .env file")
        return {"error": "API key not found"}
    
    # Get the path to the MCP server script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    server_script = os.path.join(current_dir, "01_mcp_server.py")
    server_url = "http://localhost:8000/sse"
    # Create an MCP client that connects to our server
    logger.info(f"Connecting to MCP server: {server_script}")
    client = MultiServerMCPClient(
        {
            "knowledge_tools": {
                "url": server_url,
                "transport": "sse"
            }
        }
    )
    
    # Get the tools from the MCP server
    logger.info("Fetching tools from MCP server...")
    mcp_tools = await client.get_tools()
    
    # Create the agents
    logger.info("Creating agents...")
    coordinator_agent = create_coordinator_agent(google_api_key)
    researcher_agent = create_researcher_agent(google_api_key, mcp_tools)
    analyst_agent = create_analyst_agent(google_api_key, mcp_tools)
    writer_agent = create_writer_agent(google_api_key)
    
    # Create the workflow graph
    workflow = StateGraph(MultiAgentState)
    
    # Add nodes to the graph
    workflow.add_node("coordinator", functools.partial(coordinator_node, agent=coordinator_agent, google_api_key=google_api_key))
    workflow.add_node("researcher", functools.partial(researcher_node, agent=researcher_agent, google_api_key=google_api_key))
    workflow.add_node("analyst", functools.partial(analyst_node, agent=analyst_agent, google_api_key=google_api_key))
    workflow.add_node("writer", functools.partial(writer_node, agent=writer_agent, google_api_key=google_api_key))
    
    # Add conditional edges based on the current_agent field
    workflow.add_conditional_edges(
        "coordinator",
        lambda state: state.current_agent,
        {
            "researcher": "researcher",
            "analyst": "analyst",
            "writer": "writer",
            "end": END
        }
    )
    
    # Add edges from specialist agents back to coordinator
    workflow.add_edge("researcher", "coordinator")
    workflow.add_edge("analyst", "coordinator")
    workflow.add_edge("writer", "coordinator")
    
    # Set the entry point
    workflow.set_entry_point("coordinator")
    
    # Compile the graph
    graph = workflow.compile()
    
    # Run the graph
    initial_state = {
        "query": query,
        "current_agent": "coordinator"
    }
    
    final_state = await graph.ainvoke(initial_state)
    
    logger.info(f"Final response: {final_state.get('final_response', 'No final response generated.')}")
    
    return final_state

async def main():
    """Run the multi-agent MCP example."""
    print("=" * 80)
    print("Multi-Agent System with MCP Communication")
    print("=" * 80)
    
    # Get user input
    query = input("\nEnter your query: ")
    
    print("\nProcessing with multi-agent system...")
    result = await run_multi_agent_system(query)
    
    # Display the final response
    print("\n" + "=" * 80)
    print("Final Response:")
    print(result.get("final_response", "No response generated."))
    print("=" * 80)
    
    # Display the conversation
    if "conversation" in result:
        print("\nAgent Conversation:")
        for i, message in enumerate(result["conversation"], 1):
            role = message["role"]
            content = message["content"]
            print(f"\n--- Message {i} from {role.capitalize()} ---")
            print(content)

if __name__ == "__main__":
    asyncio.run(main())
