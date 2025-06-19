"""
MCP Client for an SSE-based LangGraph Server

This script connects to a running MCP server via HTTP, fetches its tools,
and uses them in a modern LangGraph agent created with `create_agent_executor`.

To run this:
1. Start the server in one terminal: `python langgraph_mastery/03_advanced/01_mcp_server.py`
2. Run this client in another terminal: `python langgraph_mastery/03_advanced/02_mcp_client.py`
"""

import os
import sys
import logging
import asyncio
from typing import Dict, List, Any
from dotenv import load_dotenv

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
from langchain_core.messages import HumanMessage, AIMessage

async def main():
    """Run the MCP client example."""
    logger.info("Starting MCP client example...")
    
    # Get the Google API key from environment
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        logger.error("GOOGLE_API_KEY not found in environment variables")
        print("Please set your GOOGLE_API_KEY in the .env file")
        return
    
    # Define the server URL
    server_url = "http://localhost:8000/sse"

    # Create an MCP client that connects to our HTTP server
    logger.info(f"Connecting to MCP server at: {server_url}")
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
    tools = await client.get_tools()
    
    # Print the available tools
    logger.info(f"Available tools: {[tool.name for tool in tools]}")
    
    # Create a LangGraph agent with the MCP tools
    logger.info("Creating LangGraph agent with MCP tools...")
    model = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.7,
        convert_system_message_to_human=True
    )
    agent = create_react_agent(
        model=model,
        tools=tools,
        prompt="""
You are a helpful assistant with access to tools such as entity extraction, sentiment analysis, and fact checking.
When a user's query requires analysis or external knowledge, use the appropriate tool.
"""
    )
    
    # Example queries to test the agent
    example_queries = [
        "What are the company rajeev ranjan work on?",
        "Analyze the sentiment of rajeev ranjan work",
        "Extract entities from this text: Acme Corp is developing a new AI system in San Francisco using Python and LangGraph."
    ]
    
    # Process each query
    for i, query in enumerate(example_queries, 1):
        print("\n" + "=" * 80)
        print(f"Example {i}: {query}")
        print("=" * 80)
        
        # Invoke the agent
        logger.info(f"Processing query: {query}")
        # Properly wrap the query in a HumanMessage
        response = await agent.ainvoke({"messages": [HumanMessage(content=query)]})

        print("\nAgent Response:")

        # If response is a dict with 'messages', handle carefully
        messages = response.get("messages", []) if isinstance(response, dict) else [response]

        for message in messages:
            # Handle message objects correctly
            if isinstance(message, AIMessage):
                print(message.content)
        
        # Clean up
        logger.info("Example completed, shutting down...")

if __name__ == "__main__":
    asyncio.run(main())
