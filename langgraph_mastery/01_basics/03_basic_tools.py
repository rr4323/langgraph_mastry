"""
LangGraph Basics: Adding Tools to Your Agent
===========================================

This script demonstrates how to add tools to your LangGraph agent
using Google's Generative AI model.
"""

import os
import sys
import datetime
import requests
from dotenv import load_dotenv
from typing import Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent


# Add the parent directory to the path so we can import from the root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables from .env file
load_dotenv()

# Define some simple tools for our agent
def get_current_time() -> str:
    """Get the current date and time."""
    now = datetime.datetime.now()
    return f"The current date and time is {now.strftime('%Y-%m-%d %H:%M:%S')}"

def get_weather(location: str) -> str:
    """Get the current weather for a location (simulated)."""
    # This is a simulated weather function
    # In a real application, you would call a weather API
    weather_conditions = ["sunny", "cloudy", "rainy", "snowy", "windy"]
    temperatures = range(0, 35)
    
    # Simple hash function to make the weather consistent for the same location
    import hashlib
    hash_value = int(hashlib.md5(location.encode()).hexdigest(), 16)
    weather_index = hash_value % len(weather_conditions)
    temp_index = hash_value % len(temperatures)
    
    return f"The weather in {location} is currently {weather_conditions[weather_index]} with a temperature of {temperatures[temp_index]}°C"

def search_web(query: str) -> str:
    """Simulate a web search (for demonstration purposes)."""
    # This is a simulated web search function
    # In a real application, you would call a search API
    return f"Here are the simulated search results for: '{query}'. " \
           f"This is a placeholder for real search results that would be returned by an actual search API."

def create_agent_with_tools():
    """Create a LangGraph agent with tools using Google's Generative AI."""
    print("Creating a LangGraph agent with tools using Google's Generative AI...")
    
    # Initialize the chat model with Google's Gemini Pro
    model = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.7,
        convert_system_message_to_human=True
    )
    
    # Define the tools we want to give our agent
    tools = [
        get_current_time,
        get_weather,
        search_web
    ]
    
    # Create an agent with tools using LangGraph's create_react_agent
    # This uses the ReAct framework (Reasoning and Acting)
    agent = create_react_agent(
        model=model,
        tools=tools,
        prompt="""You are a helpful assistant with access to tools.
Use these tools to answer the user's questions as best you can.
Always think step by step about which tool would be most appropriate to use.
""",
        debug=True
    )
    
    return agent

def interact_with_agent(agent):
    """Interact with the agent by sending messages and receiving responses."""
    print("\n" + "=" * 50)
    print("Interacting with your LangGraph Agent with Tools")
    print("=" * 50)
    print("Available tools: get_current_time, get_weather, search_web")
    print("(Type 'exit' to end the conversation)")
    
    while True:
        # Get user input
        user_input = input("\nYou: ")
        
        if user_input.lower() == 'exit':
            print("\nEnding conversation.")
            break
        
        # Invoke the agent with the message
        response = agent.invoke(
            {"messages": [{"role": "user", "content": user_input}]}
        )
        
        # Print the agent's response
        print(f"\nAgent: {response['messages'][-1].content}")

def main():
    """Main function to create and interact with an agent with tools."""
    print("=" * 50)
    print("Creating a LangGraph Agent with Tools")
    print("=" * 50)
    
    # Create an agent with tools
    agent = create_agent_with_tools()
    
    # Interact with the agent
    interact_with_agent(agent)

if __name__ == "__main__":
    main()
