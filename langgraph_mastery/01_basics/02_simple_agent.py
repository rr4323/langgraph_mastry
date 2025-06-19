"""
LangGraph Basics: Creating Your First Agent
==========================================

This script demonstrates how to create a simple agent using LangGraph
with Google's Generative AI model.
"""

import os
import sys
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

# Add the parent directory to the path so we can import from the root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables from .env file
load_dotenv()

def create_simple_agent():
    """Create a simple LangGraph agent using Google's Generative AI."""
    print("Creating a simple LangGraph agent with Google's Generative AI...")
    
    # Initialize the chat model with Google's Gemini Pro
    model = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.7,
        convert_system_message_to_human=True
    )
    
    # Create a simple agent using LangGraph's create_react_agent
    # This is a basic agent without tools, just for conversation
    agent = create_react_agent(
        model=model,
        tools=[],  # No tools for this simple agent
        prompt="You are a helpful assistant that explains LangGraph concepts clearly and concisely."
    )
    
    return agent

def interact_with_agent(agent):
    """Interact with the agent by sending messages and receiving responses."""
    print("\n" + "=" * 50)
    print("Interacting with your LangGraph Agent")
    print("=" * 50)
    print("(Type 'exit' to end the conversation)")
    
    while True:
        # Get user input
        user_input = input("\nYou: ")
        
        if user_input.lower() == 'exit':
            print("\nEnding conversation.")
            break
        
        # Create a message for the agent
        message = HumanMessage(content=user_input)
        
        # Invoke the agent with the message
        response = agent.invoke({"messages": [message]})
        
        # Print the agent's response
        print(f"\nAgent: {response['messages'][-1].content}")

def main():
    """Main function to create and interact with a simple agent."""
    print("=" * 50)
    print("Creating Your First LangGraph Agent")
    print("=" * 50)
    
    # Create a simple agent
    agent = create_simple_agent()
    
    # Interact with the agent
    interact_with_agent(agent)

if __name__ == "__main__":
    main()
