"""
LangGraph Intermediate: Memory and Persistence
============================================

This script demonstrates how to add memory to LangGraph agents
for stateful conversations using Google's Generative AI model.
"""

import os
import sys
import uuid
from typing import Dict, Any
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver

# Add the parent directory to the path so we can import from the root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables from .env file
load_dotenv()

# Define some simple tools for our agent
def get_current_weather(location: str) -> str:
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

def create_agent_with_memory():
    """Create a LangGraph agent with memory."""
    print("Creating a LangGraph agent with memory...")
    
    # Initialize the chat model with Google's Gemini Pro
    model = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.7,
        convert_system_message_to_human=True
    )
    
    # Create an in-memory checkpointer for persistence
    checkpointer = InMemorySaver()
    
    # Define the tools
    tools = [get_current_weather]
    
    # Create the agent with memory
    agent = create_react_agent(
        model=model,
        tools=tools,
        prompt="""You are a helpful assistant with memory.
You can remember previous parts of the conversation.
Use your memory to provide consistent and contextually relevant responses.
If the user refers to something mentioned earlier, use your memory to recall it.
Use your tools when appropriate to provide accurate information.
""",
        checkpointer=checkpointer  # This enables memory
    )
    
    return agent,checkpointer

def interact_with_agent(agent):
    """Interact with the agent by sending messages and receiving responses."""
    print("\n" + "=" * 50)
    print("Interacting with your LangGraph Agent with Memory")
    print("=" * 50)
    print("Available tools: get_current_weather")
    print("(Type 'exit' to end the conversation)")
    print("(Type 'new' to start a new conversation)")
    
    # Generate a unique thread ID for this conversation
    thread_id = str(uuid.uuid4())
    print(f"\nStarting conversation with thread ID: {thread_id}")
    
    while True:
        # Get user input
        user_input = input("\nYou: ")
        
        if user_input.lower() == 'exit':
            print("\nEnding conversation.")
            break
        
        if user_input.lower() == 'new':
            thread_id = str(uuid.uuid4())
            print(f"\nStarting new conversation with thread ID: {thread_id}")
            continue
        
        # Configure the agent with the thread ID
        config = {"configurable": {"thread_id": thread_id}}
        
        # Invoke the agent with the message
        response = agent.invoke(
            {"messages": [{"role": "user", "content": user_input}]},
            config=config
        )
        
        # Print the agent's response
        print(f"\nAgent: {response['messages'][-1].content}")

def demonstrate_memory_persistence():
    """Demonstrate memory persistence across multiple conversations."""
    print("\n" + "=" * 50)
    print("Demonstrating Memory Persistence")
    print("=" * 50)
    
    # Create an agent with memory
    agent,m = create_agent_with_memory()
    
    # Create a fixed thread ID for this demonstration
    thread_id = "demo-thread-123"
    config = {"configurable": {"thread_id": thread_id}}
    
    # First conversation turn
    print("\nFirst turn: Asking about weather in London")
    response1 = agent.invoke(
        {"messages": [{"role": "user", "content": "What's the weather like in London?"}]},
        config=config
    )
    print(f"Agent: {response1['messages'][-1].content}")
    
    # Second conversation turn (referring to the first)
    print("\nSecond turn: Asking about weather in Paris")
    response2 = agent.invoke(
        {"messages": [{"role": "user", "content": "What about Paris?"}]},
        config=config
    )
    print(f"Agent: {response2['messages'][-1].content}")
    
    # Third conversation turn (referring to previous information)
    print("\nThird turn: Asking which city is warmer")
    response3 = agent.invoke(
        {"messages": [{"role": "user", "content": "Which city is warmer based on your information?"}]},
        config=config
    )
    print(f"Agent: {response3['messages'][-1].content}")
    
    # Show that a different thread ID creates a separate conversation
    new_thread_id = "new-demo-thread-456"
    new_config = {"configurable": {"thread_id": new_thread_id}}
    
    print("\nNew conversation with different thread ID")
    new_response = agent.invoke(
        {"messages": [{"role": "user", "content": "What cities have we discussed the weather for?"}]},
        config=new_config
    )
    print(f"Agent: {new_response['messages'][-1].content}")
    
    # Return to the original conversation
    print("\nReturning to original conversation")
    final_response = agent.invoke(
        {"messages": [{"role": "user", "content": "Summarize what we've discussed about weather so far."}]},
        config=config
    )
    print(f"Agent: {final_response['messages'][-1].content}")
    config = {"configurable": {"thread_id": thread_id}}
    state = m.get(config)

    print(f"\n🧠 After turn, memory for thread {thread_id}:")
    print(state)

def main():
    """Main function to demonstrate memory and persistence."""
    print("=" * 50)
    print("Memory and Persistence in LangGraph")
    print("=" * 50)
    
    # Let the user choose a mode
    print("\nSelect a mode:")
    print("1. Interactive conversation with memory")
    print("2. Automated demonstration of memory persistence")
    
    while True:
        try:
            choice = int(input("\nEnter your choice (1-2): "))
            if choice == 1:
                agent,_ = create_agent_with_memory()
                interact_with_agent(agent)
                break
            elif choice == 2:
                demonstrate_memory_persistence()
                break
            else:
                print("Please enter 1 or 2.")
        except ValueError:
            print("Please enter a valid number.")

if __name__ == "__main__":
    main()
