"""
LangGraph Intermediate: Custom Prompts
=====================================

This script demonstrates how to create and use custom prompts with LangGraph
using Google's Generative AI model.
"""

import os
import sys
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState

# Add the parent directory to the path so we can import from the root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables from .env file
load_dotenv()

# Define some simple tools for our agent
def get_random_fact() -> str:
    """Get a random fact about LangGraph."""
    facts = [
        "LangGraph is built on top of LangChain.",
        "LangGraph allows you to create stateful, multi-actor applications with LLMs.",
        "LangGraph uses a directed graph to represent the flow of information.",
        "LangGraph was created to help build agentic systems.",
        "LangGraph supports various types of memory and persistence."
    ]
    import random
    return random.choice(facts)

def search_documentation(query: str) -> str:
    """Simulate searching the LangGraph documentation."""
    # This is a simulated documentation search
    documentation = {
        "agent": "Agents are autonomous entities that can perform tasks using LLMs and tools.",
        "graph": "Graphs in LangGraph represent the flow of information and control in an agent system.",
        "memory": "Memory in LangGraph allows agents to remember previous interactions.",
        "tool": "Tools are functions that agents can use to interact with external systems.",
        "prompt": "Prompts instruct the LLM how to behave in LangGraph.",
        "state": "State in LangGraph represents the current status of an agent or workflow."
    }
    
    # Simple keyword matching
    for keyword, content in documentation.items():
        if keyword.lower() in query.lower():
            return content
    
    return "No specific documentation found for that query."

# Define different types of custom prompts

# 1. Static string prompt
static_prompt = """You are a LangGraph expert assistant.
You help users understand LangGraph concepts and implementation details.
Use your tools when appropriate to provide accurate information.
Always be concise and clear in your explanations.
"""

# 2. Static message list prompt
static_message_list_prompt = [
    SystemMessage(content="""You are a LangGraph expert assistant.
You help users understand LangGraph concepts and implementation details.
Use your tools when appropriate to provide accurate information.
Always be concise and clear in your explanations.""")
]

# 3. Dynamic prompt function
def dynamic_prompt(state: AgentState, config: RunnableConfig) -> List[AnyMessage]:
    """Create a dynamic prompt based on state and config."""
    # Get user preferences from config (if available)
    user_preferences = config.get("configurable", {})
    user_name = user_preferences.get("user_name", "User")
    expertise_level = user_preferences.get("expertise_level", "intermediate")
    
    # Customize the prompt based on user expertise level
    if expertise_level == "beginner":
        system_content = f"""You are a LangGraph tutor for beginners.
Hello {user_name}! I'll explain LangGraph concepts in simple terms.
I'll avoid technical jargon and provide clear, step-by-step explanations.
I'll use analogies and examples to make concepts easier to understand.
"""
    elif expertise_level == "intermediate":
        system_content = f"""You are a LangGraph advisor for intermediate users.
Hello {user_name}! I'll provide detailed explanations of LangGraph concepts.
I'll assume you have basic knowledge of LLMs and Python programming.
I'll focus on practical implementation details and best practices.
"""
    else:  # advanced
        system_content = f"""You are a LangGraph consultant for advanced users.
Hello {user_name}! I'll provide in-depth technical information about LangGraph.
I'll discuss advanced patterns, optimizations, and implementation details.
I'll assume you have strong knowledge of LLMs, Python, and software architecture.
"""
    
    # Create the system message
    system_message = SystemMessage(content=system_content)
    
    # Return the complete prompt (system message + conversation history)
    return [system_message] + state["messages"]

def create_agent_with_custom_prompt(prompt_type: str):
    """Create a LangGraph agent with a custom prompt."""
    print(f"Creating a LangGraph agent with a {prompt_type} prompt...")
    
    # Initialize the chat model with Google's Gemini Pro
    model = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.7,
        convert_system_message_to_human=True
    )

    
    # Define the tools
    tools = [
        get_random_fact,
        search_documentation
    ]
    
    # Select the prompt based on the prompt_type
    if prompt_type == "static_string":
        prompt = static_prompt
    elif prompt_type == "static_message_list":
        prompt = static_message_list_prompt
    elif prompt_type == "dynamic":
        prompt = dynamic_prompt
    else:
        raise ValueError(f"Unknown prompt type: {prompt_type}")
    
    # Create the agent
    agent = create_react_agent(
        model=model,
        tools=tools,
        prompt=prompt
    )
    
    return agent

def interact_with_agent(agent, prompt_type: str):
    """Interact with the agent by sending messages and receiving responses."""
    print("\n" + "=" * 50)
    print(f"Interacting with your LangGraph Agent ({prompt_type} prompt)")
    print("=" * 50)
    print("Available tools: get_random_fact, search_documentation")
    print("(Type 'exit' to end the conversation)")
    
    # For dynamic prompts, we need to set up config
    config = {}
    if prompt_type == "dynamic":
        user_name = input("\nWhat's your name? ")
        expertise_options = ["beginner", "intermediate", "advanced"]
        print("\nSelect your expertise level:")
        for i, level in enumerate(expertise_options):
            print(f"{i+1}. {level}")
        
        while True:
            try:
                choice = int(input("\nEnter your choice (1-3): "))
                if 1 <= choice <= 3:
                    expertise_level = expertise_options[choice-1]
                    break
                else:
                    print("Please enter a number between 1 and 3.")
            except ValueError:
                print("Please enter a valid number.")
        
        config = {
            "configurable": {
                "user_name": user_name,
                "expertise_level": expertise_level
            }
        }
        print(f"\nCustomizing experience for {user_name} at {expertise_level} level...")
    
    while True:
        # Get user input
        user_input = input("\nYou: ")
        
        if user_input.lower() == 'exit':
            print("\nEnding conversation.")
            break
        
        # Invoke the agent with the message
        if prompt_type == "dynamic":
            response = agent.invoke(
                {"messages": [{"role": "user", "content": user_input}]},
                config=config
            )
        else:
            response = agent.invoke(
                {"messages": [{"role": "user", "content": user_input}]}
            )
        
        # Print the agent's response
        print(f"\nAgent: {response['messages'][-1].content}")

def main():
    """Main function to demonstrate custom prompts."""
    print("=" * 50)
    print("Custom Prompts in LangGraph")
    print("=" * 50)
    
    # Let the user choose a prompt type
    prompt_types = ["static_string", "static_message_list", "dynamic"]
    print("\nSelect a prompt type to explore:")
    for i, prompt_type in enumerate(prompt_types):
        print(f"{i+1}. {prompt_type}")
    
    while True:
        try:
            choice = int(input("\nEnter your choice (1-3): "))
            if 1 <= choice <= 3:
                selected_prompt_type = prompt_types[choice-1]
                break
            else:
                print("Please enter a number between 1 and 3.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Create an agent with the selected prompt type
    agent = create_agent_with_custom_prompt(selected_prompt_type)
    
    # Interact with the agent
    interact_with_agent(agent, selected_prompt_type)

if __name__ == "__main__":
    main()
