"""
LangGraph Basics: Environment Setup and Configuration
====================================================

This script demonstrates how to set up the environment for LangGraph
and configure Google's Generative AI model for use with LangGraph.
"""

import os
import sys
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

# Add the parent directory to the path so we can import from the root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables from .env file
load_dotenv()

# Check if the Google API key is set
if not os.getenv("GOOGLE_API_KEY"):
    print("Error: GOOGLE_API_KEY environment variable not set.")
    print("Please create a .env file based on .env.example and add your Google API key.")
    sys.exit(1)

def setup_google_genai():
    """Set up and test the Google Generative AI model."""
    print("Setting up Google Generative AI model...")
    
    # Initialize the chat model with Google's Gemini Pro
    chat_model = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.7,
        convert_system_message_to_human=True
    )
    
    # Test the model with a simple query
    messages = [
        SystemMessage(content="You are a helpful AI assistant."),
        HumanMessage(content="Explain what LangGraph is in one sentence.")
    ]
    
    print("\nTesting connection to Google's Generative AI...")
    response = chat_model.invoke(messages)
    print(f"\nResponse: {response.content}")
    
    return chat_model

def main():
    """Main function to run the setup."""
    print("=" * 50)
    print("LangGraph Environment Setup")
    print("=" * 50)
    
    # Set up and test Google Generative AI
    model = setup_google_genai()
    
    print("\n" + "=" * 50)
    print("Setup complete! Your environment is ready for LangGraph development.")
    print("=" * 50)
    
    return model

if __name__ == "__main__":
    main()
