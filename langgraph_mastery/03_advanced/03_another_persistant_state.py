"""
LangGraph Advanced: Persistence and State Management (Corrected)
==============================================================

This script demonstrates how to implement persistence and state management
in LangGraph applications for long-running processes.
"""

import os
import sys
import time
import uuid
import json
from typing import Annotated, Dict, List, Literal, TypedDict, Union, Any, Optional
from enum import Enum
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
import sqlite3
import spacy
from typing import Optional

nlp = spacy.load("en_core_web_sm")

# Load environment variables from .env file
load_dotenv()


TOPIC_KEYWORDS = {
    "music": ["song", "band", "album", "artist", "playlist", "concert"],
    "food": ["food", "recipe", "restaurant", "cuisine", "meal", "cook"],
    "books": ["book", "novel", "author", "literature", "read", "story"],
    "movies": ["movie", "film", "cinema", "director", "actor", "watch"],
    "tech": ["AI", "software", "hardware", "technology", "coding", "programming"],
    "finance": ["money", "investment", "stock", "crypto", "loan", "budget"],
    "health": ["health", "fitness", "exercise", "mental", "diet", "sleep"],
}

# Define our state for the persistent workflow
class ConversationState(TypedDict):
    """State for our persistent conversation workflow."""
    messages: Annotated[List[BaseMessage], add_messages]
    context: Dict[str, Any]
    current_topic: str
    session_data: Dict[str, Any]
    user_input: str

class UserProfile(BaseModel):
    """User profile information."""
    name: str = Field(description="User's name")
    interests: List[str] = Field(description="User's interests")
    preferences: Dict[str, str] = Field(description="User's preferences")
    
    def update_interest(self, interest: str):
        """Add an interest if it doesn't exist."""
        if interest not in self.interests:
            self.interests.append(interest)
    
    def update_preference(self, key: str, value: str):
        """Update a preference."""
        self.preferences[key] = value

class SessionData(BaseModel):
    """Session data for tracking conversation state."""
    session_id: str = Field(description="Unique session identifier")
    start_time: float = Field(description="Session start time")
    last_active: float = Field(description="Last activity time")
    interaction_count: int = Field(description="Number of interactions in this session")
    topics_discussed: List[str] = Field(description="Topics discussed in this session")
    
    def update_activity(self):
        """Update the last activity time."""
        self.last_active = time.time()
    
    def increment_interaction(self):
        """Increment the interaction count."""
        self.interaction_count += 1
    
    def add_topic(self, topic: str):
        """Add a topic if it doesn't exist."""
        if topic not in self.topics_discussed:
            self.topics_discussed.append(topic)

# File paths for persistence
PROFILES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
SQLITE_DB_PATH = os.path.join(PROFILES_DIR, "langgraph_sessions.db")

# Create the data directory if it doesn't exist
os.makedirs(PROFILES_DIR, exist_ok=True)

def get_profile_path(user_id: str) -> str:
    """Get the path to a user's profile file."""
    return os.path.join(PROFILES_DIR, f"user_{user_id}.json")

def load_user_profile(user_id: str) -> UserProfile:
    """Load a user profile from disk or create a new one."""
    profile_path = get_profile_path(user_id)
    
    if os.path.exists(profile_path):
        try:
            with open(profile_path, "r") as f:
                data = json.load(f)
            return UserProfile(**data)
        except Exception as e:
            print(f"Error loading profile: {e}")
            return UserProfile(
                name=f"User_{user_id[:8]}",
                interests=[],
                preferences={}
            )
    else:
        # Create a default profile
        return UserProfile(
            name=f"User_{user_id[:8]}",
            interests=[],
            preferences={}
        )

def save_user_profile(user_id: str, profile: UserProfile):
    """Save a user profile to disk."""
    profile_path = get_profile_path(user_id)
    
    try:
        with open(profile_path, "w") as f:
            json.dump(profile.model_dump(), f, indent=2)
    except Exception as e:
        print(f"Error saving profile: {e}")

# Tools for the agent
@tool
def get_user_info(user_id: str) -> str:
    """
    Get information about the user.
    
    Args:
        user_id: The user's ID
    
    Returns:
        str: Information about the user
    """
    profile = load_user_profile(user_id)
    
    return f"""User Profile:
- Name: {profile.name}
- Interests: {', '.join(profile.interests) if profile.interests else 'No interests recorded yet'}
- Preferences: {json.dumps(profile.preferences, indent=2) if profile.preferences else 'No preferences recorded yet'}"""

@tool
def update_user_interest(user_id: str, interest: str) -> str:
    """
    Update a user's interests.
    
    Args:
        user_id: The user's ID
        interest: The interest to add
    
    Returns:
        str: Confirmation message
    """
    profile = load_user_profile(user_id)
    profile.update_interest(interest)
    save_user_profile(user_id, profile)
    
    return f"Added '{interest}' to user's interests."

@tool
def update_user_preference(user_id: str, key: str, value: str) -> str:
    """
    Update a user's preference.
    
    Args:
        user_id: The user's ID
        key: The preference key
        value: The preference value
    
    Returns:
        str: Confirmation message
    """
    profile = load_user_profile(user_id)
    profile.update_preference(key, value)
    save_user_profile(user_id, profile)
    
    return f"Updated user's preference: {key} = {value}"

@tool
def get_session_info(session_id: str, start_time: float, interaction_count: int, topics_discussed: List[str]) -> str:
    """
    Get information about the current session.
    
    Args:
        session_id: The session ID
        start_time: Session start time
        interaction_count: Number of interactions
        topics_discussed: List of topics discussed
    
    Returns:
        str: Information about the session
    """
    # Calculate session duration
    duration_seconds = time.time() - start_time
    hours, remainder = divmod(duration_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    duration_str = ""
    if hours > 0:
        duration_str += f"{int(hours)} hours, "
    if minutes > 0:
        duration_str += f"{int(minutes)} minutes, "
    duration_str += f"{int(seconds)} seconds"
    
    return f"""Session Information:
- Session ID: {session_id}
- Duration: {duration_str}
- Interactions: {interaction_count}
- Topics Discussed: {', '.join(topics_discussed) if topics_discussed else 'No topics recorded yet'}"""

def should_call_tool(state: ConversationState) -> Literal["call_model", "call_tools"]:
    """Determine whether to call the model or tools."""
    messages = state["messages"]
    last_message = messages[-1]
    
    # If the last message is from the assistant and has tool calls, execute tools
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "call_tools"
    return "call_model"

def detect_topic(text: str) -> Optional[str]:
    doc = nlp(text.lower())

    # Gather candidate keywords
    keywords = set([chunk.root.text for chunk in doc.noun_chunks])
    keywords.update(ent.text.lower() for ent in doc.ents)
    keywords.update(token.lemma_ for token in doc if token.pos_ in {"NOUN", "PROPN"} and not token.is_stop)

    # Match keywords against topic buckets
    topic_scores = {}
    for topic, topic_words in TOPIC_KEYWORDS.items():
        match_score = sum(1 for word in topic_words if word in keywords)
        topic_scores[topic] = match_score

    best_topic = max(topic_scores, key=topic_scores.get)
    return best_topic if topic_scores[best_topic] > 0 else "general"

def call_model_node(state: ConversationState) -> ConversationState:
    """Call the LLM model to generate a response."""
    print("🤖 Assistant is thinking...")
    
    messages = state["messages"]
    context = state["context"]
    session_data = state["session_data"]
    user_id = context.get("user_id", "default_user")
    
    # Update session data
    if session_data:
        session = SessionData(**session_data)
        session.update_activity()
        session.increment_interaction()
        
        # Extract topic from the last user message
        last_user_message = None
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                last_user_message = msg
                break
        
        if last_user_message:
            topic = detect_topic(last_user_message.content)
            
            session.add_topic(topic)
            state["current_topic"] = topic
        
        state["session_data"] = session.model_dump()
    
    # Create the model
    model = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.7,
        convert_system_message_to_human=True
    )
    
    # Bind tools to the model
    tools = [get_user_info, update_user_interest, update_user_preference, get_session_info]
    model_with_tools = model.bind_tools(tools)
    
    # Create system message with context
    system_message = SystemMessage(content=f"""You are a helpful assistant with persistent memory.
You can remember information about the user across conversations.

Current user ID: {user_id}
Current topic: {state.get("current_topic", "general")}

Use the available tools to:
1. Get information about the user when relevant
2. Update the user's interests based on the conversation
3. Update the user's preferences when they express likes/dislikes
4. Get information about the current session when asked

Be helpful, personalized, and contextually aware. Use tools when appropriate to provide better service.""")
    
    # Prepare messages for the model
    model_messages = [system_message] + messages
    
    try:
        # Call the model
        response = model_with_tools.invoke(model_messages)
        print(response)
        return {
            **state,
            "messages": [response]
        }
    except Exception as e:
        error_response = AIMessage(content=f"I apologize, but I encountered an error: {str(e)}")
        return {
            **state,
            "messages": [error_response]
        }

def call_tools_node(state: ConversationState) -> ConversationState:
    """Execute tool calls."""
    print("🔧 Using tools...")
    
    # Create tool node
    tools = [get_user_info, update_user_interest, update_user_preference, get_session_info]
    tool_node = ToolNode(tools)
    
    # Execute tools
    result = tool_node.invoke(state)
    
    return result

def human_input_node(state: ConversationState) -> ConversationState:
    """Get input from the human user."""
    print("\n" + "=" * 50)
    
    # Display the last assistant message if it exists
    messages = state["messages"]
    if messages:
        last_message = messages[-1]
        if isinstance(last_message, AIMessage):
            print(f"🤖 Assistant: {last_message.content}")
    
    # Get user input
    user_input = input("\n👤 You (or type 'exit' to quit): ").strip()
    
    if user_input.lower() in ['exit', 'quit', 'bye']:
        print("👋 Goodbye! Your conversation has been saved.")
        return {
            **state,
            "user_input": "exit"
        }
    
    # Add user message to the conversation
    user_message = HumanMessage(content=user_input)
    
    return {
        **state,
        "messages": [user_message],
        "user_input": user_input
    }

def should_continue(state: ConversationState) -> Literal["continue", "end"]:
    """Determine whether to continue the conversation or end."""
    if state.get("user_input") == "exit":
        return "end"
    return "continue"

def create_persistent_conversation():
    """Create a persistent conversation workflow using LangGraph."""
    print("Creating a persistent conversation workflow with LangGraph...")
    
    # Create a new graph
    workflow = StateGraph(ConversationState)
    
    # Add nodes
    workflow.add_node("human_input", human_input_node)
    workflow.add_node("call_model", call_model_node)
    workflow.add_node("call_tools", call_tools_node)
    
    workflow.add_conditional_edges(
        "human_input",
        should_continue,
        {
            "end": END,
            "continue": "call_model"  # If no tools, go back to human input
        }
    )
    
    # Add conditional edges for tool calling
    workflow.add_conditional_edges(
        "call_model",
        should_call_tool,
        {
            "call_tools": "call_tools",
            "call_model": "human_input"  # If no tools, go back to human input
        }
    )
    
    # After tools, go back to model
    workflow.add_edge("call_tools", "call_model")
    
    # Set entry point
    workflow.set_entry_point("human_input")
    
    # Create a checkpointer for persistence
    checkpointer = MemorySaver()
    
    # Compile the graph with the checkpointer
    return workflow.compile(checkpointer=checkpointer)

def display_session_summary(session_data: Dict[str, Any]):
    """Display a summary of the session."""
    if not session_data:
        return
    
    session = SessionData(**session_data)
    duration_seconds = time.time() - session.start_time
    hours, remainder = divmod(duration_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"\n📊 Session Summary:")
    print(f"   • Session ID: {session.session_id}")
    print(f"   • Duration: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print(f"   • Interactions: {session.interaction_count}")
    print(f"   • Topics: {', '.join(session.topics_discussed) if session.topics_discussed else 'None'}")

def run_persistent_conversation(workflow, user_id: str = None):
    """Run a persistent conversation with the given workflow."""
    if user_id is None:
        user_id = str(uuid.uuid4())[:8]  # Shorter ID for display
    
    print("\n" + "=" * 60)
    print(f"🔄 Persistent Conversation for User: {user_id}")
    print("=" * 60)
    
    # Configuration for persistence
    config = {"configurable": {"thread_id": user_id}}
    
    # Create initial session data
    session = SessionData(
        session_id=str(uuid.uuid4())[:8],
        start_time=time.time(),
        last_active=time.time(),
        interaction_count=0,
        topics_discussed=[]
    )
    
    # Initialize the state
    initial_state = {
        "messages": [],
        "context": {"user_id": user_id},
        "current_topic": "general",
        "session_data": session.model_dump(),
        "user_input": ""
    }
    
    print(f"💾 Session will be saved with ID: {session.session_id}")
    print("💬 Start chatting! (Type 'exit' to quit)")
    
    try:
        # Run the conversation loop
        current_state = initial_state
        
        while True:
            # Stream the workflow execution
            events = list(workflow.stream(current_state, config=config))
            
            # Get the final state
            if events:
                final_event = events[-1]
                # Extract the state from the final event
                for node_name, node_state in final_event.items():
                    current_state = node_state
                    break
            
            # Check if we should exit
            if current_state.get("user_input") == "exit":
                break
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Conversation interrupted by user.")
    except Exception as e:
        print(f"\n❌ An error occurred: {str(e)}")
    finally:
        # Display session summary
        display_session_summary(current_state.get("session_data"))
        print("✅ Conversation ended. Data saved!")

def list_user_profiles():
    """List all existing user profiles."""
    if not os.path.exists(PROFILES_DIR):
        print("No user profiles found.")
        return []
    
    profiles = []
    for filename in os.listdir(PROFILES_DIR):
        if filename.startswith("user_") and filename.endswith(".json"):
            user_id = filename[5:-5]  # Remove "user_" prefix and ".json" suffix
            profiles.append(user_id)
    
    return profiles

def main():
    """Main function to demonstrate persistence and state management."""
    print("=" * 60)
    print("🔄 Persistence and State Management in LangGraph")
    print("=" * 60)
    
    # Create a persistent conversation workflow
    workflow = create_persistent_conversation()
    
    # Show existing profiles
    existing_profiles = list_user_profiles()
    if existing_profiles:
        print(f"\n📁 Found {len(existing_profiles)} existing user profiles:")
        for i, profile in enumerate(existing_profiles, 1):
            print(f"   {i}. User ID: {profile}")
    
    # Ask user what they want to do
    print("\n🎯 What would you like to do?")
    print("1. Start a new conversation (new user)")
    print("2. Continue with an existing user")
    print("3. Resume with a specific user ID")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-3): ").strip()
            
            if choice == "1":
                # New conversation
                run_persistent_conversation(workflow)
                break
            elif choice == "2" and existing_profiles:
                # Show existing profiles and let user choose
                print("\nSelect a user profile:")
                for i, profile in enumerate(existing_profiles, 1):
                    print(f"{i}. {profile}")
                
                try:
                    profile_choice = int(input("Enter profile number: ")) - 1
                    if 0 <= profile_choice < len(existing_profiles):
                        selected_user = existing_profiles[profile_choice]
                        run_persistent_conversation(workflow, selected_user)
                        break
                    else:
                        print("Invalid selection.")
                except ValueError:
                    print("Please enter a valid number.")
            elif choice == "3":
                # Specific user ID
                user_id = input("Enter User ID: ").strip()
                if user_id:
                    run_persistent_conversation(workflow, user_id)
                    break
                else:
                    print("Please enter a valid User ID.")
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break

if __name__ == "__main__":
    main()