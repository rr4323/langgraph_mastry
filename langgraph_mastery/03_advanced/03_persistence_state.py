"""
LangGraph Advanced: Persistence and State Management
=================================================

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
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from langgraph.prebuilt import create_react_agent


# Add the parent directory to the path so we can import from the root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables from .env file
load_dotenv()

# Define our state for the persistent workflow
class ConversationState(TypedDict):
    """State for our persistent conversation workflow."""
    messages: List[Union[HumanMessage, AIMessage, SystemMessage]]
    context: Dict[str, Any]
    current_topic: str
    session_data: Dict[str, Any]
    next: Optional[str]
    input: Optional[str]

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
        with open(profile_path, "r") as f:
            data = json.load(f)
        return UserProfile(**data)
    else:
        # Create a default profile
        return UserProfile(
            name=f"User_{user_id[:5]}",
            interests=[],
            preferences={}
        )

def save_user_profile(user_id: str, profile: UserProfile):
    """Save a user profile to disk."""
    profile_path = get_profile_path(user_id)
    
    with open(profile_path, "w") as f:
        json.dump(profile.model_dump(), f, indent=2)

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
    
    return f"""
User Profile:
- Name: {profile.name}
- Interests: {', '.join(profile.interests) if profile.interests else 'No interests recorded yet'}
- Preferences: {json.dumps(profile.preferences, indent=2) if profile.preferences else 'No preferences recorded yet'}
"""

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
def get_session_info(session_data: Dict[str, Any]) -> str:
    """
    Get information about the current session.
    
    Args:
        session_data: The session data dictionary
    
    Returns:
        str: Information about the session
    """
    session = SessionData(**session_data)
    
    # Calculate session duration
    duration_seconds = time.time() - session.start_time
    hours, remainder = divmod(duration_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    duration_str = ""
    if hours > 0:
        duration_str += f"{int(hours)} hours, "
    if minutes > 0:
        duration_str += f"{int(minutes)} minutes, "
    duration_str += f"{int(seconds)} seconds"
    
    return f"""
Session Information:
- Session ID: {session.session_id}
- Duration: {duration_str}
- Interactions: {session.interaction_count}
- Topics Discussed: {', '.join(session.topics_discussed) if session.topics_discussed else 'No topics recorded yet'}
"""

def agent_node(state: ConversationState) -> ConversationState:
    """Agent node for processing user input and maintaining state."""

    # Ensure required keys exist
    state.setdefault("messages", [])
    state.setdefault("context", {})
    state.setdefault("session_data", {})
    
    messages = state["messages"]
    context = state["context"]
    session_data = state["session_data"]

    # Extract or default user_id
    user_id = context.get("user_id", "default_user")

    # Update session state
    session = SessionData(**session_data)
    session.update_activity()
    session.increment_interaction()
    # Prepare tools (must match ReAct tool signature)
    tools = [get_user_info, update_user_interest, update_user_preference, get_session_info]

    # Ensure we have a human message
    last_message = messages[-1] if messages else None

    topic_model = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.1,
        convert_system_message_to_human=True,
    )
    topic_response = topic_model.invoke(last_message.content)
    
    current_topic = topic_response.content.strip()
    session.add_topic(current_topic)
    state["current_topic"] = current_topic
    
    agent_model = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.7,
        convert_system_message_to_human=True
    )
    
    system_message = SystemMessage(content=f"""You are a helpful assistant with persistent memory.
                                                You can remember information about the user across conversations.
                                                Current user ID: {user_id}
                                                Current topic: {state.get("current_topic", "general")}

                                                Use the available tools to:
                                                1. Get information about the user
                                                2. Update the user's interests based on the conversation
                                                3. Update the user's preferences when they express likes/dislikes
                                                4. Get information about the current session

                                                Always be helpful, personalized, and contextually aware.
                                                """)

    # Rebuild message list with system message
    agent_messages = [system_message] + messages

    # Create ReAct agent and executor
    agent = create_react_agent(model=agent_model, tools=tools, prompt=system_message)

    # Step 3: Invoke the agent with last user message
    try:
        response = agent.invoke(agent_messages)
        messages.append(AIMessage(content=response["output"]))
    except Exception as e:
        messages.append(AIMessage(content=f"Sorry, an error occurred: {str(e)}"))

    # Step 4: Persist updated session
    state["session_data"] = session.model_dump()
    state["messages"] = messages
    state["next"] = "human"  # continue the loop

    return state
def human_node(state):
    """
    Human node in LangGraph for CLI use.
    Pauses and returns the current state to wait for the next user input.
    """
    user_input = input("👤 You (or type 'exit'): ")

    if user_input.lower() in {"exit", "quit"}:
        print("✅ Exiting... See you next time.")
        import sys
        sys.exit(0)  # Exit the program immediately
    else:
        state["messages"].append(HumanMessage(content=user_input))
        state["next"] = "agent"  # continue the loop
    return state


def create_persistent_conversation():
    """Create a persistent conversation workflow using LangGraph."""
    print("Creating a persistent conversation workflow with LangGraph...")
    
    # Create a new graph
    workflow = StateGraph(ConversationState)
    
    # Add the agent node
    workflow.add_node("agent", agent_node)
    workflow.add_node("human", human_node)
    workflow.add_edge("agent", "human")
    workflow.add_edge("human", "agent")
    
    # Set the entry point
    workflow.set_entry_point("human")
    
    # Create a checkpointer for persistence
    # For in-memory persistence (restarts will lose data)
    memory_checkpointer = InMemorySaver()
    
    # For SQLite persistence (data persists across restarts)
    conn = sqlite3.connect(SQLITE_DB_PATH,check_same_thread=False)
    sqlite_checkpointer = SqliteSaver(conn)
    
    # Choose which checkpointer to use
    checkpointer = sqlite_checkpointer  # or memory_checkpointer
    
    # Compile the graph with the checkpointer
    return workflow.compile(checkpointer=checkpointer)

def run_persistent_conversation(workflow, user_id: str = None):
    """Run a persistent conversation with the given workflow."""
    if user_id is None:
        user_id = str(uuid.uuid4())
    
    print("\n" + "=" * 50)
    print(f"Persistent Conversation for User: {user_id}")
    print("=" * 50)
    
    # Check if we have an existing session for this user
    config = {"configurable": {"thread_id": user_id}}
    
    try:
        # Try to resume an existing session
        print("Attempting to resume existing session...")
        state = workflow.get_state(config=config).values
        if state:
            print("Existing session found!")
            # Extract session data
            session_data = state["session_data"]
            session = SessionData(**session_data)
             # Display previous messages
            print("\nPrevious conversation:")
            for msg in state["messages"]:
                if isinstance(msg, HumanMessage):
                    print(f"User: {msg.content}")
                elif isinstance(msg, AIMessage):
                    print(f"Assistant: {msg.content}")  
        else:
            print("No existing session found. Starting a new conversation.")
            session = SessionData(
                session_id=user_id,
                start_time=time.time(),
                last_active=time.time(),
                interaction_count=0,
                topics_discussed=[]
            )
            state = {
                "messages": [],
                "context": {"user_id": user_id},
                "current_topic": "general",
                "session_data": session.model_dump()
            }
        
        print(f"Session ID: {session.session_id}")
        print(f"Interaction count: {session.interaction_count}")
        print(f"Topics discussed: {', '.join(session.topics_discussed)}")
        
        
        # Resume the conversation
        while True:
            
            # Invoke the workflow with the updated state
            state = workflow.invoke(state, config=config)
            print(state)
    
    except Exception as e:
        print(f"No existing session found or error: {str(e)}")
        print("Starting a new conversation...")
        
        # Create a new session
        session_id = str(uuid.uuid4())
        session = SessionData(
            session_id=session_id,
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
            "session_data": session.model_dump()
        }
        
        # Start the conversation
        state = initial_state
        
        while True:
            # Invoke the workflow with the updated state
            state = workflow.invoke(state, config=config)
            print(state)    


def main():
    """Main function to demonstrate persistence and state management."""
    print("=" * 50)
    print("Persistence and State Management in LangGraph")
    print("=" * 50)
    
    # Create a persistent conversation workflow
    workflow = create_persistent_conversation()
    
    # Ask if the user wants to resume an existing conversation
    print("\nDo you want to resume an existing conversation?")
    choice = input("Enter 'y' for yes, 'n' for a new conversation: ").lower()
    
    if choice == 'y':
        user_id = input("Enter your User ID: ")
        run_persistent_conversation(workflow, user_id)
    else:
        run_persistent_conversation(workflow)

if __name__ == "__main__":
    main()
