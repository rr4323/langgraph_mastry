"""
LangGraph Advanced: Event-Driven Architectures
===========================================

This script demonstrates how to build event-driven architectures
using LangGraph and Google's Generative AI model.
"""

import os
import sys
import time
import uuid
import asyncio
from typing import Annotated, Dict, List, Literal, TypedDict, Union, Any, Optional
from enum import Enum
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

# Add the parent directory to the path so we can import from the root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables from .env file
load_dotenv()

# Define event types
class EventType(str, Enum):
    """Types of events in the system."""
    USER_MESSAGE = "user_message"
    SYSTEM_ALERT = "system_alert"
    DATA_UPDATE = "data_update"
    TIMER = "timer"
    EXTERNAL_API = "external_api"

# Define event priorities
class EventPriority(str, Enum):
    """Priority levels for events."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

# Define event model
class Event(BaseModel):
    """Model for events in the system."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: EventType
    priority: EventPriority = EventPriority.MEDIUM
    timestamp: float = Field(default_factory=time.time)
    data: Dict[str, Any]
    processed: bool = False

# Define our state for the event-driven system
class EventDrivenState(TypedDict):
    """State for our event-driven system."""
    messages: List[Union[HumanMessage, AIMessage, SystemMessage]]
    events: List[Event]
    event_history: List[Dict[str, Any]]
    current_event: Optional[Event]
    response: str
    next: Optional[str]

# Event queue for the system
EVENT_QUEUE = []

# Event handlers
def handle_user_message(event: Event) -> str:
    """Handle a user message event."""
    print(f"📨 Handling user message: {event.data.get('content', '')[:50]}...")
    
    # Create a chat model
    model = ChatGoogleGenerativeAI(
        model="gemini-pro",
        temperature=0.7,
        convert_system_message_to_human=True
    )
    
    # Create messages for the model
    messages = [
        SystemMessage(content="You are a helpful assistant responding to a user message."),
        HumanMessage(content=event.data.get("content", ""))
    ]
    
    # Get the response
    response = model.invoke(messages)
    
    return response.content

def handle_system_alert(event: Event) -> str:
    """Handle a system alert event."""
    print(f"⚠️ Handling system alert: {event.data.get('alert_type', '')}...")
    
    alert_type = event.data.get("alert_type", "unknown")
    alert_message = event.data.get("message", "No details provided")
    
    # Create a chat model
    model = ChatGoogleGenerativeAI(
        model="gemini-pro",
        temperature=0.5,
        convert_system_message_to_human=True
    )
    
    # Create messages for the model
    messages = [
        SystemMessage(content="""You are a system alert processor.
Your job is to analyze system alerts and provide clear, actionable responses."""),
        HumanMessage(content=f"""
System Alert:
Type: {alert_type}
Message: {alert_message}

Please analyze this alert and provide:
1. A brief explanation of what this alert means
2. Potential impacts
3. Recommended actions
""")
    ]
    
    # Get the response
    response = model.invoke(messages)
    
    return response.content

def handle_data_update(event: Event) -> str:
    """Handle a data update event."""
    print(f"📊 Handling data update: {event.data.get('data_type', '')}...")
    
    data_type = event.data.get("data_type", "unknown")
    data_value = event.data.get("value", {})
    previous_value = event.data.get("previous_value", {})
    
    # Create a chat model
    model = ChatGoogleGenerativeAI(
        model="gemini-pro",
        temperature=0.4,
        convert_system_message_to_human=True
    )
    
    # Create messages for the model
    messages = [
        SystemMessage(content="""You are a data update analyzer.
Your job is to analyze changes in data and provide insights."""),
        HumanMessage(content=f"""
Data Update:
Type: {data_type}
Previous Value: {previous_value}
New Value: {data_value}

Please analyze this data update and provide:
1. A summary of the changes
2. Potential implications of these changes
3. Any recommended actions based on these changes
""")
    ]
    
    # Get the response
    response = model.invoke(messages)
    
    return response.content

def handle_timer(event: Event) -> str:
    """Handle a timer event."""
    print(f"⏰ Handling timer event: {event.data.get('timer_type', '')}...")
    
    timer_type = event.data.get("timer_type", "unknown")
    scheduled_time = event.data.get("scheduled_time", time.time())
    context = event.data.get("context", {})
    
    # Format the scheduled time
    scheduled_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(scheduled_time))
    
    # Create a chat model
    model = ChatGoogleGenerativeAI(
        model="gemini-pro",
        temperature=0.5,
        convert_system_message_to_human=True
    )
    
    # Create messages for the model
    messages = [
        SystemMessage(content="""You are a timer event processor.
Your job is to handle scheduled events and provide appropriate responses."""),
        HumanMessage(content=f"""
Timer Event:
Type: {timer_type}
Scheduled Time: {scheduled_time_str}
Context: {context}

Please process this timer event and provide an appropriate response.
""")
    ]
    
    # Get the response
    response = model.invoke(messages)
    
    return response.content

def handle_external_api(event: Event) -> str:
    """Handle an external API event."""
    print(f"🌐 Handling external API event: {event.data.get('api_name', '')}...")
    
    api_name = event.data.get("api_name", "unknown")
    api_response = event.data.get("response", {})
    api_status = event.data.get("status", "unknown")
    
    # Create a chat model
    model = ChatGoogleGenerativeAI(
        model="gemini-pro",
        temperature=0.5,
        convert_system_message_to_human=True
    )
    
    # Create messages for the model
    messages = [
        SystemMessage(content="""You are an API response processor.
Your job is to analyze API responses and provide meaningful interpretations."""),
        HumanMessage(content=f"""
External API Event:
API: {api_name}
Status: {api_status}
Response: {api_response}

Please analyze this API response and provide:
1. A summary of the response
2. Any important information or insights from the response
3. Recommended next steps based on this response
""")
    ]
    
    # Get the response
    response = model.invoke(messages)
    
    return response.content

# Event dispatcher
def dispatch_event(event: Event) -> str:
    """Dispatch an event to the appropriate handler."""
    if event.type == EventType.USER_MESSAGE:
        return handle_user_message(event)
    elif event.type == EventType.SYSTEM_ALERT:
        return handle_system_alert(event)
    elif event.type == EventType.DATA_UPDATE:
        return handle_data_update(event)
    elif event.type == EventType.TIMER:
        return handle_timer(event)
    elif event.type == EventType.EXTERNAL_API:
        return handle_external_api(event)
    else:
        return f"Unknown event type: {event.type}"

# Tools for event generation
@tool
def create_system_alert(alert_type: str, message: str, priority: str = "medium") -> str:
    """
    Create a system alert event.
    
    Args:
        alert_type: The type of alert
        message: The alert message
        priority: The priority level (low, medium, high, critical)
    
    Returns:
        str: Confirmation message
    """
    # Validate priority
    try:
        event_priority = EventPriority(priority.lower())
    except ValueError:
        event_priority = EventPriority.MEDIUM
    
    # Create the event
    event = Event(
        type=EventType.SYSTEM_ALERT,
        priority=event_priority,
        data={
            "alert_type": alert_type,
            "message": message
        }
    )
    
    # Add to the queue
    EVENT_QUEUE.append(event)
    
    return f"System alert created: {alert_type} ({priority})"

@tool
def create_data_update(data_type: str, value: Dict[str, Any], previous_value: Dict[str, Any] = None, priority: str = "medium") -> str:
    """
    Create a data update event.
    
    Args:
        data_type: The type of data being updated
        value: The new value
        previous_value: The previous value
        priority: The priority level (low, medium, high, critical)
    
    Returns:
        str: Confirmation message
    """
    # Validate priority
    try:
        event_priority = EventPriority(priority.lower())
    except ValueError:
        event_priority = EventPriority.MEDIUM
    
    # Create the event
    event = Event(
        type=EventType.DATA_UPDATE,
        priority=event_priority,
        data={
            "data_type": data_type,
            "value": value,
            "previous_value": previous_value or {}
        }
    )
    
    # Add to the queue
    EVENT_QUEUE.append(event)
    
    return f"Data update event created: {data_type} ({priority})"

@tool
def schedule_timer(timer_type: str, delay_seconds: int, context: Dict[str, Any] = None, priority: str = "medium") -> str:
    """
    Schedule a timer event.
    
    Args:
        timer_type: The type of timer
        delay_seconds: The delay in seconds
        context: Additional context for the timer
        priority: The priority level (low, medium, high, critical)
    
    Returns:
        str: Confirmation message
    """
    # Validate priority
    try:
        event_priority = EventPriority(priority.lower())
    except ValueError:
        event_priority = EventPriority.MEDIUM
    
    # Calculate the scheduled time
    scheduled_time = time.time() + delay_seconds
    
    # Create the event
    event = Event(
        type=EventType.TIMER,
        priority=event_priority,
        data={
            "timer_type": timer_type,
            "scheduled_time": scheduled_time,
            "context": context or {}
        }
    )
    
    # Add to the queue
    EVENT_QUEUE.append(event)
    
    return f"Timer event scheduled: {timer_type} in {delay_seconds} seconds ({priority})"

@tool
def simulate_api_response(api_name: str, status: str, response: Dict[str, Any], priority: str = "medium") -> str:
    """
    Simulate an external API response event.
    
    Args:
        api_name: The name of the API
        status: The status of the API response
        response: The API response data
        priority: The priority level (low, medium, high, critical)
    
    Returns:
        str: Confirmation message
    """
    # Validate priority
    try:
        event_priority = EventPriority(priority.lower())
    except ValueError:
        event_priority = EventPriority.MEDIUM
    
    # Create the event
    event = Event(
        type=EventType.EXTERNAL_API,
        priority=event_priority,
        data={
            "api_name": api_name,
            "status": status,
            "response": response
        }
    )
    
    # Add to the queue
    EVENT_QUEUE.append(event)
    
    return f"External API event created: {api_name} ({priority})"

# Workflow nodes
def event_listener_node(state: EventDrivenState) -> EventDrivenState:
    """Event listener node in the event-driven system."""
    print("👂 Listening for events...")
    
    # Check if there are any events in the queue
    if not EVENT_QUEUE:
        # If there are no events, check if there's a user message
        messages = state["messages"]
        if messages and isinstance(messages[-1], HumanMessage):
            # Create a user message event from the last message
            event = Event(
                type=EventType.USER_MESSAGE,
                priority=EventPriority.HIGH,
                data={
                    "content": messages[-1].content
                }
            )
            EVENT_QUEUE.append(event)
    
    # If there are still no events, wait for a bit
    if not EVENT_QUEUE:
        print("  No events in the queue. Waiting...")
        time.sleep(1)
        return {
            **state,
            "next": "event_listener"
        }
    
    # Get the highest priority event
    events_by_priority = {
        EventPriority.CRITICAL: [],
        EventPriority.HIGH: [],
        EventPriority.MEDIUM: [],
        EventPriority.LOW: []
    }
    
    for event in EVENT_QUEUE:
        events_by_priority[event.priority].append(event)
    
    # Get the highest priority event
    for priority in [EventPriority.CRITICAL, EventPriority.HIGH, EventPriority.MEDIUM, EventPriority.LOW]:
        if events_by_priority[priority]:
            current_event = events_by_priority[priority][0]
            EVENT_QUEUE.remove(current_event)
            break
    else:
        # This should never happen, but just in case
        return {
            **state,
            "next": "event_listener"
        }
    
    # Update the state with the current event
    return {
        **state,
        "current_event": current_event.model_dump(),
        "next": "event_processor"
    }

def event_processor_node(state: EventDrivenState) -> EventDrivenState:
    """Event processor node in the event-driven system."""
    # Get the current event
    current_event = state["current_event"]
    
    if not current_event:
        return {
            **state,
            "next": "event_listener"
        }
    
    # Create an Event object from the dict
    event = Event(**current_event)
    
    print(f"🔄 Processing event: {event.type} (Priority: {event.priority})...")
    
    # Dispatch the event to the appropriate handler
    response = dispatch_event(event)
    
    # Add the event to the history
    event_history = state["event_history"]
    event_history.append({
        "id": event.id,
        "type": event.type,
        "priority": event.priority,
        "timestamp": event.timestamp,
        "data": event.data,
        "response": response
    })
    
    # Update the state
    new_state = {
        **state,
        "response": response,
        "event_history": event_history,
        "current_event": None,
        "next": "response_formatter"
    }
    
    # If the event was a user message, add the response to the messages
    if event.type == EventType.USER_MESSAGE:
        new_messages = add_messages(state["messages"], [AIMessage(content=response)])
        new_state["messages"] = new_messages
    
    return new_state

def response_formatter_node(state: EventDrivenState) -> EventDrivenState:
    """Response formatter node in the event-driven system."""
    print("📝 Formatting response...")
    
    # Get the current event and response
    current_event = state["current_event"]
    response = state["response"]
    
    # Check if there are more events in the queue
    if EVENT_QUEUE:
        return {
            **state,
            "next": "event_listener"
        }
    else:
        return {
            **state,
            "next": "end"
        }

def create_event_driven_system():
    """Create an event-driven system using LangGraph."""
    print("Creating an event-driven system with LangGraph...")
    
    # Create a new graph
    workflow = StateGraph(EventDrivenState)
    
    # Add nodes to the graph
    workflow.add_node("event_listener", event_listener_node)
    workflow.add_node("event_processor", event_processor_node)
    workflow.add_node("response_formatter", response_formatter_node)
    
    # Add edges
    workflow.add_conditional_edges(
        "event_listener",
        lambda state: state["next"],
        {
            "event_listener": "event_listener",
            "event_processor": "event_processor"
        }
    )
    
    workflow.add_conditional_edges(
        "event_processor",
        lambda state: state["next"],
        {
            "response_formatter": "response_formatter"
        }
    )
    
    workflow.add_conditional_edges(
        "response_formatter",
        lambda state: state["next"],
        {
            "event_listener": "event_listener",
            "end": END
        }
    )
    
    # Set the entry point
    workflow.set_entry_point("event_listener")
    
    # Compile the graph
    return workflow.compile()

def run_event_driven_system(workflow, user_message: str = None):
    """Run the event-driven system with optional user message."""
    print("\n" + "=" * 50)
    print("Running the Event-Driven System")
    print("=" * 50)
    
    # Initialize the state
    initial_state = {
        "messages": [HumanMessage(content=user_message)] if user_message else [],
        "events": [],
        "event_history": [],
        "current_event": None,
        "response": "",
        "next": None
    }
    
    # Run the workflow
    result = workflow.invoke(initial_state)
    
    # Print the workflow results
    print("\n" + "=" * 50)
    print("Event-Driven System Results")
    print("=" * 50)
    
    print("\nEvent History:")
    print("-" * 50)
    for event in result["event_history"]:
        print(f"Type: {event['type']}, Priority: {event['priority']}")
        print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(event['timestamp']))}")
        print(f"Data: {event['data']}")
        print(f"Response: {event['response'][:100]}..." if len(event['response']) > 100 else event['response'])
        print("-" * 30)
    
    # If there were messages, print the conversation
    if result["messages"]:
        print("\nConversation:")
        print("-" * 50)
        for msg in result["messages"]:
            if isinstance(msg, HumanMessage):
                print(f"User: {msg.content}")
            elif isinstance(msg, AIMessage):
                print(f"Assistant: {msg.content}")
    
    return result

def simulate_events():
    """Simulate various events for the event-driven system."""
    # Clear the event queue
    global EVENT_QUEUE
    EVENT_QUEUE = []
    
    # Create a system alert
    create_system_alert(
        alert_type="disk_space_low",
        message="Server disk space is below 10% (5.2GB remaining)",
        priority="high"
    )
    
    # Create a data update
    create_data_update(
        data_type="user_metrics",
        value={"active_users": 1250, "new_signups": 75, "churn_rate": 0.05},
        previous_value={"active_users": 1200, "new_signups": 60, "churn_rate": 0.06},
        priority="medium"
    )
    
    # Schedule a timer
    schedule_timer(
        timer_type="daily_report",
        delay_seconds=1,
        context={"report_type": "user_activity", "period": "daily"},
        priority="low"
    )
    
    # Simulate an API response
    simulate_api_response(
        api_name="weather_api",
        status="success",
        response={"temperature": 22.5, "conditions": "partly cloudy", "precipitation": 0.2},
        priority="medium"
    )
    
    print(f"Simulated {len(EVENT_QUEUE)} events.")

def main():
    """Main function to demonstrate event-driven architectures."""
    print("=" * 50)
    print("Event-Driven Architectures in LangGraph")
    print("=" * 50)
    
    # Let the user choose a mode
    print("\nSelect a mode:")
    print("1. Process a user message")
    print("2. Simulate various system events")
    print("3. Process both user message and system events")
    
    while True:
        try:
            choice = int(input("\nEnter your choice (1-3): "))
            
            # Create an event-driven system
            workflow = create_event_driven_system()
            
            if choice == 1:
                # Get a user message
                user_message = input("\nEnter your message: ")
                
                # Run the event-driven system with the user message
                run_event_driven_system(workflow, user_message)
                break
                
            elif choice == 2:
                # Simulate events
                simulate_events()
                
                # Run the event-driven system with the simulated events
                run_event_driven_system(workflow)
                break
                
            elif choice == 3:
                # Get a user message
                user_message = input("\nEnter your message: ")
                
                # Simulate events
                simulate_events()
                
                # Run the event-driven system with both
                run_event_driven_system(workflow, user_message)
                break
                
            else:
                print("Please enter 1, 2, or 3.")
        except ValueError:
            print("Please enter a valid number.")

if __name__ == "__main__":
    main()
