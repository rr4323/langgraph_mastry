"""
This script demonstrates how to write memories in the background in a LangGraph application.

This advanced technique is crucial for production systems as it separates the main application
logic (the "hot path") from the potentially time-consuming process of creating and storing
memories. By offloading memory operations to a background task, the application can remain
responsive to user input, significantly improving the user experience.

Key Concepts:
- **Background Processing**: Using a separate thread or process to handle tasks that are not
  critical for the immediate response to the user. This prevents blocking the main thread.
- **Separation of Concerns**: The main agent is responsible for interacting with the user, while a
  dedicated memory manager handles the persistence of conversation history.
- **Thread-Safe Communication**: Using a queue (`collections.deque` in this example) to safely
  pass data (conversation history) from the main thread to the background thread.
- **Asynchronous Memory Saving**: The background thread periodically processes the queue, summarizes
  the conversation, and saves it to a simulated persistent store (a JSON file).

Workflow:
1. **Main Application (`StatefulGraph`)**: An agent interacts with the user, appending messages to a
   state object. After each interaction, it adds the latest messages to a shared queue.
2. **Shared Queue**: A `collections.deque` is used as a thread-safe buffer to hold conversation
   chunks that need to be processed and saved by the background memory writer.
3. **Background Memory Writer (`MemorySaver`)**: A separate thread that runs a continuous loop:
   - It checks the shared queue for new conversation data.
   - If data is present, it processes it (e.g., generates a summary).
   - It saves the summary to a persistent store (a local JSON file in this example).
   - It sleeps for a short interval to avoid busy-waiting.

To run this script:
```bash
python langgraph_mastery/03_advanced/09_background_memory.py
```

After running, a `memory.json` file will be created in the same directory, containing the
summarized conversation history.
"""

import json
import threading
import time
import uuid
from collections import deque
from typing import Annotated, List

from langchain_core.messages import AnyMessage, BaseMessage, SystemMessage, ToolMessage
from typing_extensions import TypedDict

from langgraph.graph import END, StateGraph



class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], lambda x, y: x + y]


# A thread-safe queue for communication between the main graph and the memory saver
memory_queue = deque()


class MemorySaver(threading.Thread):
    """A background thread that saves conversation memories from a queue."""

    def __init__(self, queue, memory_file="memory.json"):
        super().__init__()
        self.queue = queue
        self.memory_file = memory_file
        self.stop_event = threading.Event()
        self.daemon = True  # Allows main thread to exit even if this thread is running

    def run(self):
        """The main loop for the background memory saving thread."""
        print("Background memory saver started.")
        while not self.stop_event.is_set():
            if self.queue:
                # Get all messages currently in the queue
                messages_to_save = list(self.queue)
                self.queue.clear()

                print(f"\n[Background] Processing {len(messages_to_save)} messages for memory.")
                self._save_memory(messages_to_save)
            else:
                # Wait for a short period to avoid busy-waiting
                time.sleep(1)
        print("Background memory saver stopped.")

    def _save_memory(self, messages: List[BaseMessage]):
        """Simulates summarizing and saving messages to a persistent store."""
        # In a real application, this would involve a call to an LLM to summarize
        summary = f"Conversation summary of {len(messages)} messages at {time.ctime()}:\n"
        for msg in messages:
            summary += f"- {msg.type}: {msg.content}\n"

        try:
            with open(self.memory_file, "r+") as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            data = []

        data.append({"id": str(uuid.uuid4()), "summary": summary})

        with open(self.memory_file, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[Background] Memory saved to {self.memory_file}")

    def stop(self):
        """Signals the thread to stop."""
        self.stop_event.set()


def agent_node(state: AgentState):
    """Simulates the agent's response generation."""
    print("\n---AGENT NODE---")
    # In a real app, this would be an LLM call
    response_text = f"This is a response to: '{state['messages'][-1].content}'."
    return {"messages": [SystemMessage(content=response_text)]}


def memory_writer_node(state: AgentState):
    """Node that puts the latest messages into the queue for the background saver."""
    print("---MEMORY WRITER NODE---")
    # This node is the last step in the graph, so we add the messages to the queue.
    # Add the last two messages (user input and agent response) to the queue
    messages_to_queue = state["messages"][-2:]
    memory_queue.extend(messages_to_queue)
    print(f"Added {len(messages_to_queue)} messages to the background memory queue.")
    return {}


# Define the graph
builder = StateGraph(AgentState)
builder.add_node("agent", agent_node)
builder.add_node("memory_writer", memory_writer_node)

builder.set_entry_point("agent")
builder.add_edge("agent", "memory_writer")
builder.add_edge("memory_writer", END)

graph = builder.compile()


def main():
    """Main function to run the agent and the background memory saver."""
    # Start the background memory saver thread
    saver = MemorySaver(memory_queue)
    saver.start()

    print("Starting agent interaction...")
    try:
        # Simulate a few turns of conversation
        inputs = [
            {"messages": [SystemMessage(content="What is the weather like?")]},
            {"messages": [SystemMessage(content="What about tomorrow?")]},
            {"messages": [SystemMessage(content="Thanks!")]},
        ]

        for i, user_input in enumerate(inputs):
            print(f"\n--- Turn {i+1} ---")
            graph.invoke(user_input)
            # In a real app, there would be a delay here representing user thinking time
            time.sleep(2)

    finally:
        # Stop the background thread gracefully
        print("\nStopping agent and saving any remaining memories...")
        # Wait for the queue to be processed before stopping
        while memory_queue:
            time.sleep(0.5)
        saver.stop()
        saver.join()  # Wait for the thread to finish
        print("Application finished.")


if __name__ == "__main__":
    main()
