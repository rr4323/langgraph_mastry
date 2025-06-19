"""
This script provides a comprehensive example of LangGraph's streaming capabilities.

Streaming is essential for building responsive and transparent AI applications. LangGraph
provides multiple streaming modes to give developers fine-grained control over what
information is streamed back to the client during a graph's execution.

This script demonstrates three key streaming modes:
1.  **`values`**: Streams the full state of the graph after each node completes. This is useful
    for applications that need to display the complete state at each step, such as a
    dashboard that visualizes the entire data flow.

2.  **`updates`**: Streams only the *changes* to the state after each node completes. This is
    more efficient than `values` if you only need to know what's new at each step. It's
    ideal for logging or for frontends that can intelligently merge state updates.

3.  **`messages`**: Streams LLM tokens in real-time as they are generated within any node.
    This is the mode you would use to create a classic chatbot-style streaming effect,
    where the AI's response appears token by token.

Workflow:
The script defines a simple graph with two nodes:
1.  **`refine_topic`**: A simple function that modifies the initial topic.
2.  **`generate_joke`**: A node that calls an LLM (OpenAI's GPT-4o-mini) to generate a joke
    about the refined topic.

The main part of the script then invokes this graph three times, each time using a
different `stream_mode`, and prints the output to illustrate the differences.

To run this script:
```bash
# Make sure you have a GOOGLE_API_KEY set in your environment variables.
python langgraph_mastery/03_advanced/10_streaming_output.py
```
"""

import os
from typing import TypedDict

from langchain_google_genai import ChatGoogleGenerativeAI
from typing_extensions import Annotated

from langgraph.graph import END, StateGraph, START

# Set up the Google API key
# Make sure to set the GOOGLE_API_KEY environment variable in your .env file or directly.
# from dotenv import load_dotenv
# load_dotenv()

if "GOOGLE_API_KEY" not in os.environ:
    raise ValueError(
        "The GOOGLE_API_KEY environment variable is not set. Please add it to your .env file or set it directly."
    )


class StreamState(TypedDict):
    """State for the streaming example graph."""

    topic: str
    joke: str


def refine_topic_node(state: StreamState):
    """A node that refines the initial topic."""
    print("---REFINING TOPIC---")
    return {"topic": state["topic"] + " and cats"}


def generate_joke_node(state: StreamState):
    """A node that generates a joke using an LLM."""
    print("---GENERATING JOKE---")
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0)
    # Note: Even when we use .invoke() here, the `messages` stream mode will still
    # capture the token-by-token output from the LLM.
    response = llm.invoke(f"Tell me a short, one-sentence joke about {state['topic']}")
    return {"joke": response.content}


# Define the graph structure
builder = StateGraph(StreamState)
builder.add_node("refine_topic", refine_topic_node)
builder.add_node("generate_joke", generate_joke_node)

builder.add_edge(START, "refine_topic")
builder.add_edge("refine_topic", "generate_joke")
builder.add_edge("generate_joke", END)

graph = builder.compile()


def main():
    """Main function to demonstrate the different streaming modes."""
    initial_input = {"topic": "ice cream", "joke": ""}

    # --- 1. Stream Mode: `values` ---
    # Streams the full state after each step.
    print("\n--- DEMONSTRATING STREAM MODE: 'values' ---")
    print("Streaming the full state after each node execution.\n")
    for chunk in graph.stream(initial_input, stream_mode="values"):
        print(chunk)
        print("-" * 40)

    # --- 2. Stream Mode: `updates` ---
    # Streams only the updates to the state after each step.
    print("\n\n--- DEMONSTRATING STREAM MODE: 'updates' ---")
    print("Streaming only the state updates after each node execution.\n")
    for chunk in graph.stream(initial_input, stream_mode="updates"):
        print(chunk)
        print("-" * 40)

    # --- 3. Stream Mode: `messages` ---
    # Streams LLM tokens in real-time.
    print("\n\n--- DEMONSTRATING STREAM MODE: 'messages' ---")
    print("Streaming LLM tokens as they are generated.\n")
    final_joke = ""
    for chunk in graph.stream(initial_input, stream_mode="messages"):
        # The 'messages' mode yields AIMessageChunk objects.
        # We can inspect the content and print it.
        if chunk.content:
            print(chunk.content, end="", flush=True)
            final_joke += chunk.content
    print("\n\n--- Final Assembled Joke ---")
    print(final_joke)


if __name__ == "__main__":
    main()
