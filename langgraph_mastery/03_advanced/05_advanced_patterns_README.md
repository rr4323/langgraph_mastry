# Advanced LangGraph Patterns

This directory contains advanced LangGraph patterns and examples that demonstrate more sophisticated capabilities of the framework for enterprise-grade applications.

## Examples Overview

### 1. MCP Server (`01_mcp_server.py`)
- Implements a proper Model Context Protocol (MCP) server using the official `mcp` library
- Exposes knowledge base search, sentiment analysis, and entity extraction tools
- Configurable transport options (stdio, TCP)
- Production-ready logging and error handling

### 2. MCP Client (`02_mcp_client.py`)
- Implements an MCP client that connects to the MCP server
- Uses `langchain-mcp-adapters` for tool integration
- Creates a LangGraph agent that uses remote tools via MCP
- Demonstrates proper connection and error handling

### 3. Multi-Agent System with MCP (`03_multi_agent_mcp.py`)
- Implements a multi-agent system with specialized roles (Coordinator, Researcher, Analyst, Writer)
- Agents communicate through the Model Context Protocol (MCP)
- Demonstrates agent coordination and collaboration
- Uses the MCP server for tool sharing across agents

### 4. Durable Execution (`04_durable_execution.py`)
- Demonstrates LangGraph's checkpointing capabilities
- Implements persistent workflows that can resume after failures
- Uses SQLite for state persistence
- Shows how to handle errors and resume execution

### 5a. Hierarchical Agents (`05a_hierarchical_agents.py`)
- Implements a hierarchical team structure with supervisor and worker agents
- Demonstrates task delegation and coordination from supervisor to workers
- Shows how specialized worker agents (Researcher, Analyst, Writer, etc.) collaborate
- Includes supervisor review and feedback mechanisms

### 5b. Event-Driven Architecture (`05b_event_driven.py`)
- Implements an event-driven system that reacts to various event types
- Demonstrates handling of user messages, system alerts, data updates, timers, and API calls
- Shows event prioritization and asynchronous processing
- Includes tools for event generation and simulation

### 5c. Feedback Loops (`05c_feedback_loops.py`)
- Implements self-improving systems with feedback mechanisms
- Demonstrates quality evaluation metrics for AI responses
- Shows how to analyze feedback and create improvement plans
- Includes performance history tracking and learning extraction

### 5d. Dynamic Graph Construction (`05d_dynamic_graphs.py`)
- Demonstrates how to construct and modify LangGraph workflows at runtime
- Shows dynamic node selection and workflow planning based on query analysis
- Implements a flexible node registry system for modular components
- Includes specialized nodes for different tasks that can be composed dynamically

### 5e. Hybrid Human-AI Systems (`05e_hybrid_human_ai.py`)
- Implements collaborative decision-making between humans and AI
- Demonstrates confidence-based routing of decisions
- Shows different handling for simple, complex, critical, and creative decisions
- Includes AI analysis with human oversight for critical decisions

### 5f. Human-in-the-Loop Integration (`05_human_in_the_loop.py`)
- Implements workflows with human oversight and intervention
- Demonstrates how to pause execution for human feedback
- Shows how to incorporate human decisions into the workflow
- Includes a workflow manager for handling multiple concurrent workflows

### 6. Subgraphs and Branching (`06_subgraphs_branching.py`)
- Demonstrates how to create modular subgraphs
- Shows complex branching logic based on state conditions
- Implements hierarchical graph structures
- Illustrates dynamic graph construction

### 7. LangGraph Studio and Deployment (`07_studio_deployment.py`)
- Shows how to integrate with LangGraph Studio for visualization
- Demonstrates multiple deployment options (Docker, Kubernetes, Serverless, LangServe)
- Includes production-ready code samples and configuration templates
- Provides monitoring, tracing, and observability with LangSmith
- Implements a FastAPI wrapper for production deployment

## Key Concepts

### Model Context Protocol (MCP)
MCP is a protocol for sharing tools and context between different components in a distributed system. It allows for:
- Tool sharing across services
- Standardized communication between components
- Secure and efficient message passing

### Durable Execution
Durable execution allows workflows to persist through failures and run for extended periods:
- Checkpointing state at critical points
- Resuming execution from the last checkpoint
- Handling errors gracefully
- Supporting long-running workflows

### Human-in-the-Loop
Human-in-the-loop integration allows for human oversight and intervention:
- Pausing execution for human review
- Incorporating human feedback
- Allowing humans to modify agent behavior
- Ensuring quality and safety

### Subgraphs and Branching
Advanced graph patterns enable more complex workflows:
- Modular subgraphs for reusable components
- Complex branching based on state conditions
- Dynamic graph construction
- Hierarchical workflows

### Deployment Options
Production-ready deployment strategies include:
- Docker containerization for consistent environments
- Kubernetes orchestration for scalability and resilience
- Serverless functions for event-driven architectures
- LangServe integration for LangChain/LangGraph optimization
- FastAPI wrappers for RESTful APIs
- Monitoring and observability with LangSmith
- Security considerations for API keys and secrets
- Performance optimization for production workloads

## Getting Started

Each example includes detailed comments and instructions. To run an example:

```bash
python 01_mcp_server.py
```

For examples that require multiple components (like MCP server and client), you'll need to run them in separate terminals.

## Prerequisites

- Python 3.9+
- LangGraph 0.0.15+
- LangChain 0.1.0+
- Additional dependencies as specified in each example

## Environment Variables

Create a `.env` file with the following variables:
```
GOOGLE_API_KEY=your_google_api_key
```

## Additional Resources

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangGraph GitHub Repository](https://github.com/langchain-ai/langgraph)
- [LangGraph Studio](https://smith.langchain.com/langgraph)

## Usage

Each module can be run independently. Simply execute the Python file to see the pattern in action:

```bash
python 05a_hierarchical_agents.py
```

## Requirements

All modules require:
- LangGraph for workflow graph construction
- LangChain for Google Generative AI integration
- Google Gemini Pro model access (API key required)
- Pydantic for typed state dictionaries
- dotenv for environment variable loading

Make sure to set your `GOOGLE_API_KEY` in the `.env` file before running the examples.

## Learning Path

These patterns build on the basic and intermediate concepts covered in earlier sections. They demonstrate how to combine multiple LangGraph features to create sophisticated AI systems that can:

1. Collaborate in hierarchical teams
2. React to events in real-time
3. Learn and improve from feedback
4. Adapt their structure dynamically
5. Collaborate effectively with humans

Study these patterns to understand how to build enterprise-grade AI applications using LangGraph.
