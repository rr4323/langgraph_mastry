"""
LangGraph Studio Integration and Deployment Options

This script demonstrates how to integrate with LangGraph Studio for visualization
and covers various deployment options for production-ready LangGraph applications.
"""

import os
import sys
import json
import uuid
import logging
import asyncio
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Check for required packages
required_packages = ["langchain-google-genai", "langgraph", "langsmith"]
for package in required_packages:
    try:
        __import__(package.replace("-", "_"))
    except ImportError:
        logger.info(f"Installing {package}...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.graph.graph import CompiledGraph
from langchain_core.tracers.langchain import wait_for_all_tracers
import langsmith

# Define our state models
class Message(BaseModel):
    """A message in the conversation."""
    role: str
    content: str

class WorkflowState(BaseModel):
    """The state for our workflow."""
    query: str
    messages: List[Message] = Field(default_factory=list)
    research_results: Optional[str] = None
    analysis_results: Optional[str] = None
    final_response: Optional[str] = None
    errors: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

# Helper functions
def get_model(google_api_key: str, temperature: float = 0.4):
    """Get the language model."""
    return ChatGoogleGenerativeAI(
        model="gemini-pro",
        google_api_key=google_api_key,
        temperature=temperature,
        convert_system_message_to_human=True
    )

# Node functions for our workflow
async def research_node(state: Dict[str, Any], model) -> Dict[str, Any]:
    """Research information related to the query."""
    logger.info("Running research node")
    state_obj = WorkflowState.model_validate(state)
    
    try:
        # Prepare the messages for the model
        messages = [
            {"role": "system", "content": "You are a research assistant. Find relevant information for the given query."},
            {"role": "user", "content": f"Research this query: {state_obj.query}"}
        ]
        
        # Invoke the model
        response = await model.ainvoke(messages)
        research_results = response.content
        
        # Update the state
        state_obj.research_results = research_results
        state_obj.messages.append(Message(
            role="researcher",
            content=f"Research completed."
        ))
        
    except Exception as e:
        logger.error(f"Error in research node: {str(e)}", exc_info=True)
        state_obj.errors.append({
            "node": "research",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        })
    
    return state_obj.model_dump()

async def analysis_node(state: Dict[str, Any], model) -> Dict[str, Any]:
    """Analyze the research results."""
    logger.info("Running analysis node")
    state_obj = WorkflowState.model_validate(state)
    
    try:
        # Prepare the messages for the model
        messages = [
            {"role": "system", "content": "You are an analyst. Analyze the research information and provide insights."},
            {"role": "user", "content": f"Query: {state_obj.query}\n\nResearch information: {state_obj.research_results}"}
        ]
        
        # Invoke the model
        response = await model.ainvoke(messages)
        analysis_results = response.content
        
        # Update the state
        state_obj.analysis_results = analysis_results
        state_obj.messages.append(Message(
            role="analyst",
            content=f"Analysis completed."
        ))
        
    except Exception as e:
        logger.error(f"Error in analysis node: {str(e)}", exc_info=True)
        state_obj.errors.append({
            "node": "analysis",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        })
    
    return state_obj.model_dump()

async def response_node(state: Dict[str, Any], model) -> Dict[str, Any]:
    """Generate the final response."""
    logger.info("Running response node")
    state_obj = WorkflowState.model_validate(state)
    
    try:
        # Prepare the messages for the model
        messages = [
            {"role": "system", "content": "You are a response generator. Create a comprehensive response based on the research and analysis."},
            {"role": "user", "content": f"Query: {state_obj.query}\n\nResearch: {state_obj.research_results}\n\nAnalysis: {state_obj.analysis_results}"}
        ]
        
        # Invoke the model
        response = await model.ainvoke(messages)
        final_response = response.content
        
        # Update the state
        state_obj.final_response = final_response
        state_obj.messages.append(Message(
            role="assistant",
            content=final_response
        ))
        
    except Exception as e:
        logger.error(f"Error in response node: {str(e)}", exc_info=True)
        state_obj.errors.append({
            "node": "response",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        })
    
    return state_obj.model_dump()

# Create the workflow graph with LangSmith tracing
def create_workflow_graph(model, project_name: str = "langgraph_tutorial"):
    """Create the workflow graph with LangSmith tracing."""
    # Create the graph
    workflow = StateGraph(WorkflowState)
    
    # Add nodes to the graph
    workflow.add_node("research", lambda state: research_node(state, model))
    workflow.add_node("analysis", lambda state: analysis_node(state, model))
    workflow.add_node("response", lambda state: response_node(state, model))
    
    # Add edges
    workflow.add_edge("research", "analysis")
    workflow.add_edge("analysis", "response")
    workflow.add_edge("response", END)
    
    # Set the entry point
    workflow.set_entry_point("research")
    
    # Compile the graph with tracing
    return workflow.compile(
        name="LangGraph Studio Demo",
        project_name=project_name
    )

# Function to export the graph for LangGraph Studio
def export_graph_for_studio(graph: CompiledGraph, filename: str = "workflow_graph.json"):
    """Export the graph definition for LangGraph Studio."""
    try:
        # Get the graph definition
        graph_def = graph.get_graph_definition()
        
        # Save to a file
        with open(filename, "w") as f:
            json.dump(graph_def, f, indent=2)
        
        logger.info(f"Graph definition exported to {filename}")
        return filename
    except Exception as e:
        logger.error(f"Error exporting graph: {str(e)}", exc_info=True)
        return None

# Main function to run the workflow
async def run_workflow(query: str, project_name: str = "langgraph_tutorial"):
    """Run the workflow with LangGraph Studio integration."""
    logger.info(f"Running workflow for query: {query}")
    
    # Get the Google API key from environment
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        logger.error("GOOGLE_API_KEY not found in environment variables")
        print("Please set your GOOGLE_API_KEY in the .env file")
        return {"error": "API key not found"}
    
    # Check for LangSmith API key
    langsmith_api_key = os.getenv("LANGCHAIN_API_KEY")
    if not langsmith_api_key:
        logger.warning("LANGCHAIN_API_KEY not found. LangSmith tracing will not be available.")
        print("For LangSmith tracing, set your LANGCHAIN_API_KEY in the .env file")
    
    # Get the model
    model = get_model(google_api_key)
    
    # Create the workflow
    workflow = create_workflow_graph(model, project_name)
    
    # Export the graph for LangGraph Studio
    export_graph_for_studio(workflow)
    
    # Initialize the state
    initial_state = WorkflowState(query=query)
    
    # Run the workflow
    try:
        logger.info("Running the workflow...")
        result = await workflow.ainvoke(initial_state.model_dump())
        
        # Wait for all tracers to finish
        if langsmith_api_key:
            wait_for_all_tracers()
        
        return result
    except Exception as e:
        logger.error(f"Error running workflow: {str(e)}", exc_info=True)
        initial_state.errors.append({
            "node": "workflow",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        })
        return initial_state.model_dump()

# Deployment helper functions
def get_deployment_options():
    """Get the available deployment options."""
    return {
        "local": {
            "name": "Local Development",
            "description": "Run the LangGraph application locally for development and testing.",
            "requirements": ["Python 3.9+", "Required packages"],
            "command": "python app.py"
        },
        "docker": {
            "name": "Docker Container",
            "description": "Package the application in a Docker container for consistent deployment.",
            "requirements": ["Docker", "Docker Compose (optional)"],
            "command": "docker build -t langgraph-app . && docker run -p 8000:8000 langgraph-app"
        },
        "serverless": {
            "name": "Serverless Functions",
            "description": "Deploy as serverless functions for event-driven, scalable architecture.",
            "requirements": ["AWS Lambda/Google Cloud Functions/Azure Functions", "API Gateway"],
            "command": "serverless deploy"
        },
        "kubernetes": {
            "name": "Kubernetes",
            "description": "Deploy on Kubernetes for container orchestration and scaling.",
            "requirements": ["Kubernetes cluster", "kubectl", "Helm (optional)"],
            "command": "kubectl apply -f deployment.yaml"
        },
        "langserve": {
            "name": "LangServe",
            "description": "Deploy using LangServe for LangChain/LangGraph specific optimizations.",
            "requirements": ["LangServe", "FastAPI"],
            "command": "langserve deploy"
        }
    }

def generate_deployment_files(deployment_type: str, output_dir: str = "./deployment"):
    """Generate deployment files for the specified deployment type."""
    os.makedirs(output_dir, exist_ok=True)
    
    if deployment_type == "docker":
        # Generate Dockerfile
        dockerfile_content = """FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "app.py"]
"""
        with open(os.path.join(output_dir, "Dockerfile"), "w") as f:
            f.write(dockerfile_content)
        
        # Generate docker-compose.yml
        docker_compose_content = """version: '3'

services:
  langgraph-app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - GOOGLE_API_KEY=${GOOGLE_API_KEY}
      - LANGCHAIN_API_KEY=${LANGCHAIN_API_KEY}
      - LANGCHAIN_TRACING_V2=true
    volumes:
      - ./data:/app/data
"""
        with open(os.path.join(output_dir, "docker-compose.yml"), "w") as f:
            f.write(docker_compose_content)
        
        return ["Dockerfile", "docker-compose.yml"]
    
    elif deployment_type == "kubernetes":
        # Generate Kubernetes deployment.yaml
        k8s_deployment_content = """apiVersion: apps/v1
kind: Deployment
metadata:
  name: langgraph-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: langgraph-app
  template:
    metadata:
      labels:
        app: langgraph-app
    spec:
      containers:
      - name: langgraph-app
        image: langgraph-app:latest
        ports:
        - containerPort: 8000
        env:
        - name: GOOGLE_API_KEY
          valueFrom:
            secretKeyRef:
              name: langgraph-secrets
              key: google-api-key
        - name: LANGCHAIN_API_KEY
          valueFrom:
            secretKeyRef:
              name: langgraph-secrets
              key: langchain-api-key
        - name: LANGCHAIN_TRACING_V2
          value: "true"
---
apiVersion: v1
kind: Service
metadata:
  name: langgraph-app
spec:
  selector:
    app: langgraph-app
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
"""
        with open(os.path.join(output_dir, "deployment.yaml"), "w") as f:
            f.write(k8s_deployment_content)
        
        # Generate secret.yaml
        k8s_secret_content = """apiVersion: v1
kind: Secret
metadata:
  name: langgraph-secrets
type: Opaque
data:
  google-api-key: BASE64_ENCODED_GOOGLE_API_KEY
  langchain-api-key: BASE64_ENCODED_LANGCHAIN_API_KEY
"""
        with open(os.path.join(output_dir, "secret.yaml"), "w") as f:
            f.write(k8s_secret_content)
        
        return ["deployment.yaml", "secret.yaml"]
    
    elif deployment_type == "langserve":
        # Generate app.py for LangServe
        langserve_app_content = """from fastapi import FastAPI
from langserve import add_routes
import os
from dotenv import load_dotenv
from your_module import create_workflow_graph, get_model

# Load environment variables
load_dotenv()

# Get API keys
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")

# Create the app
app = FastAPI(
    title="LangGraph Application",
    version="1.0",
    description="A LangGraph application deployed with LangServe"
)

# Create the model and graph
model = get_model(google_api_key)
workflow = create_workflow_graph(model)

# Add routes
add_routes(
    app,
    workflow,
    path="/api/workflow",
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
"""
        with open(os.path.join(output_dir, "app.py"), "w") as f:
            f.write(langserve_app_content)
        
        # Generate requirements.txt
        requirements_content = """langchain>=0.1.0
langgraph>=0.0.15
langchain-google-genai>=0.0.5
langserve>=0.0.30
fastapi>=0.104.1
uvicorn>=0.24.0
python-dotenv>=1.0.0
pydantic>=2.5.2
"""
        with open(os.path.join(output_dir, "requirements.txt"), "w") as f:
            f.write(requirements_content)
        
        return ["app.py", "requirements.txt"]
    
    else:
        return []

# Interactive CLI
async def main():
    """Run the LangGraph Studio and deployment example."""
    print("=" * 80)
    print("LangGraph Studio Integration and Deployment Options")
    print("=" * 80)
    print("\nThis example demonstrates how to integrate with LangGraph Studio for visualization")
    print("and covers various deployment options for production-ready LangGraph applications.")
    
    # Check for LangSmith API key
    langsmith_api_key = os.getenv("LANGCHAIN_API_KEY")
    if not langsmith_api_key:
        print("\nWARNING: LANGCHAIN_API_KEY not found. LangSmith tracing will not be available.")
        print("For LangSmith tracing, set your LANGCHAIN_API_KEY in the .env file")
    else:
        print("\nLangSmith API key found. Tracing will be enabled.")
    
    # Get user input
    query = input("\nEnter your query: ")
    
    # Get project name
    project_name = input("\nEnter a project name for LangSmith (default: langgraph_tutorial): ") or "langgraph_tutorial"
    
    print("\nProcessing with LangGraph Studio integration...")
    result = await run_workflow(query, project_name)
    
    # Display the result
    print("\n" + "=" * 80)
    print("Workflow Result:")
    print("=" * 80)
    
    if "final_response" in result and result["final_response"]:
        print("\nFinal Response:")
        print("-" * 40)
        print(result["final_response"])
        print("-" * 40)
    else:
        print("\nNo final response generated.")
    
    # Show any errors
    if "errors" in result and result["errors"]:
        print("\nErrors:")
        for error in result["errors"]:
            print(f"- {error.get('node', 'unknown')}: {error.get('error', 'unknown error')}")
    
    # Show LangGraph Studio information
    print("\n" + "=" * 80)
    print("LangGraph Studio Integration")
    print("=" * 80)
    
    if langsmith_api_key:
        print("\nYour workflow run has been traced in LangSmith.")
        print(f"Project: {project_name}")
        print("\nTo view the trace:")
        print("1. Go to https://smith.langchain.com")
        print("2. Navigate to the project you specified")
        print("3. Find your run in the list")
        
        print("\nThe graph definition has been exported to 'workflow_graph.json'.")
        print("You can import this file into LangGraph Studio for visualization.")
    else:
        print("\nLangSmith tracing was not enabled. To enable it:")
        print("1. Sign up for LangSmith at https://smith.langchain.com")
        print("2. Get your API key")
        print("3. Set the LANGCHAIN_API_KEY environment variable")
    
    # Show deployment options
    print("\n" + "=" * 80)
    print("Deployment Options")
    print("=" * 80)
    
    deployment_options = get_deployment_options()
    
    print("\nAvailable deployment options:")
    for i, (key, option) in enumerate(deployment_options.items(), 1):
        print(f"{i}. {option['name']}: {option['description']}")
    
    # Ask if the user wants to generate deployment files
    print("\nWould you like to generate deployment files? (y/n)")
    generate_files = input("> ").lower() == "y"
    
    if generate_files:
        print("\nSelect a deployment option:")
        for i, (key, option) in enumerate(deployment_options.items(), 1):
            print(f"{i}. {option['name']}")
        
        while True:
            try:
                choice = int(input("\nEnter your choice (1-5): "))
                if 1 <= choice <= len(deployment_options):
                    deployment_type = list(deployment_options.keys())[choice-1]
                    break
                else:
                    print(f"Please enter a number between 1 and {len(deployment_options)}.")
            except ValueError:
                print("Please enter a valid number.")
        
        # Generate the files
        output_dir = "./deployment"
        files = generate_deployment_files(deployment_type, output_dir)
        
        if files:
            print(f"\nGenerated the following deployment files in {output_dir}:")
            for file in files:
                print(f"- {file}")
            
            print("\nNext steps:")
            option = deployment_options[deployment_type]
            print(f"1. Install the required dependencies: {', '.join(option['requirements'])}")
            print(f"2. Deploy using: {option['command']}")
        else:
            print(f"\nNo deployment files generated for {deployment_type}.")
    
    print("\nThank you for using the LangGraph Studio and Deployment example!")

# Sample FastAPI app for production deployment
def create_fastapi_app():
    """Create a FastAPI app for production deployment."""
    try:
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel
        
        app = FastAPI(
            title="LangGraph API",
            description="API for LangGraph workflows",
            version="1.0.0"
        )
        
        class QueryRequest(BaseModel):
            query: str
            project_name: str = "langgraph_production"
        
        class QueryResponse(BaseModel):
            result: Dict[str, Any]
        
        @app.post("/api/query", response_model=QueryResponse)
        async def process_query(request: QueryRequest):
            try:
                result = await run_workflow(request.query, request.project_name)
                return {"result": result}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/health")
        def health_check():
            return {"status": "healthy"}
        
        return app
    except ImportError:
        logger.error("FastAPI not installed. Cannot create FastAPI app.")
        return None

if __name__ == "__main__":
    asyncio.run(main())
