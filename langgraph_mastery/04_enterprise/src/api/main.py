import sys
import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Add parent directory to path to allow imports from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config import get_config
from src.main import initialize_app, process_query

config = get_config()
logger = logging.getLogger(__name__)

app = FastAPI(
    title=config.app_name,
    version="1.0.0",
    description="API for the Enterprise Knowledge Assistant"
)

# Initialize the graph on startup
graph = None

@app.on_event("startup")
def startup_event():
    global graph
    logger.info("FastAPI application startup...")
    graph = initialize_app()
    logger.info("Graph initialized and ready.")

class QueryRequest(BaseModel):
    query: str

@app.get("/health", summary="Health Check")
def health_check():
    """Check if the application is running."""
    return {"status": "healthy"}

@app.post("/query", summary="Process a Query")
def handle_query(request: QueryRequest):
    """Process a user query and return the assistant's response."""
    if not graph:
        raise HTTPException(status_code=503, detail="Graph not initialized. Please wait and try again.")
    
    try:
        result = process_query(graph, request.query)
        return result
    except Exception as e:
        logger.error(f"API error handling query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal error occurred while processing the query.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.api.host, port=config.api.port)
