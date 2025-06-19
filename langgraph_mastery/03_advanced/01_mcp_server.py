"""
MCP Server Implementation for Advanced LangGraph

This script demonstrates how to create a Model Context Protocol (MCP) server
that exposes tools for use by LangGraph agents.

The MCP server can be run as a standalone process and provides a standardized
interface for tools that can be used by any MCP client.
"""

import os
import sys
import logging
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text
try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    logger.error("MCP library not found. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "mcp"])
    from mcp.server.fastmcp import FastMCP

# Extra dependencies for semantic search, sentiment analysis and entity recognition
import subprocess
import json

try:
    # Vector store + embeddings
    from langchain_community.vectorstores.faiss import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.docstore.document import Document
except ImportError:
    logger = logging.getLogger(__name__)
    logger.info("Installing langchain-community, faiss-cpu and sentence-transformers …")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "langchain", "langchain-community", "faiss-cpu", "sentence-transformers"])
    from langchain_community.vectorstores.faiss import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.docstore.document import Document

# Sentiment
try:
    from textblob import TextBlob
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "textblob"])
    from textblob import TextBlob

# spaCy for NER
try:
    import spacy
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "spacy"])
    import spacy

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

mcp = FastMCP("KnowledgeTools")

# Vector store initialisation (semantic search)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KB_DIR = os.path.join(BASE_DIR, "knowledge_base")  # directory containing documents
INDEX_DIR = os.path.join(BASE_DIR, "faiss_index")

vector_store = None  # global store

def _init_vector_store():
    """Build or load FAISS vector store for semantic KB search."""
    global vector_store

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Try load
    if os.path.exists(INDEX_DIR):
        try:
            vector_store = FAISS.load_local(INDEX_DIR, embeddings)
            logger.info("Loaded existing FAISS index from disk")
            return
        except Exception as err:
            logger.warning(f"Failed loading FAISS index, rebuilding… ({err})")

    # Build from docs
    docs: List[Document] = []
    if not os.path.isdir(KB_DIR):
        logger.warning("Knowledge base dir not found – semantic search disabled")
        return

    for root, _dirs, files in os.walk(KB_DIR):
        for fname in files:
            if not fname.lower().endswith((".txt", ".md", ".json",".pdf")):
                continue
            if fname.lower().endswith(".pdf"):
                fpath = os.path.join(root, fname)
                content = extract_text_from_pdf(fpath)
            else:
                fpath = os.path.join(root, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as fp:
                        content = fp.read()
                except Exception as e:
                    logger.warning(f"Skip {fpath}: {e}")
                    continue

            meta = {"source": fpath}
            # if json contain content field
            if fname.lower().endswith(".json"):
                try:
                    data = json.loads(content)
                    content = data.get("content", content)
                    meta.update({k: v for k, v in data.items() if k != "content"})
                except json.JSONDecodeError:
                    pass

            docs.append(Document(page_content=content, metadata=meta))

    if not docs:
        logger.warning("No documents found in knowledge base – semantic search disabled")
        return

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(INDEX_DIR)
    logger.info("Built FAISS index for knowledge base")

_init_vector_store()

@mcp.tool()
def search_knowledge_base(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Search the knowledge base for information related to the query.
    
    Args:
        query: The search query
        max_results: Maximum number of results to return
        
    Returns:
        List of documents with content and metadata
    """
    logger.info(f"Searching knowledge base for: {query}")

    if vector_store is None:
        logger.warning("Vector store unavailable, returning empty list")
        return []

    docs_and_scores = vector_store.similarity_search_with_score(query, k=max_results)
    output: List[Dict[str, Any]] = []
    for doc, score in docs_and_scores:
        relevance = 1 / (1 + score) if score is not None else 1.0  # convert distance to similarity
        output.append({
            "content": doc.page_content,
            "source": doc.metadata.get("source", "unknown"),
            "relevance": round(relevance, 2)
        })

    return output

@mcp.tool()
def analyze_sentiment(text: str) -> Dict[str, Any]:
    """
    Analyze the sentiment of the provided text.
    
    Args:
        text: The text to analyze
        
    Returns:
        Sentiment analysis results
    """
    logger.info(f"Analyzing sentiment for text: {text[:50]}…")
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity  # −1..1
    subjectivity = blob.sentiment.subjectivity  # 0..1

    if polarity > 0.1:
        sentiment = "positive"
    elif polarity < -0.1:
        sentiment = "negative"
    else:
        sentiment = "neutral"

    return {
        "sentiment": sentiment,
        "polarity": round(polarity, 3),
        "subjectivity": round(subjectivity, 3)
    }

@mcp.tool()
def extract_entities(text: str) -> List[Dict[str, str]]:
    """
    Extract named entities from the provided text.
    
    Args:
        text: The text to analyze
        
    Returns:
        List of extracted entities with their types
    """
    logger.info(f"Extracting entities from text: {text[:50]}…")
    doc = nlp(text)
    return [{"entity": ent.text, "type": ent.label_} for ent in doc.ents]

if __name__ == "__main__":
    import uvicorn

    logger.info("Starting MCP server...")
    logger.info("The FastMCP object appears to be a self-contained FastAPI app.")
    logger.info("Running it directly with uvicorn on http://0.0.0.0:8000")
    
    mcp.run(transport="sse")
