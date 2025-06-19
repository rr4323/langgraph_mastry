"""Graph-RAG MCP server extending previous 01_mcp_server with Neo4j + Pinecone."""
import os, sys, logging
from pathlib import Path
from dotenv import load_dotenv
from langchain_mcp import MCPServer
from neo4j import GraphDatabase
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from stage2_neo4j_graph_rag.12_graph_rag import retrieve as graph_retrieve

load_dotenv()
logger = logging.getLogger("capstone.mcp")

mcp = MCPServer(name="graph_rag_tools", description="Enterprise Graph-RAG tools")
embed = SentenceTransformer("all-MiniLM-L6-v2")

uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
user = os.getenv("NEO4J_USER", "neo4j")
pwd = os.getenv("NEO4J_PASSWORD", "pass1234")
PINE_KEY = os.getenv("PINECONE_API_KEY")

@mcp.tool()
def hybrid_search(query: str, k: int = 5):
    """Search Pinecone vectors, then expand via Neo4j entities."""
    if PINE_KEY:
        pc = Pinecone(api_key=PINE_KEY)
        idx = pc.Index(os.getenv("PINECONE_INDEX", "capstone-demo"))
        vec = embed.encode(query).tolist()
        matches = idx.query(vector=vec, top_k=k, include_metadata=True)["matches"]
        docs = [m["metadata"]["source"] for m in matches]
        # expand via Neo4j
        driver = GraphDatabase.driver(uri, auth=(user, pwd))
        with driver.session() as sess:
            extra = sess.run(
                """MATCH (d:Document)-[:MENTIONS]->(e)<-[:MENTIONS]-(d2) WHERE d.id IN $ids RETURN d2.text LIMIT 10""",
                ids=docs,
            ).value()
        return docs + extra
    # fallback to stage2 Graph-RAG
    state = graph_retrieve({"question": query})
    return state["context"]

if __name__ == "__main__":
    transport = "stdio"
    if len(sys.argv) > 1:
        transport = sys.argv[1]
    mcp.run(transport=transport)
