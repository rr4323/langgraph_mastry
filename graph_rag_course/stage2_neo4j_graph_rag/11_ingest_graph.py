"""Ingest Wikipedia docs into Neo4j with embeddings + entity graph."""
import os
from pathlib import Path
from dotenv import load_dotenv
from neo4j import GraphDatabase
import spacy
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()
BASE_DIR = Path(__file__).resolve().parent.parent / "stage1_vector_rag"  # reuse data
DATA_DIR = BASE_DIR / "data"

NL_MODEL = "en_core_web_sm"
try:
    nlp = spacy.load(NL_MODEL)
except OSError:
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "spacy", "download", NL_MODEL], check=True)
    nlp = spacy.load(NL_MODEL)

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

def get_driver():
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    pwd = os.getenv("NEO4J_PASSWORD", "pass1234")
    return GraphDatabase.driver(uri, auth=(user, pwd))

def ensure_vector_index(tx, dim: int = 384):
    tx.run(
        """
        CREATE VECTOR INDEX IF NOT EXISTS document_embedding IF NOT EXISTS
        FOR (d:Document) ON (d.embedding) OPTIONS {indexConfig: {`vector.dimensions`: $dim, `vector.similarity_function`: 'cosine'}}
        """,
        dim=dim,
    )

def main():
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    documents = []
    for fp in DATA_DIR.glob("*.txt"):
        text = fp.read_text(encoding="utf-8")
        for chunk in splitter.split_text(text):
            documents.append({"source": fp.name, "text": chunk})
    print("Chunks:", len(documents))

    embeddings = embed_model.encode([d["text"] for d in documents], show_progress_bar=True)

    driver = get_driver()
    with driver.session(database="neo4j") as session:
        session.execute_write(ensure_vector_index)
        # ingest docs
        for i, doc in enumerate(documents):
            eid = f"c{i}"
            entities = [ent.text for ent in nlp(doc["text"]).ents]
            session.run(
                """
                MERGE (d:Document {id:$id})
                  SET d.text = $text, d.source=$src, d.embedding=$emb
                WITH d
                UNWIND $ents AS ename
                    MERGE (e:Entity {name:ename})
                    MERGE (d)-[:MENTIONS]->(e)
                """,
                id=eid,
                text=doc["text"],
                src=doc["source"],
                emb=embeddings[i].tolist(),
                ents=entities,
            )
    driver.close()
    print("Finished ingesting to Neo4j")

if __name__ == "__main__":
    main()
