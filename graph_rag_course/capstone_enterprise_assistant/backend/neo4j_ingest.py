"""CLI for bulk & incremental ingest into Neo4j with vector & entity graph."""
import argparse, os, glob
from pathlib import Path
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import spacy, json, sys
from dotenv import load_dotenv

load_dotenv()
embed = SentenceTransformer("all-MiniLM-L6-v2")
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
    nlp = spacy.load("en_core_web_sm")

uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
user = os.getenv("NEO4J_USER", "neo4j")
pwd = os.getenv("NEO4J_PASSWORD", "pass1234")

def ensure_schema(tx):
    tx.run(
        """
        CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE;
        CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE;
        CREATE VECTOR INDEX IF NOT EXISTS document_embedding FOR (d:Document) ON (d.embedding) OPTIONS {indexConfig:{`vector.dimensions`:384,`vector.similarity_function`:'cosine'}}
        """
    )

def ingest_file(sess, file_path: Path):
    text = file_path.read_text(encoding="utf-8")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    embeds = embed.encode(chunks)
    for idx, chunk in enumerate(chunks):
        did = f"{file_path.name}_{idx}"
        ents = [e.text for e in nlp(chunk).ents]
        sess.run(
            """
            MERGE (d:Document {id:$id}) SET d.text=$txt, d.source=$src, d.embedding=$emb
            WITH d UNWIND $ents AS ename MERGE (e:Entity {name:ename}) MERGE (d)-[:MENTIONS]->(e)
            """,
            id=did, txt=chunk, src=file_path.name, emb=embeds[idx].tolist(), ents=ents
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest .txt or .md files into Neo4j.")
    parser.add_argument("paths", nargs="+", help="Files or glob patterns")
    args = parser.parse_args()

    files = []
    for pattern in args.paths:
        files.extend(glob.glob(pattern))
    files = [Path(f) for f in files if Path(f).is_file()]
    if not files:
        print("No input files found", file=sys.stderr)
        sys.exit(1)

    driver = GraphDatabase.driver(uri, auth=(user, pwd))
    with driver.session() as sess:
        sess.execute_write(ensure_schema)
        for fp in files:
            print("Ingesting", fp)
            ingest_file(sess, fp)
    driver.close()
    print("Done ingesting", len(files), "files")
