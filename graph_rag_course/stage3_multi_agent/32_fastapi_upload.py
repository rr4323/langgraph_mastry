"""FastAPI endpoint to upload new docs and ingest into Neo4j live."""
import os
from fastapi import FastAPI, UploadFile, HTTPException
from neo4j import GraphDatabase
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import spacy, uvicorn

load_dotenv()
app = FastAPI()
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
driver = GraphDatabase.driver(uri, auth=(user, pwd))

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

@app.post("/upload")
async def upload(file: UploadFile):
    if not file.filename.endswith(('.txt', '.md')):
        raise HTTPException(status_code=400, detail="Only txt or md allowed")
    text = (await file.read()).decode()
    chunks = splitter.split_text(text)
    embeds = embed.encode(chunks)
    with driver.session() as sess:
        for i, chunk in enumerate(chunks):
            cid = f"{file.filename}_{i}"
            ents = [e.text for e in nlp(chunk).ents]
            sess.run(
                """
                MERGE (d:Document {id:$id}) SET d.text=$txt, d.source=$src, d.embedding=$emb
                WITH d UNWIND $ents AS ename MERGE (e:Entity {name:ename}) MERGE (d)-[:MENTIONS]->(e)
                """,
                id=cid, txt=chunk, src=file.filename, emb=embeds[i].tolist(), ents=ents
            )
    return {"chunks": len(chunks)}

if __name__ == "__main__":
    uvicorn.run("32_fastapi_upload:app", host="0.0.0.0", port=8000, reload=True)
