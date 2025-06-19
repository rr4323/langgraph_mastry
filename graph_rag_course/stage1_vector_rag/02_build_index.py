"""Split, embed and index documents.
Uses Pinecone if PINECONE_API_KEY is set, otherwise FAISS.
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
INDEX_DIR = BASE_DIR / "faiss_index"

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# 1. read docs
texts = []
for fp in DATA_DIR.glob("*.txt"):
    texts.append((fp.name, fp.read_text(encoding="utf-8")))
print(f"Loaded {len(texts)} documents")

# 2. split
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = []
for name, txt in texts:
    for chunk in splitter.split_text(txt):
        chunks.append({"text": chunk, "source": name})
print(f"Created {len(chunks)} chunks")

# 3. embed
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

if PINECONE_API_KEY:
    from pinecone import Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index_name = os.getenv("PINECONE_INDEX", "stage1-demo")
    if index_name not in pc.list_indexes().names():
        pc.create_index(name=index_name, dimension=384, metric="cosine")
    index = pc.Index(index_name)
    print("Upserting to Pinecone …")
    vectors = [(f"c{i}", embeddings.embed_query(c["text"]), c) for i, c in enumerate(chunks)]
    index.upsert(vectors)
    print("Done → Pinecone")
else:
    print("Building local FAISS …")
    INDEX_DIR.mkdir(exist_ok=True)
    vs = FAISS.from_texts([c["text"] for c in chunks], embeddings, metadatas=[{"source": c["source"]} for c in chunks])
    vs.save_local(INDEX_DIR)
    print("Saved FAISS index to", INDEX_DIR)
