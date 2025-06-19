# Stage 1 – Vector-RAG Starter

Welcome! This is the first hands-on project of the **Graph-RAG Course**.  It teaches
classic vector-based Retrieval-Augmented Generation and shows how to swap a local
FAISS index for a managed **Pinecone** index with zero code changes.

## What you will build
1. **Download** ten short Wikipedia pages
2. **Embed** them with `all-MiniLM-L6-v2`
3. **Index** them (FAISS _⇄ Pinecone_) 
4. **Ask** questions through a two-node **LangGraph** workflow

```
(User question) → [Retrieve] → [Generate] → answer
```

---
## Setup
Install the Python deps (isolated from your main repo):
```bash
cd graph_rag_course/stage1_vector_rag
python -m pip install -r requirements.txt
```
Create a `.env` in the **same folder**:
```ini
# choose at least one chat model provider
OPENAI_API_KEY=sk-...      # or
# GOOGLE_API_KEY=...

# Optional – if set, Pinecone is used instead of FAISS
PINECONE_API_KEY=pc-...
PINECONE_ENV=us-central1-gcp
PINECONE_INDEX=stage1-demo
```
_No Pinecone variables → local FAISS file `faiss_index/` will be created._

---
## Run it!
```bash
python 01_prepare_data.py     # download articles (one-off)
python 02_build_index.py      # embed & index chunks
python 03_langgraph_rag.py    # ask questions
```
Sample queries:
* “When was Ada Lovelace born?”
* “Which language did Guido van Rossum create?”

The script prints the retrieved chunks before each answer so you can verify grounding.

Happy hacking — when you’re comfortable, move to **Stage 2 (Neo4j Graph-RAG)**.
