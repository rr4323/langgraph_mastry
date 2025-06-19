# Stage 2 – Neo4j Graph-RAG

Welcome to the next milestone!  You will augment vector search with a **knowledge graph**
built in Neo4j and observe how adding entity-relationships improves answers.

What you will build
1. Spin up Neo4j via Docker and create a vector index.
2. Ingest the same Wikipedia corpus, but also extract entities with spaCy and
   write `(:Document)-[:MENTIONS]->(:Entity)` relationships.
3. Build a Graph-RAG LangGraph workflow:
   * vector search → find top docs
   * Cypher hop expansion (related entities / docs)
   * feed enriched context to the LLM
4. Evaluate vs. plain vector RAG.

---
## Prerequisites
```
# Docker + Docker Compose
# Python deps
python -m pip install -r requirements.txt
```
Ensure your Stage-1 `.env` still exists; add Neo4j creds:
```ini
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=pass1234
```

---
## 0️⃣  Start Neo4j
```bash
docker compose up -d          # will use docker-compose.yml provided here
```
Browse http://localhost:7474 and change the default password to the one above.

---
## 1️⃣  Ingest documents & entities
```bash
python 11_ingest_graph.py     # chunks, embeds, writes nodes & vectors
```

---
## 2️⃣  Run Graph-RAG pipeline
```bash
python 12_graph_rag.py
```
Ask:
* “Which other pioneers are related to Ada Lovelace?”
* “What areas are linked to artificial intelligence?”

---
## 3️⃣  Evaluate
```bash
jupyter notebook 13_eval_graph.ipynb
```
Compare answer correctness and context token count.

Move to **Stage 3 – Multi-Agent Graph-RAG** once comfy!
