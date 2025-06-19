# Stage 3 – Multi-Agent Graph-RAG

You now know classic vector RAG and single-agent Graph-RAG.  In this stage you
will orchestrate **multiple specialised agents** inside one LangGraph workflow:

1. **Coordinator** – decides which tool/agent to call next.
2. **Researcher** – runs Graph-RAG to collect facts.
3. **Analyst** – synthesises findings, checks gaps.
4. **Writer** – crafts the final user-facing answer.

You’ll also add:
* Self-critique feedback loop (rewrite if quality < threshold)
* FastAPI upload endpoint for live doc ingestion
* Simple Streamlit chat UI showing citations

---
## File structure
| File | Purpose |
|------|---------|
| `30_agents.py` | Defines Researcher, Analyst, Writer agents as LangChain chains |
| `31_graph.py` | LangGraph: Coordinator node, edges, feedback loop |
| `32_fastapi_upload.py` | Endpoint `/upload` -> chunk/embed -> Neo4j insertion |
| `33_streamlit_chat.py` | Minimal front-end calling FastAPI & showing sources |
| `requirements.txt` | Additional deps (FastAPI, uvicorn, streamlit) |

---
## Quick start
```bash
cd graph_rag_course/stage3_multi_agent
python -m pip install -r requirements.txt

# 1️⃣  Start the upload API (+ background ingest)
uvicorn 32_fastapi_upload:app --reload

# 2️⃣  Run the Streamlit chat
streamlit run 33_streamlit_chat.py
```

> Requires Neo4j populated via Stage-2.

Move to **Stage 4 – Capstone Enterprise Assistant** after experimenting!🍀
