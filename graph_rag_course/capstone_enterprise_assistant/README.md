# Capstone – Enterprise Knowledge Assistant

This final project packages everything into a **production-grade** application:
Graph-RAG backend, React front-end, CI/CD, observability.

## High-level architecture
```
┌────────┐   upload   ┌─────────────┐  query  ┌──────────┐
│  User  │◀──────────▶│React Client │◀───────▶│FastAPI   │
└────────┘            └─────────────┘         │Backend   │
                                              │ (LangGraph + MCP server)
                                              └─────┬────┘
                                                    ▼
                                           Neo4j ←→ Pinecone
```

## Components
| path | description |
|------|-------------|
| `backend/neo4j_ingest.py` | bulk & incremental ingest (CLI & lib) |
| `backend/mcp_server.py` | Graph-RAG MCP server (extends 01_mcp_server) |
| `backend/langgraph_app.py` | multi-agent graph with tracing & timeouts |
| `frontend/` | React chat UI + admin dashboard |
| `deploy/` | Dockerfile, docker-compose, Helm chart |
| `tests/` | pytest unit & e2e suites |

Follow the step-by-step guide below to stand up the entire system.

---
## 1. Backend setup
```bash
cd backend
python -m pip install -r requirements.txt
python neo4j_ingest.py --init  # create schema & vector index
```

## 2. Front-end dev server
```bash
cd ../frontend
npm install && npm run dev
```

## 3. Local docker compose
```bash
cd ../deploy
docker compose up --build
```

## 4. Run tests
```bash
pytest -q
```

Happy shipping! 🚀
