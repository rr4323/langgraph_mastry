# --- Backend container for Enterprise Knowledge Assistant ---
# Multi-stage keeps the final image slim (~300 MB)
FROM python:3.10-slim AS base
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
WORKDIR /app

# Install system deps (spaCy model needs gcc + libstdc++)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && rm -rf /var/lib/apt/lists/*

# ---------- Install Python requirements ----------
COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# ---------- Copy source ----------
COPY backend/ ./backend/

# Default command: run FastAPI backend via Uvicorn
CMD ["uvicorn", "backend.langgraph_app:app", "--host", "0.0.0.0", "--port", "8000"]
