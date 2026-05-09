# Backend Dockerfile for Conference Paper Agent — API-only image
# Installs only the minimal deps needed to serve the v2 FastAPI app.
# Heavy ML deps (torch, docling, marker-pdf) are NOT included here
# because the API serves pre-computed cached results, not live inference.

FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Minimal system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install only API server deps (fastapi, uvicorn, pydantic) — takes ~30 s
COPY requirements-api.txt ./
RUN pip install --no-cache-dir -r requirements-api.txt

# Copy source and data files
COPY src/ ./src/
COPY experiments/ ./experiments/

RUN mkdir -p logs output/eval

EXPOSE 8000

# v2 API — served by uvicorn; $PORT injected by Railway
CMD uvicorn src.v2.api.app:create_app --factory --host 0.0.0.0 --port ${PORT:-8000}
