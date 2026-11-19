# Multi-stage production Dockerfile for Industrial IoT Digital Twin & FastAPI Backend
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install -r requirements.txt

# Copy application source code and assets
COPY src/ ./src/
COPY frontend/ ./frontend/
COPY data/ ./data/
COPY scripts/ ./scripts/
COPY amazon/ ./amazon/
COPY main.py .
COPY .streamlit/ ./.streamlit/

# Pre-generate simulated datasets inside container
RUN python scripts/generate_simulated_dataset.py

# Healthcheck probe
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:${PORT}/ || exit 1

EXPOSE 8000

# Default entrypoint starts FastAPI with Digital Twin & Frontend
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
