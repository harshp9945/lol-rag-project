FROM python:3.11-slim

# Keep Python output unbuffered so logs show up immediately in the host
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first (this layer caches, so rebuilds are fast
# unless requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the embedding model at build time so the container starts
# fast and needs no network at runtime
RUN python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='sentence-transformers/all-MiniLM-L6-v2')"

# Copy the application code
COPY rag_assistant/ ./rag_assistant/
COPY knowledge_base/ ./knowledge_base/
COPY static/ ./static/
COPY api.py cli.py ./

# Cloud hosts provide a PORT env var; default to 8000 for local runs
ENV PORT=8000
EXPOSE 8000

# Bind to 0.0.0.0 (not 127.0.0.1) so the service is reachable from outside
# the container. Use the PORT env var the host provides.
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT}"]