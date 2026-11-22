# ==============================================================================
# TARK Studio — Production Multi-Arch Dockerfile
# Supported Architectures: linux/amd64, linux/arm64 (Oracle Ampere A1)
# ==============================================================================

FROM python:3.12-slim-bookworm AS base

# Prevent Python from writing .pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install minimal OS dependencies for network healthchecks and OCR
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Install Python production dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-cache the ONNX reranker model to ensure zero-latency startup and offline reliability
RUN python -c "from huggingface_hub import hf_hub_download; \
    hf_hub_download(repo_id='Xenova/ms-marco-MiniLM-L-6-v2', filename='onnx/model_int8.onnx'); \
    hf_hub_download(repo_id='Xenova/ms-marco-MiniLM-L-6-v2', filename='tokenizer.json')"

# Copy application source and database migration configurations
COPY app /app/app
COPY alembic /app/alembic
COPY alembic.ini /app/alembic.ini
COPY run.py /app/run.py
COPY docker-entrypoint.sh /app/docker-entrypoint.sh

# Ensure uploads directory exists and entrypoint is executable
RUN mkdir -p /app/uploads && chmod +x /app/docker-entrypoint.sh

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

ENTRYPOINT ["/app/docker-entrypoint.sh"]
