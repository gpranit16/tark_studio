# ==============================================================================
# TARK Studio — Production Multi-Arch Dockerfile
# Supported Architectures: linux/amd64, linux/arm64
# ==============================================================================

FROM python:3.12-slim-bookworm AS base

# Prevent Python from writing .pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install minimal OS dependencies for network healthchecks and ultra-fast in-container OCR
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    tesseract-ocr \
    tesseract-ocr-eng \
    libtesseract-dev \
    && rm -rf /var/lib/apt/lists/*

ENV OMP_THREAD_LIMIT=1 \
    TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata

# Install Python production dependencies
COPY backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-cache the ONNX reranker model to ensure zero-latency startup and offline reliability
RUN python -c "from huggingface_hub import hf_hub_download; \
    hf_hub_download(repo_id='Xenova/ms-marco-MiniLM-L-6-v2', filename='onnx/model_quantized.onnx'); \
    hf_hub_download(repo_id='Xenova/ms-marco-MiniLM-L-6-v2', filename='tokenizer.json')"

# Copy application source and database migration configurations
COPY backend/app /app/app
COPY backend/alembic /app/alembic
COPY backend/alembic.ini /app/alembic.ini
COPY backend/run.py /app/run.py
COPY backend/docker-entrypoint.sh /app/docker-entrypoint.sh

# Ensure uploads directory exists and entrypoint is executable
RUN mkdir -p /app/uploads && chmod +x /app/docker-entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["/app/docker-entrypoint.sh"]
