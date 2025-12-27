# Memory Service Dockerfile
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy project files and source code
COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY services/memory/ ./services/memory/
# Create services __init__.py for package structure
RUN touch ./services/__init__.py

# Install CPU-only PyTorch first (much smaller than CUDA version)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install Python dependencies
# Install base dependencies + memory + mqtt + api extras
RUN pip install --no-cache-dir -e ".[memory,mqtt,api]"

# Embedding models will be downloaded at runtime to a mounted PVC
# Set the cache directory for sentence-transformers/huggingface
ENV HF_HOME=/data/models
ENV SENTENCE_TRANSFORMERS_HOME=/data/models

# Expose HTTP API port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run the memory service
CMD ["python", "-m", "services.memory"]
