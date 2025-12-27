# Heartbeat Service Dockerfile
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Copy project files and source code
COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY services/heartbeat/ ./services/heartbeat/
# Create services __init__.py for package structure
RUN touch ./services/__init__.py

# Install Python dependencies
# Install base dependencies + mqtt extras
RUN pip install --no-cache-dir -e ".[mqtt]"

# Health check via MQTT connection (the service publishes health to MQTT)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import socket; s=socket.socket(); s.settimeout(5); s.connect(('${MQTT_HOST:-localhost}', int('${MQTT_PORT:-1883}'))); s.close()" || exit 1

# Run the heartbeat service
CMD ["python", "-m", "services.heartbeat"]
