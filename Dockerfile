# GGUF Forge Dockerfile
# Python 3.12 with build tools for llama.cpp, ODBC drivers for MSSQL

FROM python:3.12-slim-bookworm

LABEL maintainer="GGUF Forge"
LABEL description="Automatic GGUF Model Conversion Service"

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Set Python to run unbuffered (better for Docker logs)
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# HuggingFace transfer optimizations
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV HF_XET_HIGH_PERFORMANCE=1

# Working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Git for llama.cpp cloning and app updates
    git \
    # Editor for config files
    vim \
    # Build tools for llama.cpp
    build-essential \
    cmake \
    g++ \
    make \
    # For downloading models
    curl \
    wget \
    ca-certificates \
    # ODBC dependencies
    unixodbc \
    unixodbc-dev \
    # Required for some Python packages
    libffi-dev \
    libssl-dev \
    gnupg \
    # Cleanup apt cache
    && rm -rf /var/lib/apt/lists/*

# Install Microsoft ODBC Driver 18 for SQL Server (Debian 12 Bookworm)
RUN curl -fsSL https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor -o /usr/share/keyrings/microsoft-prod.gpg \
    && curl -fsSL https://packages.microsoft.com/config/debian/12/prod.list | tee /etc/apt/sources.list.d/mssql-release.list \
    && apt-get update \
    && ACCEPT_EULA=Y apt-get install -y --no-install-recommends msodbcsql18 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies
# Using --no-cache-dir to reduce image size
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories for cache and data
RUN mkdir -p /app/.cache /app/llama.cpp /data

# Volume mounts for persistent data
# - /app/.cache: Downloaded models and temporary files
# - /data: Database and credentials
VOLUME ["/app/.cache", "/data"]

# Environment variables (can be overridden at runtime)
# Database path (use /data for persistence)
ENV DB_PATH=/data/gguf_app.db

# Model cache directory
ENV MODEL_DOWNLOAD_PATH=/app/.cache

# Expose the default port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Default command
CMD ["python", "app_gguf.py"]
