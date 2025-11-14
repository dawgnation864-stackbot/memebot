FROM python:3.12-slim

# Prevent issues with pip
ENV PIP_DEFAULT_TIMEOUT=120 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies needed for solana/solders to build
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential pkg-config libssl-dev git curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first
COPY requirements.txt /app/

# Install Python packages
RUN python -m pip install --upgrade pip \
    && pip install --prefer-binary -r requirements.txt

# Copy bot script
COPY memebot.py /app/

# Let Railway control SIMULATION_MODE
ENV START_MODE=start

CMD ["python", "memebot.py"]
