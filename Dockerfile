FROM python:3.10-slim

# Make pip faster and less buggy
ENV PIP_DEFAULT_TIMEOUT=120 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Small system tools needed for building wheels (solana / solders)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential pkg-config libssl-dev git curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency list
COPY requirements.txt /app/

# Install Python packages (all requirements, including solana + solders)
RUN python -m pip install --upgrade pip \
    && pip install --prefer-binary -r requirements.txt

# Copy bot code
COPY memebot.py /app/

# Only set START_MODE here; SIMULATION_MODE is controlled by env vars
ENV START_MODE=start

# Run the bot
CMD ["python", "memebot.py"]
