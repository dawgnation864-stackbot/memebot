FROM python:3.12-slim

# Make pip faster and less buggy
ENV PIP_DEFAULT_TIMEOUT=120 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Small system tools in case wheels are not available
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential pkg-config libssl-dev git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements.txt /app/

RUN python -m pip install --upgrade pip \
    && pip install --prefer-binary -r requirements.txt

# Copy bot code
COPY memebot.py /app/

# Railway will control SIMULATION_MODE via env vars,
# so we DO NOT set it here.
ENV START_MODE=start

CMD ["python", "memebot.py"]
