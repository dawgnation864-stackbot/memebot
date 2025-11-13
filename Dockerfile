FROM python:3.12-slim

# Make pip faster and less buggy
ENV PIP_DEFAULT_TIMEOUT=120 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Small system tools in case needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential pkg-config libssl-dev git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/

# Install Python packages
RUN python -m pip install --upgrade pip \
    && pip install --prefer-binary -r requirements.txt

COPY memebot.py /app/

# REMOVE simulation default so Railway can control it
# ENV SIMULATION_MODE=True   <-- delete this line completely

ENV START_MODE=start

CMD ["python", "memebot.py"]
