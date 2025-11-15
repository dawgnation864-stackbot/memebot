FROM python:3.10-slim

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential pkg-config libssl-dev git curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency list FIRST
COPY requirements.txt /app/

# Install Python packages
RUN pip install --upgrade pip && \
    pip install --prefer-binary -r requirements.txt

# Copy bot code LAST
COPY memebot.py /app/

ENV START_MODE=start

CMD ["python", "memebot.py"]
