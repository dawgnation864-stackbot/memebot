FROM python:3.10-slim

ENV PIP_DEFAULT_TIMEOUT=120 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    --no-install-recommends \
    build-essential pkg-config libssl-dev git curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/

RUN python -m pip install --upgrade pip && \
    pip install --prefer-binary -r requirements.txt

COPY memebot.py /app/

ENV START_MODE=start

CMD ["python", "memebot.py"]
