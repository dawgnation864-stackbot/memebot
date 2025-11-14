FROM python:3.12-slim

ENV PIP_DEFAULT_TIMEOUT=120 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Required tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential pkg-config libssl-dev git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /app/
RUN pip install --upgrade pip && pip install --prefer-binary -r requirements.txt

# Copy bot code
COPY memebot.py /app/

# ENV controlled by Railway
ENV START_MODE=start

CMD ["python", "memebot.py"]
