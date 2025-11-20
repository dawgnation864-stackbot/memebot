FROM python:3.11-slim

WORKDIR /app

# Install system deps for building wheels if needed
RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

COPY memebot.py /app/

CMD ["python", "memebot.py"]
