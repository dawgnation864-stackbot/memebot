FROM python:3.11-slim

# Where the app will live inside the container
WORKDIR /app

# ---- Install system packages needed for solana & friends ----
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    pkg-config \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# ---- Copy Python dependencies list ----
COPY requirements.txt .

# Upgrade pip and basic build tools
RUN pip install --upgrade pip setuptools wheel

# Install all Python dependencies (this includes solana from requirements.txt)
RUN pip install --no-cache-dir -r requirements.txt

# ---- Copy your bot code ----
COPY memebot.py /app/

# ---- Start the bot ----
CMD ["python", "memebot.py"]
