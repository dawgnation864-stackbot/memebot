FROM python:3.12-slim

# Install system tools needed for Solana + Solders
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    libssl-dev \
    libudev-dev \
    clang \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Rust (required for Solders)
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /app

COPY requirements.txt /app/
RUN python -m pip install --upgrade pip && \
    pip install --prefer-binary -r requirements.txt

COPY memebot.py /app/

CMD ["python", "memebot.py"]
