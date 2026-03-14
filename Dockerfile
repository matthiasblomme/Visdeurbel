# --------------------------------------------------------------------------
# Visdeurbel fish detector — runtime image
# Base: python:3.11-slim (Debian Bookworm, minimal footprint)
# --------------------------------------------------------------------------
FROM python:3.11-slim

# System dependencies:
#   ffmpeg        — OpenCV's FFmpeg backend for HLS stream decoding
#   libglib2.0-0  — required at runtime by opencv-python-headless
#   libgomp1      — OpenMP support used by OpenCV's parallel processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (separate layer so it's cached on code changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source files
COPY config.py detector.py notifier.py verifier.py main.py ./

# Snapshots directory — bind-mounted at runtime via docker-compose for persistence
RUN mkdir -p snapshots

# -u forces unbuffered stdout so log lines appear immediately in docker logs
CMD ["python", "-u", "main.py"]
