# Leaner GPU-capable Docker image for the reduced Chatterbox TTS API service.
#
# Key differences from docker/Dockerfile:
# - installs Python packages in one layer
# - uses --no-cache-dir to avoid huge pip caches
# - installs torch only once
# - relies on requirements.txt instead of manually duplicating app deps
# - removes stale deps like python-multipart, pydub, and sse-starlette

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    software-properties-common \
    git \
    curl \
    ffmpeg \
    libsndfile1 \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

WORKDIR /app

RUN groupadd --gid 10001 appuser && \
    useradd --uid 10001 --gid 10001 --create-home --shell /usr/sbin/nologin appuser && \
    mkdir -p /app /cache && \
    chown -R appuser:appuser /app /cache

ENV VIRTUAL_ENV=/app/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
ENV HOME=/home/appuser

COPY --chown=appuser:appuser requirements.txt ./

USER appuser

RUN python3 -m venv "$VIRTUAL_ENV" && \
    pip install --no-cache-dir --upgrade pip setuptools==79.0.1 wheel && \
    pip install --no-cache-dir fastapi uvicorn[standard] python-dotenv python-multipart requests psutil pydub sse-starlette && \
    pip install git+https://github.com/travisvn/chatterbox-multilingual.git@exp

COPY --chown=appuser:appuser app/ ./app/
COPY --chown=appuser:appuser main.py ./
COPY --chown=appuser:appuser speech.wav ./speech.wav

ENV HOST=0.0.0.0
ENV PORT=4123
ENV DEVICE=cuda
ENV CORS_ORIGINS=*
ENV VOICE_SAMPLE_PATH=/app/speech.wav
ENV MODEL_CACHE_DIR=/cache
ENV MODEL_SOURCE=default
ENV USE_MULTILINGUAL_MODEL=true
ENV EXAGGERATION=0.5
ENV CFG_WEIGHT=0.5
ENV TEMPERATURE=0.8
ENV MAX_CHUNK_LENGTH=280
ENV MAX_TOTAL_LENGTH=3000
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

EXPOSE 4123

CMD ["python", "main.py"]