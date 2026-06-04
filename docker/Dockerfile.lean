# Leaner GPU-capable Docker image for the Chatterbox TTS API service.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    software-properties-common git curl ffmpeg libsndfile1 \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.13 python3.13-venv \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.13 1 && \
    curl -LsSf https://astral.sh/uv/install.sh | sh -s -- --no-modify-path && \
    mv /root/.local/bin/uv /usr/local/bin/uv

WORKDIR /app

RUN groupadd --gid 10001 appuser && \
    useradd --uid 10001 --gid 10001 --create-home --shell /usr/sbin/nologin appuser && \
    mkdir -p /app /cache && \
    chown -R appuser:appuser /app /cache

ENV VIRTUAL_ENV=/app/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
ENV HOME=/home/appuser

COPY --chown=appuser:appuser pyproject.toml uv.lock ./
COPY --chown=appuser:appuser app/ ./app/
COPY --chown=appuser:appuser main.py ./
COPY --chown=appuser:appuser voice-sample.mp3 ./voice-sample.mp3

USER appuser

RUN uv sync --frozen --no-dev && \
    python3 -c "import nltk; nltk.download('punkt_tab', quiet=True)"

ENV HOST=0.0.0.0
ENV PORT=4123
ENV DEVICE=cuda
ENV CORS_ORIGINS=*
ENV VOICE_SAMPLE_PATH=/app/voice-sample.mp3
ENV MODEL_CACHE_DIR=/cache
ENV MODEL_SOURCE=default
ENV MODEL_TYPE=multilingual
ENV EXAGGERATION=0.5
ENV CFG_WEIGHT=0.5
ENV TEMPERATURE=0.8
ENV MAX_TOTAL_LENGTH=3000
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

EXPOSE 4123

CMD ["python", "main.py"]
