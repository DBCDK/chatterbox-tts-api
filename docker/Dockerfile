# GPU-capable Docker image for the reduced Chatterbox TTS API service.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y \
    software-properties-common \
    git \
    curl \
    build-essential \
    ffmpeg \
    libsndfile1 \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN groupadd --gid 10001 appuser && \
    useradd --uid 10001 --gid 10001 --create-home --shell /usr/sbin/nologin appuser && \
    mkdir -p /app /cache && \
    chown -R appuser:appuser /app /cache

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

ENV VIRTUAL_ENV=/app/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
ENV HOME=/home/appuser

USER appuser

RUN python3 -m venv "$VIRTUAL_ENV"

# Install PyTorch with CUDA support using uv
# RUN uv pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu128
RUN pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128

# Install base dependencies first
RUN pip install setuptools fastapi uvicorn[standard] python-dotenv python-multipart requests psutil pydub sse-starlette

# Install chatterbox-tts — with the breaking fix (pkuseg package exclusion) 
RUN pip install git+https://github.com/travisvn/chatterbox-multilingual.git@exp

# Added after the install of chatterboxx-tts to hopefully be able to use 
RUN pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128


COPY --chown=appuser:appuser app/ ./app/
COPY --chown=appuser:appuser main.py ./
COPY --chown=appuser:appuser voice-sample.mp3 ./voice-sample.mp3

ENV HOST=0.0.0.0
ENV PORT=4123
ENV DEVICE=cuda
ENV CORS_ORIGINS=*
ENV VOICE_SAMPLE_PATH=/app/voice-sample.mp3
ENV MODEL_CACHE_DIR=/cache
ENV MODEL_SOURCE=default
ENV USE_MULTILINGUAL_MODEL=true
ENV EXAGGERATION=0.5
ENV CFG_WEIGHT=0.5
ENV TEMPERATURE=0.8
ENV MAX_CHUNK_LENGTH=280
ENV MAX_TOTAL_LENGTH=3000

EXPOSE ${PORT}

CMD ["python", "main.py"]
