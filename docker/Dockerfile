# Leaner GPU-capable Docker image for the Chatterbox TTS API service.
FROM docker-dbc.artifacts.dbccloud.dk/dbc-python3


RUN apt-get update && apt-get install -y ffmpeg git

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
COPY --chown=appuser:appuser voices/ ./voices/

USER appuser

RUN uv sync --frozen --no-dev && \
    python3 -c "import nltk; nltk.download('punkt_tab', quiet=True)"

ENV HOST=0.0.0.0
ENV PORT=4123
ENV DEVICE=cuda
ENV CORS_ORIGINS=*
ENV VOICE_SAMPLE_PATH=/app/voices/mic-voice.wav
ENV DEFAULT_VOICE_NAME=mic
ENV VOICE_LIBRARY="{\"nic\": \"/app/voices/nic-voice.wav\"}"
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
