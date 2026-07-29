"""
Request models for API validation
"""

from typing import Optional
from pydantic import BaseModel, Field, validator

from app.config import Config


class TTSRequest(BaseModel):
    """Text-to-speech request model"""

    input: str = Field(..., description="The text to generate audio for", min_length=1)
    voice: Optional[str] = Field(
        Config.DEFAULT_VOICE_NAME,
        description="Voice name to resolve against the configured voice library (DEFAULT_VOICE_NAME/VOICE_LIBRARY). Unknown names fall back to the default voice.",
    )
    response_format: Optional[str] = Field(
        "wav",
        description=(
            "Sample format: 'pcm' (raw 16-bit little-endian PCM, no header) or "
            "'wav' (streaming-safe RIFF/WAV header). Other OpenAI response_format "
            "values (mp3, opus, aac, flac) are rejected rather than silently "
            "returning wav."
        ),
    )
    speed: Optional[float] = Field(1.0, description="Speed of speech (ignored)")
    stream_format: Optional[str] = Field(
        None,
        description=(
            "Streaming format: 'audio' for a genuine chunked byte stream, 'sse' for "
            "Server-Side Events. Absent (the default) returns one buffered response."
        ),
    )

    # Custom TTS parameters
    exaggeration: Optional[float] = Field(
        None, description="Emotion intensity", ge=0.25, le=2.0
    )
    cfg_weight: Optional[float] = Field(
        None, description="Pace control", ge=0.0, le=1.0
    )
    temperature: Optional[float] = Field(
        None, description="Sampling temperature", ge=0.05, le=5.0
    )
    top_p: Optional[float] = Field(
        None, description="Nucleus sampling cutoff", gt=0.0, le=1.0
    )
    min_p: Optional[float] = Field(
        None, description="Minimum token probability floor", ge=0.0, lt=1.0
    )
    repetition_penalty: Optional[float] = Field(
        None,
        description="Repetition penalty — 1.0 disables, >1.0 discourages repeats",
        ge=1.0,
    )

    @validator("input")
    def validate_input(cls, v):
        if not v or not v.strip():
            raise ValueError("Input text cannot be empty")
        return v.strip()

    @validator("stream_format")
    def validate_stream_format(cls, v):
        if v is not None:
            allowed_formats = ["audio", "sse"]
            if v not in allowed_formats:
                raise ValueError(
                    f"stream_format must be one of: {', '.join(allowed_formats)}"
                )
        return v
