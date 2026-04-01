"""
Request models for API validation
"""

from typing import Optional
from pydantic import BaseModel, Field, validator


class TTSRequest(BaseModel):
    """Text-to-speech request model"""

    input: str = Field(
        ..., description="The text to generate audio for", min_length=1, max_length=3000
    )
    voice: Optional[str] = Field(
        "alloy",
        description="Voice name, alias, or OpenAI-style voice name to resolve against the configured/default voice sample",
    )
    response_format: Optional[str] = Field(
        "wav", description="Audio format (always returns WAV)"
    )
    speed: Optional[float] = Field(1.0, description="Speed of speech (ignored)")
    stream_format: Optional[str] = Field(
        "audio",
        description="Streaming format: 'audio' for raw audio stream, 'sse' for Server-Side Events",
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

    # Streaming-specific parameters
    streaming_chunk_size: Optional[int] = Field(
        None, description="Characters per streaming chunk", ge=50, le=500
    )
    streaming_strategy: Optional[str] = Field(
        None, description="Chunking strategy for streaming"
    )
    streaming_buffer_size: Optional[int] = Field(
        None, description="Number of chunks to buffer", ge=1, le=10
    )
    streaming_quality: Optional[str] = Field(
        None, description="Speed vs quality trade-off"
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

    @validator("streaming_strategy")
    def validate_streaming_strategy(cls, v):
        if v is not None:
            allowed_strategies = ["sentence", "paragraph", "fixed", "word"]
            if v not in allowed_strategies:
                raise ValueError(
                    f"streaming_strategy must be one of: {', '.join(allowed_strategies)}"
                )
        return v

    @validator("streaming_quality")
    def validate_streaming_quality(cls, v):
        if v is not None:
            allowed_qualities = ["fast", "balanced", "high"]
            if v not in allowed_qualities:
                raise ValueError(
                    f"streaming_quality must be one of: {', '.join(allowed_qualities)}"
                )
        return v
