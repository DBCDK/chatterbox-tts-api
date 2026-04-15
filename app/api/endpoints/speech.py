"""
Text-to-speech endpoint.
"""

import asyncio
import base64
import io
from typing import AsyncGenerator, Optional

import torch
import torchaudio as ta
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse

from app.config import Config
from app.core import concatenate_audio_chunks, split_text_into_chunks
from app.core.text_processing import get_streaming_settings, split_text_for_streaming
from app.core.tts_model import (
    ModelLease,
    ModelNotReadyError,
    ModelPoolExhaustedError,
    acquire_model_lease,
    get_default_language,
    is_multilingual,
    leased_model,
    release_model_lease,
    supports_language,
)
from app.models import (
    ErrorResponse,
    SSEAudioDelta,
    SSEAudioDone,
    SSEAudioInfo,
    SSEUsageInfo,
    TTSRequest,
)

base_router = APIRouter()


def _audio_num_frames(audio_tensor: torch.Tensor) -> int:
    if audio_tensor.dim() == 1:
        return int(audio_tensor.shape[0])
    return int(audio_tensor.shape[-1])


def _audio_duration_seconds(audio_tensor: torch.Tensor, sample_rate: int) -> float:
    return _audio_num_frames(audio_tensor) / float(sample_rate)


def _model_not_ready_http_error() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail={"error": {"message": "Model pool not ready", "type": "model_error"}},
    )


def _model_capacity_http_error() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail={
            "error": {
                "message": "No model instances available for this request",
                "type": "capacity_error",
            }
        },
    )


async def _acquire_request_lease() -> ModelLease:
    try:
        return await acquire_model_lease(Config.MAX_QUEUE_WAIT_SECONDS)
    except ModelNotReadyError as exc:
        raise _model_not_ready_http_error() from exc
    except ModelPoolExhaustedError as exc:
        raise _model_capacity_http_error() from exc


def _validate_text_length(text: str):
    if len(text) > Config.MAX_TOTAL_LENGTH:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": {
                    "message": f"Input text too long. Maximum {Config.MAX_TOTAL_LENGTH} characters allowed.",
                    "type": "invalid_request_error",
                }
            },
        )


def resolve_voice_path_and_language(
    voice_name: Optional[str],
) -> tuple[str, Optional[str]]:
    """Resolve request voice selection to the configured sample path."""
    default_language = get_default_language()
    return Config.VOICE_SAMPLE_PATH, default_language if is_multilingual() else None


def _validate_language_for_generation(language_id: Optional[str]) -> Optional[str]:
    if not is_multilingual():
        return None

    resolved_language = (language_id or get_default_language()).lower()
    if not supports_language(resolved_language):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": {
                    "message": f"Unsupported language for configured model: {resolved_language}",
                    "type": "invalid_request_error",
                }
            },
        )
    return resolved_language


def _generation_kwargs(
    text: str,
    voice_sample_path: str,
    language_id: Optional[str],
    exaggeration: Optional[float],
    cfg_weight: Optional[float],
    temperature: Optional[float],
) -> dict:
    kwargs = {
        "text": text,
        "audio_prompt_path": voice_sample_path,
        "exaggeration": exaggeration
        if exaggeration is not None
        else Config.EXAGGERATION,
        "cfg_weight": cfg_weight if cfg_weight is not None else Config.CFG_WEIGHT,
        "temperature": temperature if temperature is not None else Config.TEMPERATURE,
    }
    if is_multilingual() and language_id:
        kwargs["language_id"] = language_id
    return kwargs


async def _generate_chunk_audio(
    lease: ModelLease,
    chunk: str,
    voice_sample_path: str,
    language_id: Optional[str],
    exaggeration: Optional[float],
    cfg_weight: Optional[float],
    temperature: Optional[float],
) -> torch.Tensor:
    loop = asyncio.get_running_loop()

    try:
        with torch.no_grad():
            audio_tensor = await loop.run_in_executor(
                None,
                lambda: lease.model.generate(
                    **_generation_kwargs(
                        text=chunk,
                        voice_sample_path=voice_sample_path,
                        language_id=language_id,
                        exaggeration=exaggeration,
                        cfg_weight=cfg_weight,
                        temperature=temperature,
                    )
                ),
            )
    except Exception as exc:
        lease.mark_broken(str(exc))
        raise

    return (
        audio_tensor.detach().cpu() if hasattr(audio_tensor, "detach") else audio_tensor
    )


async def _generate_full_audio(
    lease: ModelLease,
    text: str,
    voice_sample_path: str,
    language_id: Optional[str],
    exaggeration: Optional[float],
    cfg_weight: Optional[float],
    temperature: Optional[float],
) -> tuple[io.BytesIO, float]:
    _validate_text_length(text)

    chunks = split_text_into_chunks(text, Config.MAX_CHUNK_LENGTH)
    audio_chunks: list[torch.Tensor] = []
    try:
        for chunk in chunks:
            audio_chunks.append(
                await _generate_chunk_audio(
                    lease=lease,
                    chunk=chunk,
                    voice_sample_path=voice_sample_path,
                    language_id=language_id,
                    exaggeration=exaggeration,
                    cfg_weight=cfg_weight,
                    temperature=temperature,
                )
            )

        final_audio = (
            concatenate_audio_chunks(audio_chunks, lease.model.sr)
            if len(audio_chunks) > 1
            else audio_chunks[0]
        )
        buffer = io.BytesIO()
        ta.save(buffer, final_audio, lease.model.sr, format="wav")
        buffer.seek(0)
        return buffer, _audio_duration_seconds(final_audio, lease.model.sr)
    finally:
        audio_chunks.clear()


async def generate_speech_internal(
    text: str,
    voice_sample_path: str,
    language_id: Optional[str] = None,
    exaggeration: Optional[float] = None,
    cfg_weight: Optional[float] = None,
    temperature: Optional[float] = None,
) -> io.BytesIO:
    async with leased_model(Config.MAX_QUEUE_WAIT_SECONDS) as lease:
        buffer, _ = await _generate_full_audio(
            lease=lease,
            text=text,
            voice_sample_path=voice_sample_path,
            language_id=_validate_language_for_generation(language_id),
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            temperature=temperature,
        )
        return buffer


async def generate_speech_sse(
    lease: ModelLease,
    text: str,
    voice_sample_path: str,
    language_id: Optional[str] = None,
    exaggeration: Optional[float] = None,
    cfg_weight: Optional[float] = None,
    temperature: Optional[float] = None,
    streaming_chunk_size: Optional[int] = None,
    streaming_strategy: Optional[str] = None,
    streaming_quality: Optional[str] = None,
) -> AsyncGenerator[str, None]:
    language_id = _validate_language_for_generation(language_id)
    _validate_text_length(text)

    settings = get_streaming_settings(
        streaming_chunk_size,
        streaming_strategy,
        streaming_quality,
    )
    text_chunks = split_text_for_streaming(
        text,
        chunk_size=settings["chunk_size"],
        strategy=settings["strategy"],
        quality=settings["quality"],
    )

    try:
        info_event = SSEAudioInfo(
            sample_rate=lease.model.sr,
            channels=1,
            bits_per_sample=16,
        )
        yield f"data: {info_event.model_dump_json()}\n\n"

        total_frames = 0
        for chunk in text_chunks:
            audio_tensor = await _generate_chunk_audio(
                lease=lease,
                chunk=chunk,
                voice_sample_path=voice_sample_path,
                language_id=language_id,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                temperature=temperature,
            )
            total_frames += _audio_num_frames(audio_tensor)
            pcm_tensor = (torch.clamp(audio_tensor, -1.0, 1.0) * 32767).to(torch.int16)
            payload = base64.b64encode(pcm_tensor.numpy().tobytes()).decode("ascii")
            yield f"data: {SSEAudioDelta(audio=payload).model_dump_json()}\n\n"

        usage_event = SSEAudioDone(
            usage=SSEUsageInfo(
                input_chars=len(text),
                audio_seconds=total_frames / float(lease.model.sr),
            )
        )
        yield f"data: {usage_event.model_dump_json()}\n\n"
    finally:
        await release_model_lease(lease)


@base_router.post(
    "/v1/audio/speech",
    response_class=StreamingResponse,
    responses={
        200: {"content": {"audio/wav": {}, "text/event-stream": {}}},
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
    summary="Generate speech from text",
    description="Generate speech audio from input text. Use stream_format='sse' for streaming.",
)
async def text_to_speech(request: TTSRequest):
    voice_sample_path, language_id = resolve_voice_path_and_language(request.voice)
    resolved_language = _validate_language_for_generation(language_id)
    _validate_text_length(request.input)

    if request.stream_format == "sse":
        lease = await _acquire_request_lease()
        return StreamingResponse(
            generate_speech_sse(
                lease=lease,
                text=request.input,
                voice_sample_path=voice_sample_path,
                language_id=resolved_language,
                exaggeration=request.exaggeration,
                cfg_weight=request.cfg_weight,
                temperature=request.temperature,
                streaming_chunk_size=request.streaming_chunk_size,
                streaming_strategy=request.streaming_strategy,
                streaming_quality=request.streaming_quality,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
                "X-Model-Instance-ID": str(lease.instance_id),
            },
        )

    lease = await _acquire_request_lease()
    try:
        buffer, audio_seconds = await _generate_full_audio(
            lease=lease,
            text=request.input,
            voice_sample_path=voice_sample_path,
            language_id=resolved_language,
            exaggeration=request.exaggeration,
            cfg_weight=request.cfg_weight,
            temperature=request.temperature,
        )
        return StreamingResponse(
            io.BytesIO(buffer.getvalue()),
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=speech.wav",
                "X-Usage-Input-Chars": str(len(request.input)),
                "X-Usage-Audio-Seconds": f"{audio_seconds:.6f}",
                "X-Model-Instance-ID": str(lease.instance_id),
            },
        )
    finally:
        await release_model_lease(lease)


__all__ = [
    "base_router",
    "generate_speech_internal",
    "generate_speech_sse",
    "resolve_voice_path_and_language",
]
