"""
Text-to-speech endpoint.
"""

import asyncio
import base64
import io
import logging
from dataclasses import dataclass
from typing import AsyncGenerator, Optional
from uuid import uuid4

import torch
import torchaudio as ta
from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from app.config import Config
from app.core import concatenate_audio_chunks, split_text_into_chunks
from app.core.metrics import (
    observe_lease_acquire_failure,
    observe_request_finished,
    observe_request_started,
)
from app.core.observability import get_logger, log_event
from app.core.text_processing import get_streaming_settings, split_text_for_streaming
from app.core.tts_model import (
    ModelLease,
    ModelNotReadyError,
    ModelPoolExhaustedError,
    acquire_model_lease,
    get_default_language,
    is_multilingual,
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
logger = get_logger(__name__)


class RequestTimeoutExceeded(RuntimeError):
    """Raised when a request exceeds the configured total timeout."""

    def __init__(self, stage: str):
        super().__init__(f"Request timed out during {stage}")
        self.stage = stage


class ClientDisconnected(RuntimeError):
    """Raised when the SSE client disconnects mid-request."""

    def __init__(self, stage: str):
        super().__init__(f"Client disconnected during {stage}")
        self.stage = stage


@dataclass
class RequestRuntimeContext:
    request_id: str
    mode: str
    started_at: float
    deadline: float
    client_request: Optional[Request] = None
    lease_acquired_at: Optional[float] = None

    def remaining_seconds(self) -> float:
        return max(self.deadline - asyncio.get_running_loop().time(), 0.0)

    def elapsed_seconds(self) -> float:
        return max(asyncio.get_running_loop().time() - self.started_at, 0.0)

    def lease_wait_seconds(self) -> Optional[float]:
        if self.lease_acquired_at is None:
            return None
        return max(self.lease_acquired_at - self.started_at, 0.0)

    def generation_elapsed_seconds(self) -> Optional[float]:
        if self.lease_acquired_at is None:
            return None
        return max(asyncio.get_running_loop().time() - self.lease_acquired_at, 0.0)


def _audio_num_frames(audio_tensor: torch.Tensor) -> int:
    if audio_tensor.dim() == 1:
        return int(audio_tensor.shape[0])
    return int(audio_tensor.shape[-1])


def _audio_duration_seconds(audio_tensor: torch.Tensor, sample_rate: int) -> float:
    return _audio_num_frames(audio_tensor) / float(sample_rate)


def _new_request_context(
    mode: str, client_request: Optional[Request] = None
) -> RequestRuntimeContext:
    loop = asyncio.get_running_loop()
    started_at = loop.time()
    return RequestRuntimeContext(
        request_id=uuid4().hex,
        mode=mode,
        started_at=started_at,
        deadline=started_at + Config.REQUEST_TIMEOUT_SECONDS,
        client_request=client_request,
    )


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


def _request_timeout_http_error() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_504_GATEWAY_TIMEOUT,
        detail={
            "error": {
                "message": "Request timed out while generating speech",
                "type": "timeout_error",
            }
        },
    )


def _log_request_event(
    level: int, message: str, context: RequestRuntimeContext, **fields
):
    log_event(
        logger,
        level,
        message,
        request_id=context.request_id,
        request_mode=context.mode,
        elapsed_seconds=round(context.elapsed_seconds(), 6),
        route="/v1/audio/speech",
        **fields,
    )


def _raise_if_request_expired(context: RequestRuntimeContext, stage: str):
    if context.remaining_seconds() <= 0:
        raise RequestTimeoutExceeded(stage)


async def _raise_if_client_disconnected(
    context: RequestRuntimeContext,
    stage: str,
):
    if (
        context.client_request is not None
        and await context.client_request.is_disconnected()
    ):
        raise ClientDisconnected(stage)


async def _guard_request_state(context: RequestRuntimeContext, stage: str):
    _raise_if_request_expired(context, stage)
    await _raise_if_client_disconnected(context, stage)


async def _acquire_request_lease(context: RequestRuntimeContext) -> ModelLease:
    _raise_if_request_expired(context, "lease_wait")

    try:
        lease_wait_timeout = context.remaining_seconds()
        if Config.MAX_QUEUE_WAIT_SECONDS <= 0:
            lease_wait_timeout = 0
        else:
            lease_wait_timeout = min(Config.MAX_QUEUE_WAIT_SECONDS, lease_wait_timeout)

        try:
            lease = await acquire_model_lease(lease_wait_timeout)
        except ModelPoolExhaustedError as exc:
            if (
                Config.MAX_QUEUE_WAIT_SECONDS > 0
                and context.remaining_seconds() <= 0
                and lease_wait_timeout <= Config.REQUEST_TIMEOUT_SECONDS
            ):
                raise RequestTimeoutExceeded("lease_wait") from exc
            observe_lease_acquire_failure("no_capacity")
            _log_request_event(
                logging.WARNING,
                "request_rejected_no_capacity",
                context,
                outcome="overload",
                lease_wait_seconds=round(context.elapsed_seconds(), 6),
            )
            observe_request_finished(
                "/v1/audio/speech",
                context.mode,
                "overload",
                elapsed_seconds=context.elapsed_seconds(),
            )
            raise _model_capacity_http_error() from exc
    except ModelNotReadyError as exc:
        observe_lease_acquire_failure("not_ready")
        _log_request_event(
            logging.WARNING,
            "request_rejected_model_not_ready",
            context,
            outcome="not_ready",
        )
        observe_request_finished(
            "/v1/audio/speech",
            context.mode,
            "not_ready",
            elapsed_seconds=context.elapsed_seconds(),
        )
        raise _model_not_ready_http_error() from exc

    context.lease_acquired_at = asyncio.get_running_loop().time()

    _log_request_event(
        logging.INFO,
        "request_lease_acquired",
        context,
        model_instance_id=lease.instance_id,
        stage="lease_acquired",
        lease_wait_seconds=round(context.lease_wait_seconds() or 0.0, 6),
    )
    return lease


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
    context: RequestRuntimeContext,
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
            _raise_if_request_expired(context, "chunk_generation")
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

        _raise_if_request_expired(context, "response_encoding")
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
    context = _new_request_context(mode="audio")
    observe_request_started("/v1/audio/speech", context.mode, len(text))
    _log_request_event(
        logging.INFO,
        "request_started",
        context,
        outcome="started",
        input_chars=len(text),
    )
    lease = await _acquire_request_lease(context)
    try:
        buffer, _ = await _generate_full_audio(
            context=context,
            lease=lease,
            text=text,
            voice_sample_path=voice_sample_path,
            language_id=_validate_language_for_generation(language_id),
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            temperature=temperature,
        )
        return buffer
    finally:
        await release_model_lease(lease)


async def generate_speech_sse(
    context: RequestRuntimeContext,
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
    chunk_count = len(text_chunks)

    try:
        await _guard_request_state(context, "sse_start")
        info_event = SSEAudioInfo(
            sample_rate=lease.model.sr,
            channels=1,
            bits_per_sample=16,
        )
        yield f"data: {info_event.model_dump_json()}\n\n"

        total_frames = 0
        for chunk in text_chunks:
            await _guard_request_state(context, "chunk_generation")
            audio_tensor = await _generate_chunk_audio(
                lease=lease,
                chunk=chunk,
                voice_sample_path=voice_sample_path,
                language_id=language_id,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                temperature=temperature,
            )
            await _guard_request_state(context, "chunk_emit")
            total_frames += _audio_num_frames(audio_tensor)
            pcm_tensor = (torch.clamp(audio_tensor, -1.0, 1.0) * 32767).to(torch.int16)
            payload = base64.b64encode(pcm_tensor.numpy().tobytes()).decode("ascii")
            yield f"data: {SSEAudioDelta(audio=payload).model_dump_json()}\n\n"

        await _guard_request_state(context, "done_event")
        usage_event = SSEAudioDone(
            usage=SSEUsageInfo(
                input_chars=len(text),
                audio_seconds=total_frames / float(lease.model.sr),
            )
        )
        _log_request_event(
            logging.INFO,
            "request_completed",
            context,
            model_instance_id=lease.instance_id,
            outcome="success",
            input_chars=len(text),
            audio_seconds=usage_event.usage.audio_seconds,
            chunk_count=chunk_count,
        )
        observe_request_finished(
            "/v1/audio/speech",
            context.mode,
            "success",
            elapsed_seconds=context.elapsed_seconds(),
            lease_wait_seconds=context.lease_wait_seconds(),
            generation_duration_seconds=context.generation_elapsed_seconds(),
            audio_seconds=usage_event.usage.audio_seconds,
            chunk_count=chunk_count,
        )
        yield f"data: {usage_event.model_dump_json()}\n\n"
    except RequestTimeoutExceeded as exc:
        _log_request_event(
            logging.WARNING,
            "request_timeout",
            context,
            model_instance_id=lease.instance_id,
            timeout_stage=exc.stage,
            outcome="timeout",
            chunk_count=chunk_count,
        )
        observe_request_finished(
            "/v1/audio/speech",
            context.mode,
            "timeout",
            elapsed_seconds=context.elapsed_seconds(),
            lease_wait_seconds=context.lease_wait_seconds(),
            generation_duration_seconds=context.generation_elapsed_seconds(),
            chunk_count=chunk_count,
        )
        return
    except ClientDisconnected as exc:
        _log_request_event(
            logging.INFO,
            "request_disconnected",
            context,
            model_instance_id=lease.instance_id,
            disconnect_stage=exc.stage,
            outcome="disconnect",
            chunk_count=chunk_count,
        )
        observe_request_finished(
            "/v1/audio/speech",
            context.mode,
            "disconnect",
            elapsed_seconds=context.elapsed_seconds(),
            lease_wait_seconds=context.lease_wait_seconds(),
            generation_duration_seconds=context.generation_elapsed_seconds(),
            chunk_count=chunk_count,
        )
        return
    except Exception as exc:
        _log_request_event(
            logging.ERROR,
            "request_failed",
            context,
            model_instance_id=lease.instance_id,
            outcome="error",
            error_type=type(exc).__name__,
            chunk_count=chunk_count,
        )
        observe_request_finished(
            "/v1/audio/speech",
            context.mode,
            "error",
            elapsed_seconds=context.elapsed_seconds(),
            lease_wait_seconds=context.lease_wait_seconds(),
            generation_duration_seconds=context.generation_elapsed_seconds(),
            chunk_count=chunk_count,
        )
        raise
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
        504: {"model": ErrorResponse},
    },
    summary="Generate speech from text",
    description="Generate speech audio from input text. Use stream_format='sse' for streaming.",
)
async def text_to_speech(request: TTSRequest, client_request: Request):
    voice_sample_path, language_id = resolve_voice_path_and_language(request.voice)
    resolved_language = _validate_language_for_generation(language_id)
    _validate_text_length(request.input)

    if request.stream_format == "sse":
        context = _new_request_context(mode="sse", client_request=client_request)
        observe_request_started("/v1/audio/speech", context.mode, len(request.input))
        _log_request_event(
            logging.INFO,
            "request_started",
            context,
            outcome="started",
            input_chars=len(request.input),
        )
        lease = await _acquire_request_lease(context)
        return StreamingResponse(
            generate_speech_sse(
                context=context,
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
                "X-Request-ID": context.request_id,
            },
        )

    context = _new_request_context(mode="audio", client_request=client_request)
    observe_request_started("/v1/audio/speech", context.mode, len(request.input))
    _log_request_event(
        logging.INFO,
        "request_started",
        context,
        outcome="started",
        input_chars=len(request.input),
    )
    lease = None
    try:
        lease = await _acquire_request_lease(context)
        buffer, audio_seconds = await _generate_full_audio(
            context=context,
            lease=lease,
            text=request.input,
            voice_sample_path=voice_sample_path,
            language_id=resolved_language,
            exaggeration=request.exaggeration,
            cfg_weight=request.cfg_weight,
            temperature=request.temperature,
        )
        response = StreamingResponse(
            io.BytesIO(buffer.getvalue()),
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=speech.wav",
                "X-Usage-Input-Chars": str(len(request.input)),
                "X-Usage-Audio-Seconds": f"{audio_seconds:.6f}",
                "X-Model-Instance-ID": str(lease.instance_id),
                "X-Request-ID": context.request_id,
            },
        )
        _log_request_event(
            logging.INFO,
            "request_completed",
            context,
            model_instance_id=lease.instance_id,
            outcome="success",
            input_chars=len(request.input),
            audio_seconds=round(audio_seconds, 6),
            chunk_count=len(
                split_text_into_chunks(request.input, Config.MAX_CHUNK_LENGTH)
            ),
        )
        observe_request_finished(
            "/v1/audio/speech",
            context.mode,
            "success",
            elapsed_seconds=context.elapsed_seconds(),
            lease_wait_seconds=context.lease_wait_seconds(),
            generation_duration_seconds=context.generation_elapsed_seconds(),
            audio_seconds=round(audio_seconds, 6),
            chunk_count=len(
                split_text_into_chunks(request.input, Config.MAX_CHUNK_LENGTH)
            ),
        )
        return response
    except RequestTimeoutExceeded as exc:
        _log_request_event(
            logging.WARNING,
            "request_timeout",
            context,
            model_instance_id=lease.instance_id if lease is not None else None,
            timeout_stage=exc.stage,
            outcome="timeout",
            input_chars=len(request.input),
        )
        observe_request_finished(
            "/v1/audio/speech",
            context.mode,
            "timeout",
            elapsed_seconds=context.elapsed_seconds(),
            lease_wait_seconds=context.lease_wait_seconds(),
            generation_duration_seconds=context.generation_elapsed_seconds(),
        )
        raise _request_timeout_http_error() from exc
    except HTTPException:
        raise
    except Exception as exc:
        _log_request_event(
            logging.ERROR,
            "request_failed",
            context,
            model_instance_id=lease.instance_id if lease is not None else None,
            outcome="error",
            error_type=type(exc).__name__,
            input_chars=len(request.input),
        )
        observe_request_finished(
            "/v1/audio/speech",
            context.mode,
            "error",
            elapsed_seconds=context.elapsed_seconds(),
            lease_wait_seconds=context.lease_wait_seconds(),
            generation_duration_seconds=context.generation_elapsed_seconds(),
        )
        raise
    finally:
        if lease is not None:
            await release_model_lease(lease)


__all__ = [
    "ClientDisconnected",
    "RequestRuntimeContext",
    "RequestTimeoutExceeded",
    "base_router",
    "generate_speech_internal",
    "generate_speech_sse",
    "resolve_voice_path_and_language",
]
