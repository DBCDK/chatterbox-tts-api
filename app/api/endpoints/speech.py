"""
Text-to-speech endpoint.
"""

import asyncio
import io
import logging
from typing import Optional
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from app.api.endpoints._request_context import (
    ClientDisconnected,
    RequestRuntimeContext,
    RequestTimeoutExceeded,
    _audio_num_frames,
    _generation_kwargs,
    _log_request_event,
    _raise_if_request_expired,
)
from app.api.endpoints.speech_stream import _build_framer, generate_speech_chunks, generate_speech_sse
from app.config import Config
from app.core.audio import build_wav_header, pcm16_bytes
from app.core.metrics import (
    observe_lease_acquire_failure,
    observe_request_failure,
    observe_request_finished,
    observe_request_started,
    observe_requests_waiting_for_lease,
)
from app.core.tts_model import (
    ModelLease,
    ModelNotReadyError,
    ModelPoolExhaustedError,
    acquire_model_lease,
    apply_voice_to_lease,
    get_default_language,
    is_fatal_generation_error,
    is_multilingual,
    release_model_lease,
    supports_language,
)
from app.models import ErrorResponse, TTSRequest

base_router = APIRouter()


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


async def _acquire_request_lease(context: RequestRuntimeContext) -> ModelLease:
    _raise_if_request_expired(context, "lease_wait")

    try:
        lease_wait_timeout = context.remaining_seconds()
        if Config.MAX_QUEUE_WAIT_SECONDS <= 0:
            lease_wait_timeout = 0
        else:
            lease_wait_timeout = min(Config.MAX_QUEUE_WAIT_SECONDS, lease_wait_timeout)

        try:
            observe_requests_waiting_for_lease(1)
            lease = await acquire_model_lease(lease_wait_timeout)
        except ModelPoolExhaustedError as exc:
            if (
                Config.MAX_QUEUE_WAIT_SECONDS > 0
                and context.remaining_seconds() <= 0
                and lease_wait_timeout <= Config.REQUEST_TIMEOUT_SECONDS
            ):
                _log_request_event(
                    logging.WARNING,
                    "request_timeout",
                    context,
                    outcome="timeout",
                    timeout_stage="lease_wait",
                )
                raise RequestTimeoutExceeded("lease_wait") from exc
            observe_lease_acquire_failure("no_capacity")
            observe_request_failure("no_capacity", "lease_wait", context.mode)
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
        finally:
            observe_requests_waiting_for_lease(-1)
    except ModelNotReadyError as exc:
        observe_lease_acquire_failure("not_ready")
        observe_request_failure("not_ready", "lease_acquire", context.mode)
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


def _validate_text_length(text: str, mode: Optional[str] = None):
    if len(text) < Config.MIN_TEXT_LENGTH:
        if mode is not None:
            observe_request_failure("input_too_short", "validation", mode)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": {
                    "message": f"Input text too short. Minimum {Config.MIN_TEXT_LENGTH} characters required.",
                    "type": "invalid_request_error",
                }
            },
        )
    if len(text) > Config.MAX_TOTAL_LENGTH:
        if mode is not None:
            observe_request_failure("input_too_long", "validation", mode)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": {
                    "message": f"Input text too long. Maximum {Config.MAX_TOTAL_LENGTH} characters allowed.",
                    "type": "invalid_request_error",
                }
            },
        )


def _validate_response_format(response_format: Optional[str], mode: Optional[str] = None) -> str:
    resolved_format = (response_format or "wav").lower()
    if resolved_format not in ("pcm", "wav"):
        if mode is not None:
            observe_request_failure("unsupported_response_format", "validation", mode)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": {
                    "message": (
                        f"Unsupported response_format: '{response_format}'. "
                        "This service supports 'pcm' and 'wav' only."
                    ),
                    "type": "invalid_request_error",
                }
            },
        )
    return resolved_format


def resolve_voice_and_language(
    voice_name: Optional[str],
) -> tuple[str, Optional[str]]:
    """Resolve a request's voice selection to a configured voice name and language.

    Unknown voice names fall back to the configured default voice rather than
    erroring, so existing OpenAI-style clients passing arbitrary voice names
    keep working.
    """
    library = Config.get_voice_library()
    resolved_name = (voice_name or "").strip().lower()
    if resolved_name not in library:
        resolved_name = Config.DEFAULT_VOICE_NAME
    default_language = get_default_language()
    return resolved_name, default_language if is_multilingual() else None


def _validate_language_for_generation(
    language_id: Optional[str], mode: Optional[str] = None
) -> Optional[str]:
    if not is_multilingual():
        return None

    resolved_language = (language_id or get_default_language()).lower()
    if not supports_language(resolved_language):
        if mode is not None:
            observe_request_failure("unsupported_language", "validation", mode)
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


async def _generate_full_audio(
    context: RequestRuntimeContext,
    lease: ModelLease,
    text: str,
    language_id: Optional[str],
    exaggeration: Optional[float],
    cfg_weight: Optional[float],
    temperature: Optional[float],
    top_p: Optional[float],
    min_p: Optional[float],
    repetition_penalty: Optional[float],
    response_format: str = "wav",
    voice_name: Optional[str] = None,
) -> tuple[io.BytesIO, float]:
    """Exit A: buffer the same streaming source Exit B/C use, instead of a separate
    generate() call. Gated on tests/audio_quality/test_buffered_stream_equivalence.py --
    see that module's docstring before changing this to call generate() again."""
    _validate_text_length(text)
    _raise_if_request_expired(context, "generation")

    if voice_name is not None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, apply_voice_to_lease, lease, voice_name)

    pcm_chunks: list[bytes] = []
    total_frames = 0
    try:
        async for audio_tensor in lease.model.generate_stream_async(
            **_generation_kwargs(
                text=text,
                language_id=language_id,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                temperature=temperature,
                top_p=top_p,
                min_p=min_p,
                repetition_penalty=repetition_penalty,
            )
        ):
            audio_tensor = audio_tensor.detach().cpu()
            total_frames += _audio_num_frames(audio_tensor)
            pcm_chunks.append(pcm16_bytes(audio_tensor))
    except Exception as exc:
        if is_fatal_generation_error(exc):
            lease.mark_broken(str(exc))
        else:
            lease.mark_soft_failure(str(exc))
        raise

    _raise_if_request_expired(context, "response_encoding")
    pcm_bytes = b"".join(pcm_chunks)
    buffer = io.BytesIO()
    if response_format == "pcm":
        buffer.write(pcm_bytes)
    else:
        buffer.write(build_wav_header(len(pcm_bytes), lease.model.sr))
        buffer.write(pcm_bytes)
    buffer.seek(0)
    return buffer, total_frames / float(lease.model.sr)


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
    voice_name, language_id = resolve_voice_and_language(request.voice)
    request_mode = request.stream_format or "audio"
    resolved_language = _validate_language_for_generation(language_id, request_mode)
    _validate_text_length(request.input, request_mode)
    resolved_response_format = _validate_response_format(request.response_format, request_mode)

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
                language_id=resolved_language,
                exaggeration=request.exaggeration,
                cfg_weight=request.cfg_weight,
                temperature=request.temperature,
                top_p=request.top_p,
                min_p=request.min_p,
                repetition_penalty=request.repetition_penalty,
                voice_name=voice_name,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
                "X-Audio-Sample-Rate": str(lease.model.sr),
                "X-Audio-Channels": "1",
                "X-Audio-Bits-Per-Sample": "16",
                "X-Model-Instance-ID": str(lease.instance_id),
                "X-Request-ID": context.request_id,
            },
        )

    if request.stream_format == "audio":
        context = _new_request_context(mode="audio_stream", client_request=client_request)
        observe_request_started("/v1/audio/speech", context.mode, len(request.input))
        _log_request_event(
            logging.INFO,
            "request_started",
            context,
            outcome="started",
            input_chars=len(request.input),
        )
        lease = await _acquire_request_lease(context)
        framer = _build_framer(resolved_response_format, lease.model.sr)
        return StreamingResponse(
            generate_speech_chunks(
                context=context,
                lease=lease,
                text=request.input,
                framer=framer,
                language_id=resolved_language,
                exaggeration=request.exaggeration,
                cfg_weight=request.cfg_weight,
                temperature=request.temperature,
                top_p=request.top_p,
                min_p=request.min_p,
                repetition_penalty=request.repetition_penalty,
                voice_name=voice_name,
            ),
            media_type=framer.media_type,
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "X-Audio-Sample-Rate": str(lease.model.sr),
                "X-Audio-Channels": "1",
                "X-Audio-Bits-Per-Sample": "16",
                "X-Usage-Input-Chars": str(len(request.input)),
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
            language_id=resolved_language,
            exaggeration=request.exaggeration,
            cfg_weight=request.cfg_weight,
            temperature=request.temperature,
            top_p=request.top_p,
            min_p=request.min_p,
            repetition_penalty=request.repetition_penalty,
            response_format=resolved_response_format,
            voice_name=voice_name,
        )
        media_type = "audio/pcm" if resolved_response_format == "pcm" else "audio/wav"
        response = StreamingResponse(
            buffer,
            media_type=media_type,
            headers={
                "Content-Disposition": f"attachment; filename=speech.{resolved_response_format}",
                "X-Audio-Sample-Rate": str(lease.model.sr),
                "X-Audio-Channels": "1",
                "X-Audio-Bits-Per-Sample": "16",
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
        )
        observe_request_finished(
            "/v1/audio/speech",
            context.mode,
            "success",
            elapsed_seconds=context.elapsed_seconds(),
            lease_wait_seconds=context.lease_wait_seconds(),
            generation_duration_seconds=context.generation_elapsed_seconds(),
            audio_seconds=round(audio_seconds, 6),
        )
        return response
    except RequestTimeoutExceeded as exc:
        observe_request_failure("timeout", exc.stage, context.mode)
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
        observe_request_failure("internal_error", "request", context.mode)
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
    "generate_speech_chunks",
    "generate_speech_sse",
    "resolve_voice_and_language",
]
