"""
Streaming transports for the text-to-speech endpoint (Exit B and Exit C).

Request-lifecycle helpers (context, guards, generation kwargs) live in
``app.api.endpoints._request_context``, shared with ``app.api.endpoints.speech``,
which keeps validation, lease acquisition, the buffered Exit A path, and the routed
endpoint itself -- see Phase 1 of the streaming redesign
(prompts/streaming-phase-1-plan.md). This module holds only the two generator
functions that turn one audio source into bytes on the wire, plus the framer
selection helper they share with Exit A's response headers.
"""

import asyncio
import base64
import logging
from typing import AsyncGenerator, Optional

from app.core.audio import Framer, PcmFramer, WavStreamFramer, pcm16_bytes
from app.core.metrics import (
    observe_request_failure,
    observe_request_finished,
    observe_time_to_first_chunk,
)
from app.core.tts_model import ModelLease, apply_voice_to_lease, release_model_lease
from app.models import SSEAudioDelta, SSEAudioDone, SSEAudioInfo, SSEUsageInfo

from app.api.endpoints._request_context import (
    ClientDisconnected,
    RequestRuntimeContext,
    RequestTimeoutExceeded,
    _audio_num_frames,
    _generation_kwargs,
    _guard_request_state,
    _log_request_event,
)


def _build_framer(response_format: str, sample_rate: int) -> Framer:
    if response_format == "pcm":
        return PcmFramer()
    return WavStreamFramer(sample_rate=sample_rate)


async def generate_speech_sse(
    context: RequestRuntimeContext,
    lease: ModelLease,
    text: str,
    language_id: Optional[str] = None,
    exaggeration: Optional[float] = None,
    cfg_weight: Optional[float] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    min_p: Optional[float] = None,
    repetition_penalty: Optional[float] = None,
    voice_name: Optional[str] = None,
) -> AsyncGenerator[str, None]:
    chunk_count = 0
    try:
        await _guard_request_state(context, "sse_start")
        info_event = SSEAudioInfo(
            sample_rate=lease.model.sr,
            channels=1,
            bits_per_sample=16,
            format="pcm",
        )
        yield f"data: {info_event.model_dump_json()}\n\n"

        if voice_name is not None:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, apply_voice_to_lease, lease, voice_name)

        total_frames = 0
        first_chunk_observed = False

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
            await _guard_request_state(context, "chunk_generation")
            audio_tensor = audio_tensor.detach().cpu()
            if not first_chunk_observed:
                observe_time_to_first_chunk("/v1/audio/speech", context.elapsed_seconds())
                first_chunk_observed = True
            chunk_count += 1
            total_frames += _audio_num_frames(audio_tensor)
            payload = base64.b64encode(pcm16_bytes(audio_tensor)).decode("ascii")
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
        observe_request_failure("timeout", exc.stage, context.mode)
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
        observe_request_failure("client_disconnect", exc.stage, context.mode)
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
        observe_request_failure("internal_error", "request", context.mode)
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


async def generate_speech_chunks(
    context: RequestRuntimeContext,
    lease: ModelLease,
    text: str,
    framer: Framer,
    language_id: Optional[str] = None,
    exaggeration: Optional[float] = None,
    cfg_weight: Optional[float] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    min_p: Optional[float] = None,
    repetition_penalty: Optional[float] = None,
    voice_name: Optional[str] = None,
) -> AsyncGenerator[bytes, None]:
    """Exit B: a genuine chunked byte stream, framed by ``framer``.

    Mirrors generate_speech_sse's guard/metrics/logging structure -- same audio
    source, same failure handling, different terminal encoding (raw framed bytes
    instead of base64-in-JSON).
    """
    chunk_count = 0
    total_frames = 0
    try:
        await _guard_request_state(context, "stream_start")
        yield framer.header()

        if voice_name is not None:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, apply_voice_to_lease, lease, voice_name)

        first_chunk_observed = False
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
            await _guard_request_state(context, "chunk_generation")
            audio_tensor = audio_tensor.detach().cpu()
            if not first_chunk_observed:
                observe_time_to_first_chunk("/v1/audio/speech", context.elapsed_seconds())
                first_chunk_observed = True
            chunk_count += 1
            total_frames += _audio_num_frames(audio_tensor)
            yield framer.frame(audio_tensor)

        yield framer.finalize()

        audio_seconds = total_frames / float(lease.model.sr)
        _log_request_event(
            logging.INFO,
            "request_completed",
            context,
            model_instance_id=lease.instance_id,
            outcome="success",
            input_chars=len(text),
            audio_seconds=round(audio_seconds, 6),
            chunk_count=chunk_count,
        )
        observe_request_finished(
            "/v1/audio/speech",
            context.mode,
            "success",
            elapsed_seconds=context.elapsed_seconds(),
            lease_wait_seconds=context.lease_wait_seconds(),
            generation_duration_seconds=context.generation_elapsed_seconds(),
            audio_seconds=round(audio_seconds, 6),
            chunk_count=chunk_count,
        )
    except RequestTimeoutExceeded as exc:
        observe_request_failure("timeout", exc.stage, context.mode)
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
        observe_request_failure("client_disconnect", exc.stage, context.mode)
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
        observe_request_failure("internal_error", "request", context.mode)
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


__all__ = ["generate_speech_chunks", "generate_speech_sse"]
