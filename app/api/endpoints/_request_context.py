"""
Shared request-lifecycle types and helpers for the text-to-speech endpoint.

Used by both ``app.api.endpoints.speech`` (buffered Exit A, endpoint dispatch, lease
acquisition) and ``app.api.endpoints.speech_stream`` (Exit B/C transports). Living
here, rather than in either of those two modules, avoids a circular import: speech
needs generate_speech_sse/generate_speech_chunks back from speech_stream, and
speech_stream needs these lifecycle helpers -- both edges can't point at the same
two files without a cycle.

The logger is bound to the "app.api.endpoints.speech" name (not this module's own
__name__) so log lines look identical regardless of which file emits them --
logging.getLogger() returns the same singleton for a given name no matter who asks.
"""

import asyncio
from dataclasses import dataclass
from typing import Optional

import torch
from fastapi import Request

from app.config import Config
from app.core.observability import get_logger, log_event
from app.core.tts_model import is_multilingual

logger = get_logger("app.api.endpoints.speech")


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


def _generation_kwargs(
    text: str,
    language_id: Optional[str],
    exaggeration: Optional[float],
    cfg_weight: Optional[float],
    temperature: Optional[float],
    top_p: Optional[float],
    min_p: Optional[float],
    repetition_penalty: Optional[float],
) -> dict:
    kwargs = {
        "text": text,
        "exaggeration": exaggeration if exaggeration is not None else Config.EXAGGERATION,
        "cfg_weight": cfg_weight if cfg_weight is not None else Config.CFG_WEIGHT,
        "temperature": temperature if temperature is not None else Config.TEMPERATURE,
        "top_p": top_p if top_p is not None else Config.TOP_P,
        "min_p": min_p if min_p is not None else Config.MIN_P,
        "repetition_penalty": repetition_penalty if repetition_penalty is not None else Config.REPETITION_PENALTY,
    }
    if is_multilingual() and language_id:
        kwargs["language_id"] = language_id
    return kwargs
