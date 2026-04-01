"""Response models for the active API surface."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Health check response model"""

    status: str
    model_loaded: bool
    device: str
    config: Dict[str, Any]
    memory_info: Optional[Dict[str, float]] = None
    initialization_state: Optional[str] = None
    initialization_progress: Optional[str] = None
    initialization_error: Optional[str] = None


class ModelInfo(BaseModel):
    """Individual model information"""

    id: str
    object: str
    created: int
    owned_by: str


class ModelsResponse(BaseModel):
    """Models listing response"""

    object: str
    data: List[ModelInfo]


class ErrorResponse(BaseModel):
    """Error response model"""

    error: Dict[str, str]


class SSEUsageInfo(BaseModel):
    """Usage information for SSE completion event"""

    input_chars: int
    audio_seconds: float


class SSEAudioInfo(BaseModel):
    """SSE audio metadata event model"""

    type: str = "speech.audio.info"
    sample_rate: int
    channels: int
    bits_per_sample: int


class SSEAudioDelta(BaseModel):
    """SSE audio delta event model"""

    type: str = "speech.audio.delta"
    audio: str  # Base64 encoded audio chunk


class SSEAudioDone(BaseModel):
    """SSE audio completion event model"""

    type: str = "speech.audio.done"
    usage: SSEUsageInfo
