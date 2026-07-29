"""Byte framing for the speech endpoint's audio source.

Establishes the seam described in ``prompts/streaming-phase-1-plan.md``:

    AudioSource (async iterator of float32 tensors) -> Framer -> Exit

A ``Framer`` turns float32 tensors into bytes for one of ``response_format``'s
two values. Both framers emit **16-bit signed little-endian integer PCM** --
that contract is load-bearing for glyph-gate's byte-derived usage metering
(``payload_bytes / (sample_rate * channels * bytes_per_sample)``), which reads
exactly half the true duration if a framer ever emits 32-bit float instead.
"""

from typing import Protocol

import torch

#: Sentinel for RIFF/data chunk sizes that are not known up front, matching
#: what ffmpeg's WAV muxer writes to non-seekable output.
UNKNOWN_SIZE = 0xFFFFFFFF


class Framer(Protocol):
    media_type: str

    def header(self) -> bytes:
        """Bytes to send before the first frame. Empty for raw PCM."""
        ...

    def frame(self, chunk: torch.Tensor) -> bytes:
        """Encode one float32 chunk in [-1, 1] to wire bytes."""
        ...

    def finalize(self) -> bytes:
        """Bytes to send after the last frame. Empty for both framers here --
        neither can rewrite an already-sent streaming header, so there is
        nothing left to say once framing starts."""
        ...


def pcm16_bytes(chunk: torch.Tensor) -> bytes:
    """Convert a float32 tensor in [-1, 1] to 16-bit little-endian PCM bytes.

    Clamping happens before scaling so samples at +/-1.5 land at +/-32767
    rather than wrapping through int16 overflow. The explicit ``<i2`` dtype
    forces little-endian output regardless of host byte order -- the prior
    SSE path relied on native-endian ``.tobytes()``, an unstated contract
    that happens to hold on x86/ARM but is not guaranteed.
    """
    clamped = torch.clamp(chunk, -1.0, 1.0)
    int16 = (clamped * 32767.0).to(torch.int16)
    return int16.squeeze().cpu().numpy().astype("<i2").tobytes()


class PcmFramer:
    """Headerless 16-bit LE PCM. What OpenAI and compatible servers call
    ``pcm``. Not ``audio/L16`` -- that IANA type specifies big-endian."""

    media_type = "audio/pcm"

    def header(self) -> bytes:
        return b""

    def frame(self, chunk: torch.Tensor) -> bytes:
        return pcm16_bytes(chunk)

    def finalize(self) -> bytes:
        return b""


def build_wav_header(
    data_size: int,
    sample_rate: int,
    channels: int = 1,
    bits_per_sample: int = 16,
) -> bytes:
    """Hand-written 44-byte canonical RIFF/WAV header for integer PCM.

    Pass ``UNKNOWN_SIZE`` for ``data_size`` when the total length is not yet
    known (the streaming case); pass the real byte count once it is (the
    buffered case). Never build this with ``torchaudio.save`` on a float32
    tensor -- it writes ``audioFormat: 3`` at double the byte rate, which
    silently halves glyph-gate's byte-derived ``audio_seconds``.
    """
    block_align = channels * bits_per_sample // 8
    byte_rate = sample_rate * block_align
    chunk_size = UNKNOWN_SIZE if data_size == UNKNOWN_SIZE else 36 + data_size

    header = bytearray(44)
    header[0:4] = b"RIFF"
    header[4:8] = chunk_size.to_bytes(4, "little")
    header[8:12] = b"WAVE"
    header[12:16] = b"fmt "
    header[16:20] = (16).to_bytes(4, "little")  # subchunk size
    header[20:22] = (1).to_bytes(2, "little")  # audioFormat: integer PCM
    header[22:24] = channels.to_bytes(2, "little")
    header[24:28] = sample_rate.to_bytes(4, "little")
    header[28:32] = byte_rate.to_bytes(4, "little")
    header[32:34] = block_align.to_bytes(2, "little")
    header[34:36] = bits_per_sample.to_bytes(2, "little")
    header[36:40] = b"data"
    header[40:44] = data_size.to_bytes(4, "little")
    return bytes(header)


class WavStreamFramer:
    """RIFF/WAV framing with an unknown-length streaming header.

    ``header()`` always emits ``UNKNOWN_SIZE`` placeholders, since a
    streaming HTTP body cannot seek back to patch in a real size once bytes
    have been sent. The buffered exit (Exit A) does not use this class's
    ``header()`` for its response -- it knows the true length up front and
    calls ``build_wav_header`` directly with the real ``data_size``.
    """

    media_type = "audio/wav"

    def __init__(self, sample_rate: int, channels: int = 1, bits_per_sample: int = 16):
        self.sample_rate = sample_rate
        self.channels = channels
        self.bits_per_sample = bits_per_sample

    def header(self) -> bytes:
        return build_wav_header(
            UNKNOWN_SIZE, self.sample_rate, self.channels, self.bits_per_sample
        )

    def frame(self, chunk: torch.Tensor) -> bytes:
        return pcm16_bytes(chunk)

    def finalize(self) -> bytes:
        return b""
