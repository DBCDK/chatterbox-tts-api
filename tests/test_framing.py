"""Unit tests for app/core/audio/framing.py. No server required."""

import struct

import torch

from app.core.audio import PcmFramer, WavStreamFramer, UNKNOWN_SIZE, build_wav_header


def _parse_wav_header(header: bytes):
    assert len(header) == 44
    riff, chunk_size, wave, fmt_tag, subchunk_size = struct.unpack("<4sI4s4sI", header[0:20])
    audio_format, channels, sample_rate, byte_rate, block_align, bits_per_sample = (
        struct.unpack("<HHIIHH", header[20:36])
    )
    data_tag, data_size = struct.unpack("<4sI", header[36:44])
    assert riff == b"RIFF"
    assert wave == b"WAVE"
    assert fmt_tag == b"fmt "
    assert subchunk_size == 16
    assert data_tag == b"data"
    return {
        "chunk_size": chunk_size,
        "audio_format": audio_format,
        "channels": channels,
        "sample_rate": sample_rate,
        "byte_rate": byte_rate,
        "block_align": block_align,
        "bits_per_sample": bits_per_sample,
        "data_size": data_size,
    }


class TestWavStreamFramerHeader:
    def test_unbuffered_header_uses_unknown_size(self):
        header = WavStreamFramer(sample_rate=24000).header()
        fields = _parse_wav_header(header)
        assert fields["chunk_size"] == UNKNOWN_SIZE
        assert fields["data_size"] == UNKNOWN_SIZE

    def test_buffered_header_uses_real_size(self):
        header = build_wav_header(data_size=1000, sample_rate=24000)
        fields = _parse_wav_header(header)
        assert fields["chunk_size"] == 36 + 1000
        assert fields["data_size"] == 1000

    def test_header_fields_are_integer_pcm_16_bit_mono_24k(self):
        for header in (
            WavStreamFramer(sample_rate=24000).header(),
            build_wav_header(data_size=0, sample_rate=24000),
        ):
            fields = _parse_wav_header(header)
            assert fields["audio_format"] == 1  # integer PCM, not float (3)
            assert fields["bits_per_sample"] == 16
            assert fields["channels"] == 1
            assert fields["sample_rate"] == 24000
            assert fields["byte_rate"] == 48000
            assert fields["block_align"] == 2

    def test_empty_buffered_header_is_still_valid(self):
        header = build_wav_header(data_size=0, sample_rate=24000)
        fields = _parse_wav_header(header)
        assert fields["chunk_size"] == 36
        assert fields["data_size"] == 0


class TestPcmFramer:
    def test_header_and_finalize_are_empty(self):
        framer = PcmFramer()
        assert framer.header() == b""
        assert framer.finalize() == b""

    def test_frame_byte_count_is_two_per_sample(self):
        framer = PcmFramer()
        chunk = torch.zeros(1, 100)
        assert len(framer.frame(chunk)) == 200

    def test_frame_is_little_endian(self):
        framer = PcmFramer()
        # 0.5 * 32767 = 16383.5 -> truncates to 16383 = 0x3FFF
        chunk = torch.tensor([[0.5]])
        frame = framer.frame(chunk)
        assert frame == bytes([0xFF, 0x3F])

    def test_clamping_does_not_wrap_sign(self):
        framer = PcmFramer()
        chunk = torch.tensor([[1.5, -1.5]])
        high, low = struct.unpack("<2h", framer.frame(chunk))
        assert high == 32767
        assert low == -32767
