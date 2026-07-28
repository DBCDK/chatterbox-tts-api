"""Tier 2 null test: end-to-end statistical comparison (Phase 0 Deliverable 4).

For full-request comparisons where sample-exactness is not achievable (the doc's own
example: after Phase 3 changes the decode loop's randomness consumption). Catches gross
regressions -- duration drift, missing inter-sentence silence, a wrong voice/spectral
envelope -- without requiring bit-exact waveforms. Not a substitute for the Tier 1
sample-exact gate or for human listening sign-off.
"""

from dataclasses import dataclass

import torch
import torchaudio


SILENCE_RMS_THRESHOLD = 0.02
MIN_SILENCE_MS = 40
MIN_SEGMENT_MS = 100  # below this, treat as a detector artifact, not a real sentence


@dataclass
class SentenceFingerprint:
    duration_s: float
    rms: float
    mean_log_mel: torch.Tensor


@dataclass
class Fingerprint:
    total_duration_s: float
    sentences: list
    silence_gaps_ms: list


def _mel_spectrogram(sr: int):
    return torchaudio.transforms.MelSpectrogram(
        sample_rate=sr, n_fft=1024, hop_length=256, n_mels=80
    )


def find_silence_gaps(wav: torch.Tensor, sr: int, frame_ms: float = 10.0):
    """Detect silent regions by frame RMS, merging adjacent silent frames. Returns a list
    of (start_sample, end_sample) for gaps at least MIN_SILENCE_MS long."""
    wav = wav.squeeze()
    frame_len = int(sr * frame_ms / 1000)
    n_frames = wav.shape[-1] // frame_len
    gaps = []
    gap_start = None
    for i in range(n_frames):
        frame = wav[i * frame_len : (i + 1) * frame_len]
        rms = frame.pow(2).mean().sqrt().item()
        if rms < SILENCE_RMS_THRESHOLD:
            if gap_start is None:
                gap_start = i * frame_len
        else:
            if gap_start is not None:
                gap_end = i * frame_len
                if (gap_end - gap_start) / sr * 1000 >= MIN_SILENCE_MS:
                    gaps.append((gap_start, gap_end))
                gap_start = None
    if gap_start is not None:
        gap_end = n_frames * frame_len
        if (gap_end - gap_start) / sr * 1000 >= MIN_SILENCE_MS:
            gaps.append((gap_start, gap_end))
    return gaps


def split_on_silence(wav: torch.Tensor, sr: int, gaps):
    """Split wav into non-silent segments using detected gaps as boundaries. Segments
    shorter than MIN_SEGMENT_MS are dropped as detector artifacts (e.g. a brief energy
    blip splitting what is otherwise one silence region) rather than real sentences --
    real sentences are hundreds of ms or longer."""
    wav = wav.squeeze()
    min_samples = int(sr * MIN_SEGMENT_MS / 1000)
    segments = []
    cursor = 0
    for start, end in gaps:
        if start > cursor:
            segments.append(wav[cursor:start])
        cursor = end
    if cursor < wav.shape[-1]:
        segments.append(wav[cursor:])
    return [s for s in segments if s.shape[-1] >= min_samples]


def compute_fingerprint(wav: torch.Tensor, sr: int) -> Fingerprint:
    wav = wav.squeeze().float()
    gaps = find_silence_gaps(wav, sr)
    segments = split_on_silence(wav, sr, gaps)
    mel = _mel_spectrogram(sr)

    sentences = []
    for seg in segments:
        log_mel = torch.log(mel(seg.unsqueeze(0)) + 1e-6).mean(dim=-1).squeeze(0)
        sentences.append(
            SentenceFingerprint(
                duration_s=seg.shape[-1] / sr,
                rms=seg.pow(2).mean().sqrt().item(),
                mean_log_mel=log_mel,
            )
        )

    return Fingerprint(
        total_duration_s=wav.shape[-1] / sr,
        sentences=sentences,
        silence_gaps_ms=[(e - s) / sr * 1000 for s, e in gaps],
    )


@dataclass
class FingerprintDiff:
    duration_diff_s: float
    sentence_count_diff: int
    rms_diffs: list
    mel_cosine_distances: list
    silence_gap_count_diff: int
    silence_gap_ms_diffs: list


def compare_fingerprints(a: Fingerprint, b: Fingerprint) -> FingerprintDiff:
    n = min(len(a.sentences), len(b.sentences))
    rms_diffs = [abs(a.sentences[i].rms - b.sentences[i].rms) for i in range(n)]
    mel_cosine_distances = []
    for i in range(n):
        ma, mb = a.sentences[i].mean_log_mel, b.sentences[i].mean_log_mel
        cos_sim = torch.nn.functional.cosine_similarity(ma.unsqueeze(0), mb.unsqueeze(0)).item()
        mel_cosine_distances.append(1 - cos_sim)

    m = min(len(a.silence_gaps_ms), len(b.silence_gaps_ms))
    silence_gap_ms_diffs = [abs(a.silence_gaps_ms[i] - b.silence_gaps_ms[i]) for i in range(m)]

    return FingerprintDiff(
        duration_diff_s=abs(a.total_duration_s - b.total_duration_s),
        sentence_count_diff=abs(len(a.sentences) - len(b.sentences)),
        rms_diffs=rms_diffs,
        mel_cosine_distances=mel_cosine_distances,
        silence_gap_count_diff=abs(len(a.silence_gaps_ms) - len(b.silence_gaps_ms)),
        silence_gap_ms_diffs=silence_gap_ms_diffs,
    )
