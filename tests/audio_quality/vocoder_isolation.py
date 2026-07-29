"""Tier 1 null test: vocoder isolation for chunk-boundary regressions.

Phase 0 Deliverable 4 (see prompts/streaming-phase-0-plan.md). This isolates exactly the
code path Phase 3 (token-level streaming) will change -- the HiFTGAN mel-to-waveform
stage -- independent of T3's stochastic token sampling, by generating speech tokens once
and reusing that fixed tensor for every comparison below.

Two real findings from building this harness, worth knowing before Phase 3 starts:

1. The flow-matching stage's streaming hook is currently broken: calling
   ``S3Token2Wav.flow_inference(..., finalize=False)`` on a growing token prefix raises a
   shape-mismatch RuntimeError. ``flow.py``'s ``inference()`` trims ``h`` by
   ``pre_lookahead_len * token_mel_ratio`` (6 frames) when ``finalize=False`` but computes
   ``h_lengths``/``mask`` from the *pre-trim* encoder mask, so they never match. This means
   true incremental (token-level) mel computation is not usable via the public API today.
2. ``HiFTGenerator``'s ``cache_source`` mechanism ("avoid glitch", hifigan.py) alone does
   NOT prevent chunk-boundary discontinuities: carrying the excitation-signal tail across
   calls barely changes the boundary error versus not carrying it at all (both show large,
   >100%-of-peak deviations right at the seam for a naive non-overlapping mel split). Only
   the excitation source is carried -- the mel-decoder's own convolutional state is not.
   What actually works is overlap-and-trim: decode ``mel_chunk`` plus a few frames of
   context on each side, then keep only the center portion. Empirically, ~10 frames of
   context on each side reduces boundary deviation by 50-100x. This is the mechanism this
   harness exercises for the "good" case, and Phase 3 should account for it -- naive
   token-level chunking without ~10 frames of look-ahead/look-behind context around each
   chunk boundary in the HiFTGAN stage will glitch.
"""

import os
from functools import lru_cache

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
import torch.nn.functional as F

from chatterbox.models.s3tokenizer import drop_invalid_tokens
from chatterbox.mtl_tts import ChatterboxMultilingualTTS, punc_norm

FLOW_SEED = 1234
# voices/mic-voice.wav is checked into git.
DEFAULT_VOICE_SAMPLE = os.path.join(
    os.path.dirname(__file__), "..", "..", "voices", "mic-voice.wav"
)


@lru_cache(maxsize=1)
def load_model(
    voice_sample_path: str = DEFAULT_VOICE_SAMPLE,
) -> ChatterboxMultilingualTTS:
    model = ChatterboxMultilingualTTS.from_pretrained(device=None)
    model.prepare_conditionals(voice_sample_path)
    return model


def generate_speech_tokens(
    model: ChatterboxMultilingualTTS,
    text: str,
    language_id: str = "da",
    seed: int = 0,
) -> torch.Tensor:
    """Generate speech tokens once for a fixed input. Cache/reuse the returned tensor --
    T3 sampling is stochastic, so this is the only call in the pipeline not made
    deterministic by seeding alone; callers must reuse the same tensor across comparisons."""
    torch.manual_seed(seed)
    text_tokens = model.tokenizer.text_to_tokens(
        punc_norm(text), language_id=language_id
    )
    text_tokens = text_tokens.to(model.device)
    text_tokens = torch.cat([text_tokens, text_tokens], dim=0)  # CFG needs two seqs
    sot, eot = model.t3.hp.start_text_token, model.t3.hp.stop_text_token
    text_tokens = F.pad(text_tokens, (1, 0), value=sot)
    text_tokens = F.pad(text_tokens, (0, 1), value=eot)

    with torch.inference_mode():
        speech_tokens = model.t3.inference(
            t3_cond=model.conds.t3,
            text_tokens=text_tokens,
            max_new_tokens=1000,
            temperature=0.8,
            cfg_weight=0.5,
            repetition_penalty=2.0,
            min_p=0.05,
            top_p=1.0,
        )
    speech_tokens = speech_tokens[0]
    return drop_invalid_tokens(speech_tokens).to(model.device)


def synthesize_whole(model: ChatterboxMultilingualTTS, speech_tokens: torch.Tensor):
    """The reference: one flow_inference + one hift_inference call over all tokens."""
    torch.manual_seed(FLOW_SEED)
    with torch.inference_mode():
        mel = model.s3gen.flow_inference(
            speech_tokens.unsqueeze(0), ref_dict=model.conds.gen, finalize=True
        )
        wav, _ = model.s3gen.hift_inference(mel, None)
    return mel, wav.cpu()


def synthesize_chunked(
    model: ChatterboxMultilingualTTS,
    mel: torch.Tensor,
    mel_chunk_frames: int = 40,
    overlap_frames: int = 10,
):
    """Split a precomputed mel into chunks and vocode each independently, keeping only
    the center [start:end] portion of each chunk's decode and discarding the
    overlap-context edges. Returns (wav, boundary_sample_offsets).

    overlap_frames=0 reproduces the naive, glitchy chunking this harness exists to catch.
    """
    total_frames = mel.shape[-1]
    hop = 1  # filled in on the first chunk once we know the true mel->sample ratio
    wavs = []
    boundaries = []
    start = 0
    while start < total_frames:
        end = min(start + mel_chunk_frames, total_frames)
        ctx_start = max(0, start - overlap_frames)
        ctx_end = min(total_frames, end + overlap_frames)
        mel_ctx = mel[:, :, ctx_start:ctx_end]
        with torch.inference_mode():
            wav_ctx, _ = model.s3gen.hift_inference(mel_ctx, None)
        hop = wav_ctx.shape[-1] // mel_ctx.shape[-1]
        left_trim = (start - ctx_start) * hop
        right_len = (end - start) * hop
        wav_center = wav_ctx[:, left_trim : left_trim + right_len]
        if start > 0:
            boundaries.append(sum(w.shape[-1] for w in wavs))
        wavs.append(wav_center.cpu())
        start = end
    return torch.cat(wavs, dim=-1), boundaries


def boundary_max_deviation(
    reference: torch.Tensor, test: torch.Tensor, boundaries, window: int = 80
):
    """Max absolute sample deviation in a window around each chunk boundary. This is the
    doc-recommended check: a global average SNR hides localized clicks, so look at the
    seams specifically."""
    reference = reference.squeeze()
    test = test.squeeze()
    n = min(reference.shape[-1], test.shape[-1])
    results = []
    for b in boundaries:
        lo, hi = max(0, b - window), min(n, b + window)
        ref_win = reference[lo:hi]
        test_win = test[lo:hi]
        results.append(
            {
                "boundary": b,
                "max_deviation": (test_win - ref_win).abs().max().item(),
                "ref_peak": ref_win.abs().max().item(),
            }
        )
    return results


def segmental_snr_db(
    reference: torch.Tensor,
    test: torch.Tensor,
    seg_len: int = 2400,
    floor: float = 1e-4,
):
    """Segmental SNR over non-silent segments (doc-suggested threshold: >40 dB). Reported
    as a secondary/informational metric -- see the module docstring and test file for why
    the boundary-window check above is the primary gate."""
    reference = reference.squeeze()
    test = test.squeeze()
    n = min(reference.shape[-1], test.shape[-1])
    reference, test = reference[:n], test[:n]
    snrs = []
    for i in range(0, n - seg_len, seg_len):
        ref_seg = reference[i : i + seg_len]
        if ref_seg.abs().max() < floor:
            continue
        err = test[i : i + seg_len] - ref_seg
        num = (ref_seg**2).sum()
        den = (err**2).sum().clamp_min(1e-12)
        snrs.append((10 * torch.log10(num / den)).item())
    return snrs
