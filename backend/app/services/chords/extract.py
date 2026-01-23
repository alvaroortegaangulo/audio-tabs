from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np

from app.core.config import settings
from app.schemas import ChordSegment
from app.services.chords.deep_chords import extract_chords_deep
from app.services.chords.template import (
    build_chord_library,
    emission_probs,
    finalize_segments,
    frames_to_segments,
)
from app.services.chords.viterbi import viterbi_decode

_DEEPCHROMA_FPS = 10


def _as_madmom_input(
    y_or_path: np.ndarray | str | Path,
    sr: int | None,
):
    if isinstance(y_or_path, (str, Path)):
        return str(y_or_path)

    if sr is None:
        raise ValueError("Sample rate is required when providing a numpy array.")

    try:
        from madmom.audio.signal import Signal
    except Exception as exc:
        raise RuntimeError("madmom is required for DeepChroma processing.") from exc

    arr = np.asarray(y_or_path, dtype=np.float32)
    if arr.ndim == 1:
        return Signal(arr, sample_rate=int(sr), num_channels=1)
    if arr.ndim == 2:
        return Signal(arr, sample_rate=int(sr))
    raise ValueError(f"Unsupported audio shape: {arr.shape}")


def _deep_chroma(
    y_or_path: np.ndarray | str | Path,
    sr: int | None,
) -> tuple[np.ndarray, np.ndarray, float]:
    try:
        from madmom.audio.chroma import DeepChromaProcessor
    except Exception as exc:
        raise RuntimeError("madmom is required for DeepChroma processing.") from exc

    proc = DeepChromaProcessor()
    chroma_frames = proc(_as_madmom_input(y_or_path, sr))
    chroma_frames = np.asarray(chroma_frames, dtype=np.float32)
    if chroma_frames.ndim != 2 or chroma_frames.shape[1] != 12:
        raise ValueError(f"Unexpected DeepChroma shape: {chroma_frames.shape}")

    chroma_raw = chroma_frames.T  # [12, frames]
    harm_rms = np.mean(chroma_raw, axis=0)
    harm_rms = np.clip(harm_rms, 0.0, 1.0).astype(np.float32)

    chroma_norm = chroma_raw / (np.linalg.norm(chroma_raw, axis=0, keepdims=True) + 1e-9)
    # DeepChromaProcessor en madmom 0.16.1 usa fps=10 por defecto
    fps = float(getattr(proc, "fps", _DEEPCHROMA_FPS))
    return chroma_norm.astype(np.float32), harm_rms, fps


def extract_chords_template(
    y: np.ndarray | str | Path,
    sr: int | None = None,
    *,
    vocab: str = "majmin7",
    switch_penalty: float = 2.5,
    min_segment_sec: float = 0.25,
    hop_length: int = 512,
    beat_times: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray, List[ChordSegment]]:
    """
    Chord extraction via DeepChroma + templates + Viterbi.
    hop_length is kept for compatibility but ignored.

    Returns: (chroma[12,frames], times[frames] seconds, segments[])
    """
    backend = str(getattr(settings, "CHORD_DETECTION_BACKEND", "template") or "template").strip().lower()
    if backend == "deep":
        return extract_chords_deep(
            y,
            sr,
            vocab=vocab,
            switch_penalty=switch_penalty,
            min_segment_sec=min_segment_sec,
            hop_length=hop_length,
            beat_times=beat_times,
        )

    chroma, harm_rms, fps = _deep_chroma(y, sr)
    labels, T = build_chord_library(vocab)
    emissions = emission_probs(chroma, harm_rms, labels, T)
    path, conf = viterbi_decode(emissions, switch_penalty=float(switch_penalty))

    if beat_times is not None and len(beat_times) > 1:
        beat_frames = np.round(np.asarray(beat_times, dtype=np.float32) * float(fps)).astype(int)
        beat_frames = beat_frames[(beat_frames > 0) & (beat_frames < chroma.shape[1])]
        if beat_frames.size > 0:
            beat_frames = np.unique(np.concatenate(([0], beat_frames, [chroma.shape[1]])))
            for a, b in zip(beat_frames[:-1], beat_frames[1:]):
                seg = path[a:b]
                if seg.size == 0:
                    continue
                vals, cnts = np.unique(seg, return_counts=True)
                path[a:b] = vals[int(np.argmax(cnts))]
            conf = emissions[path, np.arange(len(path))].astype(np.float32)

    times = (np.arange(chroma.shape[1], dtype=np.float32) / float(fps)).astype(np.float32)

    raw_segments = frames_to_segments(path, conf, times, min_len=float(min_segment_sec))
    segments = finalize_segments(raw_segments, labels)

    out: List[ChordSegment] = []
    for seg in segments:
        out.append(
            ChordSegment(
                start=float(seg.start),
                end=float(seg.end),
                label=str(seg.label),
                confidence=float(seg.confidence),
            )
        )

    return chroma, times, out
