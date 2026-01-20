from __future__ import annotations

from typing import List, Tuple

import librosa
import numpy as np

from app.schemas import ChordSegment
from app.services.chords.template import (
    build_chord_library,
    chroma_features,
    emission_probs,
    finalize_segments,
    frames_to_segments,
)
from app.services.chords.viterbi import viterbi_decode


def extract_chords_template(
    y: np.ndarray,
    sr: int,
    *,
    vocab: str = "majmin7",
    switch_penalty: float = 2.5,
    min_segment_sec: float = 0.25,
    hop_length: int = 512,
) -> Tuple[np.ndarray, np.ndarray, List[ChordSegment]]:
    """
    Extrae una progresión de acordes a partir de cromagrama + plantillas + Viterbi.

    Returns: (chroma[12,frames], times[frames] seconds, segments[])
    """
    chroma = chroma_features(y=y, sr=int(sr), hop_length=int(hop_length))
    labels, T = build_chord_library(vocab)
    emissions = emission_probs(chroma, labels, T)
    path, conf = viterbi_decode(emissions, switch_penalty=float(switch_penalty))

    times = librosa.frames_to_time(
        np.arange(chroma.shape[1], dtype=np.int32),
        sr=int(sr),
        hop_length=int(hop_length),
    ).astype(np.float32)

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

