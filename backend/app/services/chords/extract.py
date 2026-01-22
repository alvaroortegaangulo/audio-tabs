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
    beat_times: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray, List[ChordSegment]]:
    """
    Chord extraction via chroma + templates + Viterbi.

    Returns: (chroma[12,frames], times[frames] seconds, segments[])
    """
    chroma, harm_rms = chroma_features(y=y, sr=int(sr), hop_length=int(hop_length))
    labels, T = build_chord_library(vocab)
    emissions = emission_probs(chroma, harm_rms, labels, T)
    path, conf = viterbi_decode(emissions, switch_penalty=float(switch_penalty))

    if beat_times is not None and len(beat_times) > 1:
        beat_frames = librosa.time_to_frames(
            np.asarray(beat_times, dtype=np.float32),
            sr=int(sr),
            hop_length=int(hop_length),
        )
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
