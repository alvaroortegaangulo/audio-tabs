from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np

from app.core.config import settings
from app.schemas import ChordSegment
from app.services.chords.chord_vocabulary import normalize_madmom_chord_label

_DEEPCHROMA_FPS = 10
_CHORD_PROCESSOR = None


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
        raise RuntimeError("madmom is required for chord recognition.") from exc

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
    fps = float(getattr(proc, "fps", _DEEPCHROMA_FPS))
    return chroma_norm.astype(np.float32), harm_rms, fps


def _get_chord_processor():
    """
    Use CRFChordRecognitionProcessor by default because it models temporal
    structure explicitly and exposes a richer chord vocabulary than the
    plain DeepChromaChordRecognitionProcessor.
    """
    global _CHORD_PROCESSOR
    if _CHORD_PROCESSOR is not None:
        return _CHORD_PROCESSOR

    try:
        from madmom.features.chords import CRFChordRecognitionProcessor, DeepChromaChordRecognitionProcessor
    except Exception as exc:
        raise RuntimeError("madmom is required for chord recognition.") from exc

    try:
        _CHORD_PROCESSOR = CRFChordRecognitionProcessor()
    except Exception:
        _CHORD_PROCESSOR = DeepChromaChordRecognitionProcessor()
    return _CHORD_PROCESSOR


def _segments_from_labels(
    times: list[float],
    labels: list[str],
    default_step: float,
) -> list[tuple[float, float, str]]:
    segments: list[tuple[float, float, str]] = []
    if not times or not labels:
        return segments
    n = min(len(times), len(labels))
    for i in range(n):
        start = float(times[i])
        if i + 1 < n:
            end = float(times[i + 1])
        else:
            end = float(times[i] + default_step)
        segments.append((start, end, str(labels[i])))
    return segments


def _coerce_segments(
    raw: object,
    *,
    fps: float,
    classes: list[str] | None,
) -> list[tuple[float, float, str, float]]:
    if raw is None:
        return []

    if isinstance(raw, np.ndarray):
        raw_list = raw.tolist()
    elif isinstance(raw, (list, tuple)):
        raw_list = list(raw)
    else:
        raw_list = [raw]

    if not raw_list:
        return []

    if all(isinstance(item, str) for item in raw_list):
        times = [i / float(fps) for i in range(len(raw_list))]
        segments = _segments_from_labels(times, list(raw_list), 1.0 / float(fps))
        return [(s, e, lbl, 1.0) for s, e, lbl in segments]

    segments: list[tuple[float, float, str, float]] = []
    time_labels: list[tuple[float, str]] = []

    for entry in raw_list:
        if hasattr(entry, "start") and hasattr(entry, "end") and hasattr(entry, "label"):
            start = float(getattr(entry, "start"))
            end = float(getattr(entry, "end"))
            label = str(getattr(entry, "label"))
            conf = float(getattr(entry, "confidence", 1.0))
            segments.append((start, end, label, conf))
            continue

        if isinstance(entry, (list, tuple)):
            if len(entry) >= 3 and isinstance(entry[2], str):
                start = float(entry[0])
                end = float(entry[1])
                label = str(entry[2])
                conf = float(entry[3]) if len(entry) > 3 else 1.0
                segments.append((start, end, label, conf))
                continue

            if len(entry) >= 2 and isinstance(entry[1], str):
                time_labels.append((float(entry[0]), str(entry[1])))
                continue

            if len(entry) >= 2 and classes is not None:
                idx = int(entry[1])
                if 0 <= idx < len(classes):
                    time_labels.append((float(entry[0]), str(classes[idx])))
                    continue

    if segments:
        return segments

    if time_labels:
        times = [t for t, _ in time_labels]
        labels = [l for _, l in time_labels]
        default_step = 1.0 / float(fps if fps > 0 else _DEEPCHROMA_FPS)
        segs = _segments_from_labels(times, labels, default_step)
        return [(s, e, lbl, 1.0) for s, e, lbl in segs]

    return []


def _merge_adjacent_segments(
    segments: list[ChordSegment],
) -> list[ChordSegment]:
    if not segments:
        return []
    merged: list[ChordSegment] = [segments[0]]
    for seg in segments[1:]:
        prev = merged[-1]
        if seg.label == prev.label:
            merged[-1] = ChordSegment(
                start=float(prev.start),
                end=float(seg.end),
                label=str(prev.label),
                confidence=max(float(prev.confidence), float(seg.confidence)),
            )
        else:
            merged.append(seg)
    return merged


def _smooth_segments(
    segments: list[ChordSegment],
    min_len: float,
) -> list[ChordSegment]:
    if not segments or min_len <= 0:
        return segments

    out: list[ChordSegment] = list(segments)
    i = 0
    while i < len(out):
        dur = float(out[i].end) - float(out[i].start)
        if dur < min_len and len(out) > 1:
            if i == 0:
                j = 1
            elif i == len(out) - 1:
                j = i - 1
            else:
                j = i - 1 if out[i - 1].confidence >= out[i + 1].confidence else i + 1

            if j < i:
                out[j] = ChordSegment(
                    start=float(out[j].start),
                    end=float(out[i].end),
                    label=str(out[j].label),
                    confidence=max(float(out[j].confidence), float(out[i].confidence)),
                )
            else:
                out[j] = ChordSegment(
                    start=float(out[i].start),
                    end=float(out[j].end),
                    label=str(out[j].label),
                    confidence=max(float(out[j].confidence), float(out[i].confidence)),
                )
            out.pop(i)
            i = max(i - 1, 0)
            continue
        i += 1

    return out


def extract_chords_deep(
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
    Chord extraction via madmom deep learning backends.
    Returns: (chroma[12,frames], times[frames] seconds, segments[])
    """
    _ = vocab, switch_penalty, hop_length, beat_times
    chroma, _harm_rms, fps = _deep_chroma(y, sr)
    proc = _get_chord_processor()
    raw = proc(_as_madmom_input(y, sr))

    classes = getattr(proc, "chord_classes", None)
    classes_list = list(classes) if classes is not None else None
    segments_raw = _coerce_segments(raw, fps=float(fps), classes=classes_list)

    segments: list[ChordSegment] = []
    for start, end, label, conf in segments_raw:
        norm = normalize_madmom_chord_label(str(label))
        segments.append(
            ChordSegment(
                start=float(start),
                end=float(end),
                label=norm,
                confidence=float(conf),
            )
        )

    segments = _merge_adjacent_segments(segments)
    smooth_sec = float(getattr(settings, "CHORD_SMOOTHING_SEC", 0.0) or 0.0)
    min_len = max(float(min_segment_sec), float(smooth_sec))
    segments = _smooth_segments(segments, min_len=min_len)

    times = (np.arange(chroma.shape[1], dtype=np.float32) / float(fps)).astype(np.float32)
    return chroma, times, segments
