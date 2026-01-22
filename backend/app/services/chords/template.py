from __future__ import annotations
import numpy as np
import librosa
from dataclasses import dataclass
from typing import List, Tuple

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
NON_CHORD_TONE_PENALTY = 0.35
COMPLEXITY_PENALTY = 0.18


@dataclass
class Segment:
    start: float
    end: float
    label: str
    confidence: float


def _templates(vocab: str, *, alpha: float = NON_CHORD_TONE_PENALTY) -> List[Tuple[str, np.ndarray]]:
    """
    Return pairs (suffix, template_12).
    """
    # intervals in semitones from the root
    if vocab == "majmin":
        types = {
            "maj": [0, 4, 7],
            "min": [0, 3, 7],
        }
    elif vocab == "majmin7":
        # Keep the vocabulary conservative: maj7 is a common false-positive when a melody
        # hits the major-7th over a plain major triad.
        types = {
            "maj": [0, 4, 7],
            "min": [0, 3, 7],
            "7": [0, 4, 7, 10],
            "min7": [0, 3, 7, 10],
        }
    elif vocab == "majmin7plus":
        types = {
            "maj": [0, 4, 7],
            "min": [0, 3, 7],
            "7": [0, 4, 7, 10],
            "min7": [0, 3, 7, 10],
            "maj7": [0, 4, 7, 11],
        }
    else:
        types = {
            "maj": [0, 4, 7],
            "min": [0, 3, 7],
            "7": [0, 4, 7, 10],
            "min7": [0, 3, 7, 10],
            "maj7": [0, 4, 7, 11],
        }

    out = []
    for tname, ints in types.items():
        v = np.full(12, -float(alpha), dtype=np.float32)
        for i in ints:
            v[i % 12] = 1.0
        v /= (np.linalg.norm(v) + 1e-9)
        out.append((tname, v))
    return out


def build_chord_library(vocab: str) -> Tuple[List[str], np.ndarray]:
    """
    labels: ["N", "C:maj", ...]
    T: [n_states, 12]
    """
    labels = ["N"]
    rows = [np.zeros(12, dtype=np.float32)]
    rows[0] = rows[0]  # all zeros

    types = _templates(vocab)
    for root in range(12):
        for suffix, base in types:
            tpl = np.roll(base, root)
            labels.append(f"{NOTE_NAMES[root]}:{suffix}")
            rows.append(tpl.astype(np.float32))

    T = np.stack(rows, axis=0)
    # normalize just in case
    T = T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-9)
    return labels, T


def chroma_features(y: np.ndarray, sr: int, hop_length: int = 512) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      chroma_norm: [12, frames] normalized for template matching
      harm_rms: [frames] harmonic energy for the N state
    """
    y_h = librosa.effects.harmonic(y)
    harm_rms = librosa.feature.rms(y=y_h, frame_length=2048, hop_length=hop_length)[0].astype(np.float32)
    harm_rms /= (np.max(harm_rms) + 1e-9)

    chroma_raw = librosa.feature.chroma_cqt(y=y_h, sr=sr, hop_length=hop_length)
    chroma_raw = chroma_raw.astype(np.float32)
    chroma_norm = chroma_raw / (np.linalg.norm(chroma_raw, axis=0, keepdims=True) + 1e-9)
    return chroma_norm, harm_rms  # [12, frames], [frames]


def emission_probs(chroma: np.ndarray, harm_rms: np.ndarray, labels: List[str], T: np.ndarray) -> np.ndarray:
    """
    Cosine similarity => logits => softmax, plus an N-state energy model.
    """
    # scores: [states, frames]
    scores = (T @ chroma).astype(np.float32)

    # Penalize more complex qualities unless evidence is strong.
    if labels:
        penalties = np.zeros((len(labels),), dtype=np.float32)
        for i, lbl in enumerate(labels):
            if ":" not in lbl:
                continue
            qual = lbl.split(":", 1)[1].strip().lower()
            if qual in ("7", "min7", "m7", "maj7"):
                penalties[i] = float(COMPLEXITY_PENALTY)
        if np.any(penalties > 0):
            scores = scores - penalties[:, None]

    harm = np.asarray(harm_rms, dtype=np.float32).reshape(-1) if harm_rms is not None else None
    if harm is None or harm.shape[0] != chroma.shape[1]:
        energy = np.clip(np.mean(chroma, axis=0), 0.0, 1.0)
    else:
        energy = np.clip(harm, 0.0, 1.0)

    bias = 2.0
    slope = 6.0
    scores[0, :] = float(bias) - float(slope) * energy

    # stable softmax
    m = np.max(scores, axis=0, keepdims=True)
    ex = np.exp(scores - m)
    probs = ex / (np.sum(ex, axis=0, keepdims=True) + 1e-9)
    return probs.astype(np.float32)  # [states, frames]


def frames_to_segments(best_states: np.ndarray, best_conf: np.ndarray, times: np.ndarray, min_len: float) -> List[Segment]:
    segs: List[Segment] = []
    if len(best_states) == 0:
        return segs

    spans: List[Tuple[int, int]] = []
    start = 0
    for i in range(1, len(best_states)):
        if int(best_states[i]) != int(best_states[i - 1]):
            spans.append((start, i))
            start = i
    spans.append((start, len(best_states)))

    step = float(times[1] - times[0]) if len(times) > 1 else 0.02
    out: List[Segment] = []
    for a, b in spans:
        t0 = float(times[a])
        t1 = float(times[b - 1] + step)
        conf = float(np.mean(best_conf[a:b])) if b > a else float(best_conf[a])
        seg = Segment(start=t0, end=t1, label="", confidence=conf)
        seg.label_state = int(best_states[a])  # type: ignore[attr-defined]
        out.append(seg)

    if min_len <= 0:
        return out

    i = 0
    while i < len(out):
        dur = float(out[i].end - out[i].start)
        if dur < min_len and len(out) > 1:
            if i == 0:
                j = 1
            elif i == len(out) - 1:
                j = i - 1
            else:
                j = i - 1 if out[i - 1].confidence >= out[i + 1].confidence else i + 1

            if j < i:
                out[j].end = out[i].end
            else:
                out[j].start = out[i].start
            out[j].confidence = float(max(out[j].confidence, out[i].confidence))
            out.pop(i)
            i = max(i - 1, 0)
            continue
        i += 1

    return out


def finalize_segments(raw_segs: List[Segment], labels: List[str]) -> List[Segment]:
    out: List[Segment] = []
    for s in raw_segs:
        state = getattr(s, "label_state")
        out.append(Segment(start=s.start, end=s.end, label=labels[int(state)], confidence=float(s.confidence)))
    return out
