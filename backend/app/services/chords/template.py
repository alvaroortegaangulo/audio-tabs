from __future__ import annotations
import numpy as np
import librosa
from dataclasses import dataclass
from typing import List, Tuple

NOTE_NAMES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

@dataclass
class Segment:
    start: float
    end: float
    label: str
    confidence: float

def _templates(vocab: str) -> List[Tuple[str, np.ndarray]]:
    """
    Devuelve pares (suffix, template_12).
    """
    # intervalos en semitonos desde la raíz
    if vocab == "majmin":
        types = {
            "maj": [0,4,7],
            "min": [0,3,7],
        }
    else:  # majmin7
        types = {
            "maj": [0,4,7],
            "min": [0,3,7],
            "7":   [0,4,7,10],
        }

    out = []
    for tname, ints in types.items():
        v = np.zeros(12, dtype=np.float32)
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
    # normaliza por si acaso
    T = T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-9)
    return labels, T

def chroma_features(y: np.ndarray, sr: int, hop_length: int = 512) -> np.ndarray:
    """
    Cromagrama robusto (CQT) + normalización frame-wise.
    """
    y_h = librosa.effects.harmonic(y)
    chroma = librosa.feature.chroma_cqt(y=y_h, sr=sr, hop_length=hop_length)
    chroma = chroma.astype(np.float32)
    chroma /= (np.linalg.norm(chroma, axis=0, keepdims=True) + 1e-9)
    return chroma  # [12, frames]

def emission_probs(chroma: np.ndarray, labels: List[str], T: np.ndarray) -> np.ndarray:
    """
    Similaridad coseno (dot) => logits => softmax.
    """
    # scores: [states, frames]
    scores = (T @ chroma).astype(np.float32)

    # Estado "N": penaliza si hay energía armónica
    energy = np.clip(np.mean(chroma, axis=0), 0, 1)
    scores[0, :] = 0.25 - 1.25 * energy

    # softmax estable
    m = np.max(scores, axis=0, keepdims=True)
    ex = np.exp(scores - m)
    probs = ex / (np.sum(ex, axis=0, keepdims=True) + 1e-9)
    return probs.astype(np.float32)  # [states, frames]

def frames_to_segments(best_states: np.ndarray, best_conf: np.ndarray, times: np.ndarray, min_len: float) -> List[Segment]:
    segs: List[Segment] = []
    if len(best_states) == 0:
        return segs

    s0 = int(best_states[0])
    t0 = float(times[0])
    conf_acc = float(best_conf[0])
    n = 1

    for i in range(1, len(best_states)):
        s = int(best_states[i])
        if s != s0:
            t1 = float(times[i])
            conf = conf_acc / max(n, 1)
            if (t1 - t0) >= min_len:
                segs.append(Segment(start=t0, end=t1, label="", confidence=conf))
                segs[-1].label_state = s0  # type: ignore[attr-defined]
            t0 = float(times[i])
            s0 = s
            conf_acc = float(best_conf[i])
            n = 1
        else:
            conf_acc += float(best_conf[i])
            n += 1

    t1 = float(times[-1] + (times[1]-times[0] if len(times) > 1 else 0.02))
    conf = conf_acc / max(n, 1)
    if (t1 - t0) >= min_len:
        segs.append(Segment(start=t0, end=t1, label="", confidence=conf))
        segs[-1].label_state = s0  # type: ignore[attr-defined]

    return segs

def finalize_segments(raw_segs: List[Segment], labels: List[str]) -> List[Segment]:
    out: List[Segment] = []
    for s in raw_segs:
        state = getattr(s, "label_state")
        out.append(Segment(start=s.start, end=s.end, label=labels[int(state)], confidence=float(s.confidence)))
    return out
