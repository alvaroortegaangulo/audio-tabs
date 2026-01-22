from __future__ import annotations

from typing import Iterable

import librosa
import numpy as np


def _beat_mapping(beat_times: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    beats = np.asarray(beat_times, dtype=np.float64)
    beats = beats[np.isfinite(beats)]
    beats = np.sort(beats)
    indices = np.arange(len(beats), dtype=np.float64)
    avg_dur = float(np.mean(np.diff(beats))) if len(beats) > 1 else 0.5
    if avg_dur <= 0:
        avg_dur = 0.5
    return beats, indices, avg_dur


def _to_beats(times_s: np.ndarray, beat_times: np.ndarray) -> np.ndarray:
    beats, indices, avg_dur = _beat_mapping(beat_times)
    res = np.interp(times_s, beats, indices, left=-1.0, right=-1.0)

    mask_l = times_s < beats[0]
    if np.any(mask_l):
        res[mask_l] = indices[0] - (beats[0] - times_s[mask_l]) / avg_dur

    mask_r = times_s > beats[-1]
    if np.any(mask_r):
        res[mask_r] = indices[-1] + (times_s[mask_r] - beats[-1]) / avg_dur

    return res


def _from_beats(beats_idx: np.ndarray, beat_times: np.ndarray) -> np.ndarray:
    beats, indices, avg_dur = _beat_mapping(beat_times)
    res = np.interp(beats_idx, indices, beats, left=beats[0], right=beats[-1])

    mask_l = beats_idx < indices[0]
    if np.any(mask_l):
        res[mask_l] = beats[0] + beats_idx[mask_l] * avg_dur

    mask_r = beats_idx > indices[-1]
    if np.any(mask_r):
        res[mask_r] = beats[-1] + (beats_idx[mask_r] - indices[-1]) * avg_dur

    return res


def _choose_grid(positions: np.ndarray) -> float:
    if positions.size == 0:
        return 0.5

    candidates = [
        (0.25, 1.1),
        (0.5, 1.0),
        (1.0, 1.05),
    ]
    best = None
    for grid, penalty in candidates:
        q = np.round(positions / grid) * grid
        err = float(np.mean(np.abs(positions - q)))
        cost = err * penalty
        if best is None or cost < best[0]:
            best = (cost, grid)
    assert best is not None
    return float(best[1])


def _quantize_onsets(
    onsets_s: np.ndarray,
    *,
    beat_times: np.ndarray | None,
    tempo_bpm: float | None,
) -> np.ndarray:
    if onsets_s.size == 0:
        return onsets_s

    if beat_times is not None and len(beat_times) > 1:
        beat_pos = _to_beats(onsets_s, beat_times)
        grid = _choose_grid(beat_pos)
        beat_q = np.round(beat_pos / grid) * grid
        return _from_beats(beat_q, beat_times)

    tempo = float(tempo_bpm or 0.0)
    if tempo <= 0:
        return onsets_s

    sec_per_q = 60.0 / tempo
    positions = onsets_s / sec_per_q
    grid = _choose_grid(positions)
    q = np.round(positions / grid) * grid
    return q * sec_per_q


def detect_strum_onsets(
    y: np.ndarray,
    sr: int,
    *,
    beat_times: Iterable[float] | None = None,
    tempo_bpm: float | None = None,
    min_interval_s: float = 0.12,
    onset_delta: float = 0.2,
    backtrack: bool = False,
) -> np.ndarray:
    """
    Detect strum onsets in seconds. If beat_times are provided, onsets are
    quantized to a beat grid for cleaner rhythms.
    """
    y = np.asarray(y, dtype=np.float32)
    if y.size == 0:
        return np.asarray([], dtype=np.float32)

    onset_env = librosa.onset.onset_strength(y=y, sr=int(sr), aggregate=np.median)
    onset_env = librosa.util.normalize(onset_env)

    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=int(sr),
        units="frames",
        delta=float(onset_delta),
        backtrack=bool(backtrack),
    )
    if onset_frames.size == 0:
        return np.asarray([], dtype=np.float32)

    strengths = onset_env[np.clip(onset_frames, 0, len(onset_env) - 1)]
    thr = float(max(0.1, np.percentile(strengths, 40)))
    keep = strengths >= thr
    onset_frames = onset_frames[keep]
    strengths = strengths[keep]

    if onset_frames.size == 0:
        return np.asarray([], dtype=np.float32)

    onset_times = librosa.frames_to_time(onset_frames, sr=int(sr)).astype(np.float32)
    # Enforce minimum spacing by keeping the stronger onset.
    ordered = sorted(zip(onset_times, strengths), key=lambda t: float(t[0]))
    filtered: list[float] = []
    last_t = None
    last_s = None
    for t, s in ordered:
        if last_t is None or (t - last_t) >= float(min_interval_s):
            filtered.append(float(t))
            last_t = float(t)
            last_s = float(s)
        else:
            if float(s) > float(last_s or 0.0):
                filtered[-1] = float(t)
                last_t = float(t)
                last_s = float(s)

    onsets = np.asarray(filtered, dtype=np.float32)
    bt = np.asarray(list(beat_times), dtype=np.float32) if beat_times is not None else None
    onsets = _quantize_onsets(onsets, beat_times=bt, tempo_bpm=tempo_bpm)

    # Deduplicate near-identical onsets after quantization.
    onsets = np.sort(onsets)
    unique: list[float] = []
    for t in onsets:
        if not unique or float(t) - float(unique[-1]) > 1e-3:
            unique.append(float(t))

    return np.asarray(unique, dtype=np.float32)
