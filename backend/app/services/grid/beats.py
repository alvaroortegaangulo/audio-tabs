from __future__ import annotations

from pathlib import Path
import logging
import numpy as np

_FPS = 100
_BEATS_PER_BAR = [3, 4]

_LOG = logging.getLogger(__name__)


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
        raise RuntimeError("madmom is required for beat tracking.") from exc

    arr = np.asarray(y_or_path, dtype=np.float32)
    if arr.ndim == 1:
        return Signal(arr, sample_rate=int(sr), num_channels=1)
    if arr.ndim == 2:
        return Signal(arr, sample_rate=int(sr))
    raise ValueError(f"Unsupported audio shape: {arr.shape}")


def _estimate_tempo(beat_times: np.ndarray) -> float:
    if beat_times.size < 2:
        return 0.0
    diffs = np.diff(beat_times)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size == 0:
        return 0.0
    return float(60.0 / float(np.mean(diffs)))


def _infer_meter(beat_positions: np.ndarray) -> str | None:
    if beat_positions.size == 0:
        return None
    beat_positions = beat_positions[np.isfinite(beat_positions)]
    if beat_positions.size == 0:
        return None
    count4 = int(np.sum(beat_positions == 4))
    count3 = int(np.sum(beat_positions == 3))
    if count4 > 0 and count4 >= max(1, count3 // 2):
        return "4/4"
    if count3 > 0:
        return "3/4"
    return None


def estimate_beats_librosa(
    y: np.ndarray | str | Path,
    sr: int | None = None,
    *,
    use_harmonic: bool = True,
) -> tuple[float, np.ndarray]:
    """
    Beat and downbeat tracking via madmom RNN + DBN.
    use_harmonic is kept for compatibility but ignored.
    """
    try:
        from madmom.features.beats import RNNDownBeatProcessor, DBNDownBeatTrackingProcessor
    except Exception as exc:
        raise RuntimeError("madmom is required for beat tracking.") from exc

    if use_harmonic:
        # Madmom does its own internal preprocessing; ignore this flag.
        pass

    input_data = _as_madmom_input(y, sr)
    rnn = RNNDownBeatProcessor(fps=_FPS)
    activations = rnn(input_data)

    dbn = DBNDownBeatTrackingProcessor(
        beats_per_bar=_BEATS_PER_BAR,
        fps=_FPS,
    )
    beats = dbn(activations)

    if beats is None or len(beats) == 0:
        return 0.0, np.asarray([], dtype=np.float32)

    beats = np.asarray(beats, dtype=np.float32)
    if beats.ndim == 1:
        beat_times = beats
        beat_positions = np.asarray([], dtype=np.float32)
    else:
        beat_times = beats[:, 0]
        beat_positions = beats[:, 1]

    tempo = _estimate_tempo(beat_times)
    meter = _infer_meter(beat_positions)
    if meter:
        _LOG.info("Detected meter: %s", meter)

    return float(tempo), beat_times.astype(np.float32)


def normalize_beat_times(beat_times: np.ndarray | None) -> tuple[np.ndarray | None, float]:
    if beat_times is None:
        return None, 0.0
    bt = np.asarray(beat_times, dtype=np.float32)
    bt = bt[np.isfinite(bt)]
    if bt.size == 0:
        return None, 0.0
    bt = np.sort(bt)
    offset = float(bt[0])
    return (bt - offset).astype(np.float32), offset
