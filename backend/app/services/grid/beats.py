from __future__ import annotations
import numpy as np
import librosa


def _scipy_hann_compat() -> None:
    """
    Librosa 0.10.x usa scipy.signal.hann; SciPy 1.15 lo eliminó.
    Aplicamos un shim para evitar que el pipeline reviente si el entorno cambia.
    """
    try:
        import scipy.signal  # type: ignore

        if not hasattr(scipy.signal, "hann"):
            from scipy.signal import windows  # type: ignore

            scipy.signal.hann = windows.hann  # type: ignore[attr-defined]
    except Exception:
        return


def estimate_beats_librosa(y: np.ndarray, sr: int) -> tuple[float, np.ndarray]:
    _scipy_hann_compat()
    y_h = librosa.effects.harmonic(y)
    try:
        tempo, beat_frames = librosa.beat.beat_track(y=y_h, sr=sr, units="frames")
    except Exception:
        tempo, beat_frames = 0.0, np.asarray([], dtype=np.int32)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    return float(tempo), beat_times.astype(np.float32)
