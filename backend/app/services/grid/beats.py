from __future__ import annotations
import numpy as np
import librosa

def estimate_beats_librosa(y: np.ndarray, sr: int) -> tuple[float, np.ndarray]:
    y_h = librosa.effects.harmonic(y)
    tempo, beat_frames = librosa.beat.beat_track(y=y_h, sr=sr, units="frames")
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    return float(tempo), beat_times.astype(np.float32)
