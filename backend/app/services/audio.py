from __future__ import annotations
from pathlib import Path
import subprocess
import soundfile as sf
import numpy as np

def ffmpeg_to_wav_mono_44k(input_path: Path, out_wav: Path) -> None:
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-ac", "1",
        "-ar", "44100",
        str(out_wav),
    ]
    subprocess.run(cmd, check=True, capture_output=True)

def load_wav(path: Path) -> tuple[np.ndarray, int]:
    y, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    return y.astype(np.float32), sr

def peak_normalize(y: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    m = float(np.max(np.abs(y)) + eps)
    return (y / m).astype(np.float32)
