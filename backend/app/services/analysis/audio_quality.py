from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import librosa
import numpy as np

from app.services.analysis.content_classifier import _compute_harmonic_ratio, _compute_onset_density

_LOG = logging.getLogger(__name__)

_ANALYSIS_SR = 22050
_ANALYSIS_MAX_SEC = 60.0
_CACHE_TTL_SEC = 24 * 60 * 60


def _to_db(value: float) -> float:
    return float(20.0 * np.log10(max(float(value), 1e-12)))


def _interp_clamped(x: float, x0: float, x1: float, y0: float, y1: float) -> float:
    if x <= x0:
        return float(y0)
    if x >= x1:
        return float(y1)
    ratio = (float(x) - float(x0)) / (float(x1) - float(x0))
    return float(y0 + ratio * (float(y1) - float(y0)))


def _cache_key(audio_path: Path) -> str:
    mtime = audio_path.stat().st_mtime
    return f"{audio_path.stem}_{hash(mtime)}.json"


def _get_cached_characteristics(audio_path: Path, cache_dir: Path) -> dict[str, float] | None:
    cache_file = cache_dir / "audio_analysis" / _cache_key(audio_path)
    if not cache_file.exists():
        return None
    age = time.time() - cache_file.stat().st_mtime
    if age > float(_CACHE_TTL_SEC):
        return None
    try:
        payload = json.loads(cache_file.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return {str(k): float(v) for k, v in payload.items()}
    except Exception:
        return None
    return None


def _save_cached_characteristics(
    characteristics: dict[str, float],
    audio_path: Path,
    cache_dir: Path,
) -> None:
    cache_root = cache_dir / "audio_analysis"
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_file = cache_root / _cache_key(audio_path)
    cache_file.write_text(json.dumps(characteristics, indent=2), encoding="utf-8")


def analyze_audio_characteristics(
    audio_path: Path,
    *,
    cache_dir: Path | None = None,
) -> dict[str, float]:
    audio_path = Path(audio_path)
    if cache_dir is not None:
        cached = _get_cached_characteristics(audio_path, cache_dir)
        if cached is not None:
            return cached

    y, sr = librosa.load(str(audio_path), sr=int(_ANALYSIS_SR), mono=True)
    if y.size == 0:
        raise ValueError("Audio loaded empty for analysis")

    max_samples = int(float(_ANALYSIS_MAX_SEC) * float(sr))
    if y.size > max_samples:
        y = y[:max_samples]

    rms = librosa.feature.rms(y=y)[0]
    rms_median = float(np.percentile(rms, 50)) if rms.size else 0.0
    noise_rms = float(np.percentile(rms, 10)) if rms.size else 0.0
    rms_db = _to_db(rms_median)
    noise_floor_db = _to_db(noise_rms)

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)
    spectral_centroid = float(np.mean(centroid)) if centroid.size else 0.0
    spectral_rolloff = float(np.mean(rolloff)) if rolloff.size else 0.0

    harmonic_ratio = float(_compute_harmonic_ratio(y, sr))
    onset_density = float(_compute_onset_density(y, sr))

    characteristics = {
        "rms_db": float(rms_db),
        "spectral_centroid": float(spectral_centroid),
        "spectral_rolloff": float(spectral_rolloff),
        "harmonic_ratio": float(harmonic_ratio),
        "onset_density": float(onset_density),
        "noise_floor_db": float(noise_floor_db),
    }

    if cache_dir is not None:
        try:
            _save_cached_characteristics(characteristics, audio_path, cache_dir)
        except Exception as exc:
            _LOG.warning("Failed to save audio analysis cache: %s", exc)

    return characteristics


def calibrate_thresholds(characteristics: dict[str, float]) -> tuple[float, float]:
    onset = 0.5
    frame = 0.3

    rms_db = float(characteristics.get("rms_db", -20.0))
    onset += _interp_clamped(rms_db, -25.0, -12.0, -0.12, 0.10)
    frame += _interp_clamped(rms_db, -25.0, -12.0, -0.10, 0.08)

    harmonic_ratio = float(characteristics.get("harmonic_ratio", 0.55))
    onset += _interp_clamped(harmonic_ratio, 0.4, 0.7, 0.12, -0.08)
    frame += _interp_clamped(harmonic_ratio, 0.4, 0.7, 0.10, -0.06)

    onset_density = float(characteristics.get("onset_density", 5.0))
    onset += _interp_clamped(onset_density, 3.0, 8.0, -0.05, 0.08)

    noise_floor_db = float(characteristics.get("noise_floor_db", -45.0))
    frame += _interp_clamped(noise_floor_db, -50.0, -35.0, -0.08, 0.10)

    onset = max(0.25, min(0.75, float(onset)))
    frame = max(0.15, min(0.55, float(frame)))
    return float(onset), float(frame)
