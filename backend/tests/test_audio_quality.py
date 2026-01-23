import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from app.services.analysis.audio_quality import (
    analyze_audio_characteristics,
    calibrate_thresholds,
    _get_cached_characteristics,
)


class AudioQualityTests(unittest.TestCase):
    def test_analyze_audio_characteristics_outputs(self) -> None:
        sr = 22050
        t = np.linspace(0.0, 1.0, int(sr), endpoint=False)
        y = 0.1 * np.sin(2.0 * np.pi * 440.0 * t)

        with tempfile.TemporaryDirectory() as tmp:
            audio_path = Path(tmp) / "tone.wav"
            sf.write(str(audio_path), y, sr)
            cache_dir = Path(tmp)

            metrics = analyze_audio_characteristics(audio_path, cache_dir=cache_dir)

            self.assertIn("rms_db", metrics)
            self.assertIn("spectral_centroid", metrics)
            self.assertIn("spectral_rolloff", metrics)
            self.assertIn("harmonic_ratio", metrics)
            self.assertIn("onset_density", metrics)
            self.assertIn("noise_floor_db", metrics)
            self.assertGreater(metrics["spectral_centroid"], 0.0)
            self.assertGreaterEqual(metrics["harmonic_ratio"], 0.0)
            self.assertLessEqual(metrics["harmonic_ratio"], 1.0)

            cached = _get_cached_characteristics(audio_path, cache_dir)
            self.assertIsNotNone(cached)

            metrics_cached = analyze_audio_characteristics(audio_path, cache_dir=cache_dir)
            for key in metrics:
                self.assertAlmostEqual(metrics[key], metrics_cached[key], places=6)

    def test_calibrate_thresholds_clean_audio(self) -> None:
        characteristics = {
            "rms_db": -30.0,
            "spectral_centroid": 600.0,
            "spectral_rolloff": 1200.0,
            "harmonic_ratio": 0.85,
            "onset_density": 2.0,
            "noise_floor_db": -55.0,
        }
        onset, frame = calibrate_thresholds(characteristics)
        self.assertLess(onset, 0.5)
        self.assertLess(frame, 0.3)
        self.assertGreaterEqual(onset, 0.25)
        self.assertGreaterEqual(frame, 0.15)

    def test_calibrate_thresholds_noisy_audio(self) -> None:
        characteristics = {
            "rms_db": -8.0,
            "spectral_centroid": 2500.0,
            "spectral_rolloff": 5000.0,
            "harmonic_ratio": 0.25,
            "onset_density": 10.0,
            "noise_floor_db": -30.0,
        }
        onset, frame = calibrate_thresholds(characteristics)
        self.assertGreater(onset, 0.5)
        self.assertGreater(frame, 0.3)
        self.assertLessEqual(onset, 0.75)
        self.assertLessEqual(frame, 0.55)


if __name__ == "__main__":
    unittest.main()
