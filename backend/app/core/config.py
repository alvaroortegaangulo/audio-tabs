from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    APP_NAME: str = "chord-extractor"
    APP_ENV: str = "dev"

    DATA_DIR: str = "./data"
    FRONTEND_ORIGIN: str = "http://localhost:3000"
    MAX_UPLOAD_MB: int = 500

    CELERY_ENABLED: bool = True
    REDIS_URL: str = "redis://localhost:6379/0"

    # Demucs stem separation - htdemucs_6s includes dedicated guitar stem
    # htdemucs_6s stems: drums, bass, vocals, guitar, piano, other
    # htdemucs_ft stems: drums, bass, vocals, other (guitar mixed in "other")
    # Note: htdemucs_6s requires ~2x processing time vs htdemucs_ft
    ENABLE_DEMUCS: bool = True
    DEMUCS_MODEL: str = "htdemucs_6s"
    DEMUCS_DEVICE: str = "auto"  # auto|cpu|cuda
    TRANSCRIPTION_STEM_PRIORITY: str = "guitar,other,vocals"  # comma-separated priority list

    ENABLE_BASIC_PITCH: bool = True
    BASIC_PITCH_ONSET_THRESHOLD: float = 0.5
    BASIC_PITCH_FRAME_THRESHOLD: float = 0.3
    BASIC_PITCH_MIN_NOTE_MS: float = 127.70
    BASIC_PITCH_HOP_SEC: float = 0.05
    ENABLE_AUTO_THRESHOLD_CALIBRATION: bool = True

    # Musical post-processing (harmonics/temporal clustering/theory rules)
    HARMONIC_DUPLICATE_WINDOW_MS: float = 100.0
    HARMONIC_TOLERANCE_CENTS: float = 50.0
    HARMONIC_EVEN_THRESHOLD: float = 0.7
    HARMONIC_ODD_THRESHOLD: float = 0.55
    TEMPORAL_CLUSTER_WINDOW_MS: float = 80.0
    TEMPORAL_CLUSTER_GAP_MS: float = 50.0
    DISSONANCE_CORRECTION_AGGRESSIVENESS: float = 0.5  # 0.0 (off) to 1.0 (aggressive)
    DISSONANCE_WINDOW_MS: float = 60.0
    VOICE_ASSIGN_WINDOW_MS: float = 60.0

    # Guitar tuning (standard|drop_d|open_g|dadgad|half_step_down)
    GUITAR_TUNING: str = "standard"

    # Chord detection backend:
    # - "deep": madmom deep learning chord recognition (expanded vocabulary)
    # - "template": legacy template+Viterbi matcher
    CHORD_DETECTION_BACKEND: str = "deep"  # deep|template
    CHORD_SMOOTHING_SEC: float = 0.3

    # Transcription mode:
    # - "guitar": Hybrid mode that analyzes content and applies melodic transcription
    #             for solos/melodies and chord detection for accompaniment sections
    # - "notes": Pure note-by-note transcription (best for melodies/solos only)
    # - "accompaniment": Chord strumming patterns only (best for rhythm guitar only)
    TRANSCRIPTION_MODE: str = "guitar"  # guitar|notes|accompaniment

    # Content analysis settings for guitar mode
    CONTENT_ANALYSIS_WINDOW_SEC: float = 3.0  # Window size for content classification
    CONTENT_ANALYSIS_HOP_SEC: float = 1.5  # Hop between windows (50% overlap)

    # majmin: triads only
    # majmin7: triads + dominant7 + minor7 (no maj7; avoids overfitting to melody tones)
    # majmin7plus: adds maj7 (more complex, more false positives on pop/rock)
    CHORD_VOCAB: str = "majmin7"  # majmin|majmin7|majmin7plus
    SWITCH_PENALTY: float = 2.5
    MIN_SEGMENT_SEC: float = 0.25

    BEAT_BACKEND: str = "librosa"  # librosa|madmom

settings = Settings()
