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

    ENABLE_DEMUCS: bool = False
    DEMUCS_MODEL: str = "htdemucs"
    DEMUCS_DEVICE: str = "auto"  # auto|cpu|cuda

    ENABLE_BASIC_PITCH: bool = True
    BASIC_PITCH_ONSET_THRESHOLD: float = 0.5
    BASIC_PITCH_FRAME_THRESHOLD: float = 0.3
    BASIC_PITCH_MIN_NOTE_MS: float = 127.70
    BASIC_PITCH_HOP_SEC: float = 0.05

    CHORD_VOCAB: str = "majmin7"  # majmin|majmin7
    SWITCH_PENALTY: float = 2.5
    MIN_SEGMENT_SEC: float = 0.25

    BEAT_BACKEND: str = "librosa"  # librosa|madmom

settings = Settings()
