from __future__ import annotations

from pathlib import Path
from typing import Iterable, TYPE_CHECKING
import os

import numpy as np
import soundfile as sf

# Optional imports for demucs/torch
_DEMUCS_AVAILABLE = False
try:
    import torch
    from demucs.apply import apply_model
    from demucs.audio import AudioFile
    from demucs.pretrained import get_model
    _DEMUCS_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore
    apply_model = None  # type: ignore
    AudioFile = None  # type: ignore
    get_model = None  # type: ignore

# Default model - htdemucs_6s includes dedicated guitar stem for better transcription
_DEFAULT_MODEL = "htdemucs_6s"
_SHIFTS = 2
_OVERLAP = 0.25

# Mapping of known models to their stem names
_MODEL_STEMS: dict[str, list[str]] = {
    "htdemucs": ["drums", "bass", "vocals", "other"],
    "htdemucs_ft": ["drums", "bass", "vocals", "other"],
    "htdemucs_6s": ["drums", "bass", "vocals", "guitar", "piano", "other"],
    "mdx_extra": ["drums", "bass", "vocals", "other"],
    "mdx_extra_q": ["drums", "bass", "vocals", "other"],
}

# Default stem priority for guitar transcription
DEFAULT_STEM_PRIORITY = ("guitar", "other", "vocals")

import logging
_LOG = logging.getLogger(__name__)


def get_model_stems(model_name: str) -> list[str]:
    """Return the list of stems for a given model name."""
    return _MODEL_STEMS.get(model_name, ["drums", "bass", "vocals", "other"])


def _resolve_model(requested: str) -> str:
    """Resolve and validate model name, with fallback to default."""
    req = (requested or "").strip()
    if not req or req.lower() == "auto":
        return _DEFAULT_MODEL

    if req not in _MODEL_STEMS:
        _LOG.warning(
            "Unknown Demucs model '%s', using fallback '%s'. "
            "Known models: %s",
            req, _DEFAULT_MODEL, list(_MODEL_STEMS.keys())
        )
        return _DEFAULT_MODEL

    return req


def _configure_torch_threads() -> int:
    count = int(os.cpu_count() or 1)
    try:
        torch.set_num_threads(count)
    except Exception:
        pass
    try:
        torch.set_num_interop_threads(max(1, count // 2))
    except Exception:
        pass
    return count


def _load_audio(input_wav: Path, model: torch.nn.Module, device: torch.device) -> torch.Tensor:
    samplerate = int(getattr(model, "samplerate", 44100))
    channels = int(getattr(model, "audio_channels", 2))
    audio = AudioFile(str(input_wav))
    wav = audio.read(streams=0, samplerate=samplerate, channels=channels)

    if wav.dim() == 1:
        wav = wav.unsqueeze(0).unsqueeze(0)
    elif wav.dim() == 2:
        wav = wav.unsqueeze(0)
    elif wav.dim() != 3:
        raise ValueError(f"Unexpected audio tensor shape: {wav.shape}")

    return wav.to(device=device, dtype=torch.float32)


def _write_stem(stem_path: Path, audio: np.ndarray, sr: int) -> None:
    stem_path.parent.mkdir(parents=True, exist_ok=True)
    if audio.ndim == 2:
        audio = audio.T
    sf.write(str(stem_path), audio, int(sr), subtype="FLOAT")


def _stems_output_dir(out_dir: Path, model_name: str, input_wav: Path) -> Path:
    return out_dir / model_name / input_wav.stem


def select_stem_path(
    stems_dir: Path,
    preference: Iterable[str] = DEFAULT_STEM_PRIORITY,
) -> Path:
    """
    Select the best available stem for transcription based on priority.

    Args:
        stems_dir: Directory containing stem WAV files
        preference: Ordered list of stem names to try (first available wins)
                   Default: ("guitar", "other", "vocals") for guitar transcription

    Returns:
        Path to the selected stem file
    """
    for stem in preference:
        stem_name = f"{stem}.wav"
        p = stems_dir / stem_name
        if p.exists():
            _LOG.info("Selected stem '%s' for transcription from %s", stem, stems_dir)
            return p

    # Fallback: pick any stem that is not drums or bass
    for p in stems_dir.glob("*.wav"):
        if p.stem not in ("drums", "bass"):
            _LOG.info("Fallback: selected stem '%s' for transcription", p.stem)
            return p

    raise FileNotFoundError(f"No usable stem found for transcription in {stems_dir}")


def get_stem_path(stems_dir: Path, stem: str) -> Path | None:
    stem_name = f"{stem}.wav"
    p = stems_dir / stem_name
    return p if p.exists() else None


def run_demucs(
    input_wav: Path,
    out_dir: Path,
    model: str = _DEFAULT_MODEL,
    *,
    stem_preference: Iterable[str] = DEFAULT_STEM_PRIORITY,
    return_stem: bool = True,
) -> Path:
    """
    Run Demucs stem separation via the Python API.

    Supports both 4-stem models (htdemucs, htdemucs_ft) and 6-stem models
    (htdemucs_6s). The 6-stem model includes dedicated guitar and piano stems.

    Args:
        input_wav: Path to input audio file
        out_dir: Output directory for stems
        model: Demucs model name (default: htdemucs_6s)
        stem_preference: Priority order for stem selection (default: guitar, other, vocals)
        return_stem: If True, return path to selected stem; if False, return stems directory

    Returns:
        Path to selected stem file (if return_stem=True) or stems directory
    """
    if not _DEMUCS_AVAILABLE:
        raise ImportError("torch and demucs are required for stem separation. Install with: pip install torch demucs")

    out_dir.mkdir(parents=True, exist_ok=True)

    model_name = _resolve_model(model)
    _configure_torch_threads()
    device = torch.device("cpu")

    model_obj = get_model(model_name)
    model_obj.to(device)
    model_obj.eval()

    wav = _load_audio(input_wav, model_obj, device)

    with torch.no_grad():
        sources = apply_model(
            model_obj,
            wav,
            shifts=int(_SHIFTS),
            split=True,
            overlap=float(_OVERLAP),
            progress=False,
        )

    if sources.dim() == 3:
        sources = sources.unsqueeze(0)

    if sources.dim() != 4:
        raise ValueError(f"Unexpected sources tensor shape: {sources.shape}")

    sources = sources[0].detach().cpu().numpy().astype(np.float32, copy=False)

    samplerate = int(getattr(model_obj, "samplerate", 44100))
    stems_dir = _stems_output_dir(out_dir, model_name, input_wav)
    stems_dir.mkdir(parents=True, exist_ok=True)

    stem_names = list(getattr(model_obj, "sources", []))
    if len(stem_names) != sources.shape[0]:
        stem_names = [f"stem_{i}" for i in range(sources.shape[0])]

    for idx, stem in enumerate(stem_names):
        stem_path = stems_dir / f"{stem}.wav"
        _write_stem(stem_path, sources[idx], samplerate)

    _LOG.info(
        "Demucs separation complete: model=%s, stems=%s, output=%s",
        model_name, stem_names, stems_dir
    )

    if return_stem:
        return select_stem_path(stems_dir, stem_preference)
    return stems_dir


# Alias for backwards compatibility
run_demucs_4stems = run_demucs
