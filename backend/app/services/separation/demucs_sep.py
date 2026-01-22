from __future__ import annotations

from pathlib import Path
from typing import Iterable
import os

import numpy as np
import soundfile as sf
import torch
from demucs.apply import apply_model
from demucs.audio import AudioFile
from demucs.pretrained import get_model

_DEFAULT_MODEL = "htdemucs_ft"
_SHIFTS = 2
_OVERLAP = 0.25


def _resolve_model(requested: str) -> str:
    req = (requested or "").strip()
    if not req or req.lower() == "auto":
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


def select_stem_path(stems_dir: Path, preference: Iterable[str] = ("other", "vocals")) -> Path:
    for stem in preference:
        stem_name = f"{stem}.wav"
        p = stems_dir / stem_name
        if p.exists():
            return p

    # Fallback: pick any stem that is not drums or bass.
    for p in stems_dir.glob("*.wav"):
        if p.stem not in ("drums", "bass"):
            return p
    raise FileNotFoundError("No usable stem found for transcription")


def get_stem_path(stems_dir: Path, stem: str) -> Path | None:
    stem_name = f"{stem}.wav"
    p = stems_dir / stem_name
    return p if p.exists() else None


def run_demucs_4stems(
    input_wav: Path,
    out_dir: Path,
    model: str = _DEFAULT_MODEL,
    *,
    stem_preference: Iterable[str] = ("other", "vocals"),
    return_stem: bool = True,
) -> Path:
    """
    Run Demucs via the Python API. Outputs stems to out_dir and returns the
    selected stem (default: other/vocals) to avoid transcribing drums/bass.
    """
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

    if return_stem:
        return select_stem_path(stems_dir, stem_preference)
    return stems_dir
