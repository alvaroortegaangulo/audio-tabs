from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Iterable


def _list_demucs_models() -> set[str]:
    try:
        res = subprocess.run(
            ["python", "-m", "demucs", "--list-models"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return set()

    models: set[str] = set()
    for line in res.stdout.splitlines():
        name = line.strip()
        if name:
            models.add(name)
    return models


def _resolve_model(requested: str) -> str:
    req = (requested or "").strip()
    if not req or req == "auto":
        available = _list_demucs_models()
        if "htdemucs_ft" in available:
            return "htdemucs_ft"
        return "htdemucs"
    if req == "htdemucs_ft":
        available = _list_demucs_models()
        if "htdemucs_ft" in available:
            return req
        return "htdemucs"
    return req


def _find_stems_dir(out_dir: Path, model: str) -> Path:
    model_dir = out_dir / model
    if not model_dir.exists():
        model_dir = out_dir

    for stem in ("other.wav", "vocals.wav"):
        for p in model_dir.rglob(stem):
            return p.parent
    raise FileNotFoundError("No stems found after running Demucs")


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


def run_demucs_4stems(
    input_wav: Path,
    out_dir: Path,
    model: str = "auto",
    *,
    stem_preference: Iterable[str] = ("other", "vocals"),
    return_stem: bool = True,
) -> Path:
    """
    Run Demucs via CLI. Outputs stems to out_dir and returns the selected stem
    (default: other/vocals) to avoid transcribing drums/bass.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    model = _resolve_model(model)
    cmd = [
        "python", "-m", "demucs",
        "-n", model,
        "-o", str(out_dir),
        str(input_wav),
    ]
    subprocess.run(cmd, check=True)

    # Demucs creates: out_dir/<model>/<trackname>/{bass,drums,other,vocals}.wav
    stems_dir = _find_stems_dir(out_dir, model)
    if return_stem:
        return select_stem_path(stems_dir, stem_preference)
    return stems_dir
