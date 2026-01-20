from __future__ import annotations
from pathlib import Path
import subprocess

def run_demucs_4stems(input_wav: Path, out_dir: Path, model: str = "htdemucs") -> Path:
    """
    Ejecuta Demucs por CLI. Deja stems en out_dir y devuelve la carpeta final del track.
    Modelos típicos: htdemucs, htdemucs_ft (ver README Demucs v4). :contentReference[oaicite:5]{index=5}
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python", "-m", "demucs",
        "-n", model,
        "-o", str(out_dir),
        str(input_wav),
    ]
    subprocess.run(cmd, check=True)

    # Demucs crea: out_dir/<model>/<trackname>/{bass,drums,other,vocals}.wav
    # Buscamos la carpeta más reciente con stems.
    model_dir = out_dir / model
    if not model_dir.exists():
        # algunos installs usan "separated/<model>/..."
        model_dir = out_dir
    # heurística: primera carpeta que contenga other.wav
    for p in model_dir.rglob("other.wav"):
        return p.parent
    raise FileNotFoundError("No se encontró other.wav tras ejecutar Demucs")
