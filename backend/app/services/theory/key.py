from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

import numpy as np
from madmom.features.key import CNNKeyRecognitionProcessor, key_prediction_to_label

Mode = Literal["major", "minor"]


NOTE_NAMES_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
NOTE_NAMES_FLAT = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]

NOTE_TO_PC: dict[str, int] = {
    "C": 0,
    "B#": 0,
    "C#": 1,
    "Db": 1,
    "D": 2,
    "D#": 3,
    "Eb": 3,
    "E": 4,
    "Fb": 4,
    "E#": 5,
    "F": 5,
    "F#": 6,
    "Gb": 6,
    "G": 7,
    "G#": 8,
    "Ab": 8,
    "A": 9,
    "A#": 10,
    "Bb": 10,
    "B": 11,
    "Cb": 11,
}


@dataclass(frozen=True)
class KeyEstimate:
    tonic_pc: int
    tonic: str
    mode: Mode
    fifths: int
    name: str
    vexflow: str
    use_flats: bool
    score: float

    def to_dict(self) -> dict:
        return asdict(self)


def _key_name_and_fifths(pc: int, mode: Mode) -> tuple[str, int]:
    pc = int(pc) % 12

    # Only include musically sensible key signatures within [-7, 7].
    if mode == "major":
        variants: dict[int, list[tuple[str, int]]] = {
            0: [("C", 0)],
            1: [("Db", -5), ("C#", 7)],
            2: [("D", 2)],
            3: [("Eb", -3)],
            4: [("E", 4)],
            5: [("F", -1)],
            6: [("Gb", -6), ("F#", 6)],
            7: [("G", 1)],
            8: [("Ab", -4)],
            9: [("A", 3)],
            10: [("Bb", -2)],
            11: [("B", 5)],
        }
    else:
        variants = {
            9: [("A", 0)],
            4: [("E", 1)],
            11: [("B", 2)],
            6: [("F#", 3)],
            1: [("C#", 4)],
            8: [("G#", 5)],
            3: [("Eb", -6), ("D#", 6)],
            10: [("Bb", -5), ("A#", 7)],
            2: [("D", -1)],
            7: [("G", -2)],
            0: [("C", -3)],
            5: [("F", -4)],
        }

    opts = variants.get(pc, [(NOTE_NAMES_SHARP[pc], 0)])
    # Prefer fewer accidentals; if tie, prefer flats (more common enharmonics).
    tonic, fifths = sorted(opts, key=lambda it: (abs(it[1]), 0 if it[1] < 0 else 1))[0]
    return tonic, int(fifths)


@lru_cache(maxsize=1)
def _get_key_processor() -> CNNKeyRecognitionProcessor:
    return CNNKeyRecognitionProcessor()


def _normalize_tonic(tonic: str) -> str:
    tonic = str(tonic or "").strip()
    if not tonic:
        return tonic
    tonic = tonic.replace("♯", "#").replace("♭", "b")
    return tonic[0].upper() + tonic[1:]


def _parse_key_label(label: str) -> Optional[tuple[str, Mode]]:
    label = str(label or "").strip()
    if not label:
        return None

    if ":" in label:
        tonic, mode = label.split(":", 1)
        mode = mode.strip().lower()
        if mode in ("major", "minor", "maj", "min"):
            mode = "major" if mode in ("major", "maj") else "minor"
            return _normalize_tonic(tonic), mode

    tokens = label.replace("maj", "major").replace("min", "minor").split()
    if len(tokens) >= 2:
        tonic = _normalize_tonic(tokens[0])
        mode = tokens[1].lower()
        if mode in ("major", "minor"):
            return tonic, mode

    return None


def estimate_key_madmom(file_path: Path | str) -> Optional[KeyEstimate]:
    """
    Estimate the global key with Madmom CNN. Returns None if inference fails.
    """
    path = Path(file_path)
    if not path.exists():
        return None

    try:
        processor = _get_key_processor()
        probs = processor(str(path))
    except Exception:
        return None

    probs = np.asarray(probs, dtype=np.float32)
    if probs.ndim > 1:
        probs = np.mean(probs, axis=0)
    if probs.size < 1 or not np.isfinite(probs).all():
        return None

    label = key_prediction_to_label(probs)
    parsed = _parse_key_label(label)
    if parsed is None:
        return None
    tonic_raw, mode = parsed
    tonic_pc = NOTE_TO_PC.get(tonic_raw)
    if tonic_pc is None:
        return None

    tonic, fifths = _key_name_and_fifths(tonic_pc, mode)
    use_flats = fifths < 0
    vexflow = f"{tonic}{'m' if mode == 'minor' else ''}"
    name = f"{tonic} {'minor' if mode == 'minor' else 'major'}"
    score = float(np.max(probs)) if probs.size else 0.0

    return KeyEstimate(
        tonic_pc=int(tonic_pc),
        tonic=tonic,
        mode=mode,
        fifths=int(fifths),
        name=name,
        vexflow=vexflow,
        use_flats=bool(use_flats),
        score=float(score),
    )


def spell_chord_label(label: str, use_flats: bool) -> str:
    """
    Convert chord labels like 'C#:maj' into a preferred enharmonic spelling.
    Only rewrites the root; quality suffix is preserved.
    """
    if not label or label == "N":
        return label

    if ":" in label:
        root, qual = label.split(":", 1)
        qual = qual.strip()
    else:
        root, qual = label, ""

    root = root.strip()
    pc = NOTE_TO_PC.get(root)
    if pc is None:
        # Normalize some common odd encodings and try again.
        root2 = root.replace("♯", "#").replace("♭", "b")
        pc = NOTE_TO_PC.get(root2)
    if pc is None:
        return label

    spelled = NOTE_NAMES_FLAT[int(pc)] if use_flats else NOTE_NAMES_SHARP[int(pc)]
    return f"{spelled}:{qual}" if qual else spelled
