from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, Optional

import numpy as np

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


_KRUMHANSL_MAJOR = np.asarray(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
    dtype=np.float32,
)
_KRUMHANSL_MINOR = np.asarray(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17],
    dtype=np.float32,
)


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    return v / (float(np.linalg.norm(v)) + 1e-9)


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
    # prefer fewer accidentals; if tie, prefer flats (more common enharmonics)
    tonic, fifths = sorted(opts, key=lambda it: (abs(it[1]), 0 if it[1] < 0 else 1))[0]
    return tonic, int(fifths)


def estimate_key_from_chroma(chroma: np.ndarray) -> Optional[KeyEstimate]:
    """
    Krumhansl-Schmuckler style key estimation from a chroma matrix [12, frames].

    Returns None if chroma is empty/invalid.
    """
    if chroma is None:
        return None
    chroma = np.asarray(chroma, dtype=np.float32)
    if chroma.ndim != 2 or chroma.shape[0] != 12 or chroma.shape[1] < 1:
        return None

    profile = chroma.mean(axis=1)
    if not np.isfinite(profile).all() or float(np.sum(profile)) <= 1e-9:
        return None

    profile_u = _unit(profile)
    major_u = _unit(_KRUMHANSL_MAJOR)
    minor_u = _unit(_KRUMHANSL_MINOR)

    major_scores = np.asarray([float(np.dot(profile_u, np.roll(major_u, k))) for k in range(12)], dtype=np.float32)
    minor_scores = np.asarray([float(np.dot(profile_u, np.roll(minor_u, k))) for k in range(12)], dtype=np.float32)

    major_k = int(np.argmax(major_scores))
    minor_k = int(np.argmax(minor_scores))
    major_best = float(major_scores[major_k])
    minor_best = float(minor_scores[minor_k])

    if major_best >= minor_best:
        tonic_pc = major_k
        mode: Mode = "major"
        score = major_best
    else:
        tonic_pc = minor_k
        mode = "minor"
        score = minor_best

    tonic, fifths = _key_name_and_fifths(tonic_pc, mode)
    use_flats = fifths < 0
    vexflow = f"{tonic}{'m' if mode == 'minor' else ''}"
    name = f"{tonic} {'minor' if mode == 'minor' else 'major'}"

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
        # normalize some common odd encodings and try again
        root2 = root.replace("♯", "#").replace("♭", "b")
        pc = NOTE_TO_PC.get(root2)
    if pc is None:
        return label

    spelled = NOTE_NAMES_FLAT[int(pc)] if use_flats else NOTE_NAMES_SHARP[int(pc)]
    return f"{spelled}:{qual}" if qual else spelled

