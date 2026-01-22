from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from app.services.theory.key import NOTE_TO_PC


STANDARD_TUNING = (40, 45, 50, 55, 59, 64)  # E2 A2 D3 G3 B3 E4


@dataclass(frozen=True)
class Shape:
    frets: tuple[int, int, int, int, int, int]  # string 6 -> 1
    root: str
    quality: str
    label: str

    @property
    def position(self) -> int:
        frets = [f for f in self.frets if f >= 0]
        if not frets:
            return 0
        non_zero = [f for f in frets if f > 0]
        return min(non_zero) if non_zero else 0


_OPEN_SHAPES: dict[tuple[str, str], tuple[int, int, int, int, int, int]] = {
    ("C", "maj"): (-1, 3, 2, 0, 1, 0),
    ("A", "maj"): (-1, 0, 2, 2, 2, 0),
    ("A", "min"): (-1, 0, 2, 2, 1, 0),
    ("D", "maj"): (-1, -1, 0, 2, 3, 2),
    ("D", "min"): (-1, -1, 0, 2, 3, 1),
    ("E", "maj"): (0, 2, 2, 1, 0, 0),
    ("E", "min"): (0, 2, 2, 0, 0, 0),
    ("G", "maj"): (3, 2, 0, 0, 0, 3),
}


def _parse_chord_label(label: str) -> tuple[str | None, str | None]:
    if not label or label == "N":
        return None, None
    if ":" in label:
        root, qual = label.split(":", 1)
        root = root.strip()
        qual = qual.strip().lower() or "maj"
    else:
        root, qual = label.strip(), "maj"
    if not root:
        return None, None
    if qual in ("min", "m", "min7", "m7", "dim"):
        qual = "min"
    else:
        qual = "maj"
    return root, qual


def _transpose_shape(shape: Iterable[int], fret: int) -> tuple[int, int, int, int, int, int]:
    out: list[int] = []
    for f in shape:
        if f < 0:
            out.append(-1)
        elif f == 0:
            out.append(int(fret))
        else:
            out.append(int(f) + int(fret))
    return tuple(out)  # type: ignore[return-value]


def _barre_candidates(root_pc: int, quality: str) -> list[tuple[int, int, int, int, int, int]]:
    out: list[tuple[int, int, int, int, int, int]] = []
    e_pc = NOTE_TO_PC.get("E")
    a_pc = NOTE_TO_PC.get("A")
    if e_pc is None or a_pc is None:
        return out

    e_shape = (0, 2, 2, 1, 0, 0) if quality == "maj" else (0, 2, 2, 0, 0, 0)
    a_shape = (-1, 0, 2, 2, 2, 0) if quality == "maj" else (-1, 0, 2, 2, 1, 0)

    fret_e = (int(root_pc) - int(e_pc)) % 12
    fret_a = (int(root_pc) - int(a_pc)) % 12

    out.append(_transpose_shape(e_shape, fret_e))
    out.append(_transpose_shape(a_shape, fret_a))
    return out


def shape_pitches(shape: Shape) -> list[int]:
    pitches: list[int] = []
    for i, fret in enumerate(shape.frets):
        if fret < 0:
            continue
        pitches.append(int(STANDARD_TUNING[i]) + int(fret))
    return pitches


def shape_positions(shape: Shape) -> list[tuple[int, int]]:
    positions: list[tuple[int, int]] = []
    for i, fret in enumerate(shape.frets):
        if fret < 0:
            continue
        string_num = 6 - int(i)
        positions.append((int(string_num), int(fret)))
    return positions


def pick_shape_for_chord(label: str, prev_shape: Shape | None = None) -> Shape | None:
    root, quality = _parse_chord_label(label)
    if root is None or quality is None:
        return None

    pc = NOTE_TO_PC.get(root)
    if pc is None:
        return None

    candidates: list[Shape] = []
    open_shape = _OPEN_SHAPES.get((root, quality))
    if open_shape is not None:
        candidates.append(Shape(open_shape, root, quality, label))

    for shape in _barre_candidates(int(pc), quality):
        candidates.append(Shape(shape, root, quality, label))

    if not candidates:
        return None

    def cost(shape: Shape) -> float:
        frets = [f for f in shape.frets if f >= 0]
        if not frets:
            return 1e9
        avg_f = float(sum(frets)) / float(len(frets))
        min_f = float(min(frets))
        max_f = float(max(frets))
        span = max_f - min_f
        open_bonus = -0.5 if any(f == 0 for f in frets) else 0.0
        pos = float(shape.position)
        jump = 0.0
        if prev_shape is not None:
            prev_f = [f for f in prev_shape.frets if f >= 0]
            prev_avg = float(sum(prev_f)) / float(len(prev_f)) if prev_f else 0.0
            jump = abs(pos - float(prev_shape.position)) * 0.9 + abs(avg_f - prev_avg) * 0.4
        return avg_f * 0.7 + max_f * 0.25 + span * 0.35 + jump + open_bonus

    best = min(candidates, key=cost)
    return best


def shape_to_dict(shape: Shape) -> dict[str, object]:
    return {
        "frets": list(shape.frets),
        "root": shape.root,
        "quality": shape.quality,
        "label": shape.label,
        "position": int(shape.position),
    }
