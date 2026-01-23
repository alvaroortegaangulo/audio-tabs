from __future__ import annotations

from typing import Iterable

from app.services.chords.chord_vocabulary import split_chord_label
from app.services.guitar.fretboard import STANDARD_TUNING, positions_to_pitches

# Open chord shapes use frets for strings 6 -> 1. -1 means muted.
OPEN_POSITION_CHORDS: dict[str, tuple[int, int, int, int, int, int]] = {
    "C:maj": (-1, 3, 2, 0, 1, 0),
    "G:maj": (3, 2, 0, 0, 0, 3),
    "D:maj": (-1, -1, 0, 2, 3, 2),
    "A:maj": (-1, 0, 2, 2, 2, 0),
    "E:maj": (0, 2, 2, 1, 0, 0),
    "A:min": (-1, 0, 2, 2, 1, 0),
    "E:min": (0, 2, 2, 0, 0, 0),
    "D:min": (-1, -1, 0, 2, 3, 1),
    "C:7": (-1, 3, 2, 3, 1, 0),
    "G:7": (3, 2, 0, 0, 0, 1),
    "D:7": (-1, -1, 0, 2, 1, 2),
    "A:7": (-1, 0, 2, 0, 2, 0),
    "E:7": (0, 2, 0, 1, 0, 0),
    "C:maj7": (-1, 3, 2, 0, 0, 0),
    "A:min7": (-1, 0, 2, 0, 1, 0),
    "E:min7": (0, 2, 0, 0, 0, 0),
    "D:min7": (-1, -1, 0, 2, 1, 1),
}


def _shape_positions(shape: tuple[int, int, int, int, int, int]) -> list[tuple[int, int]]:
    positions: list[tuple[int, int]] = []
    for i, fret in enumerate(shape):
        if fret < 0:
            continue
        string_num = 6 - int(i)
        positions.append((int(string_num), int(fret)))
    return positions


def _pitch_classes(pitches: Iterable[int]) -> set[int]:
    return {int(p) % 12 for p in pitches}


def _best_open_chord_for_pitches(
    pitches: list[int],
    chord_label: str,
    tuning: tuple[int, ...],
) -> tuple[str | None, list[tuple[int, int]]]:
    root, quality, _bass = split_chord_label(chord_label)
    key = None
    if root and quality:
        key = f"{root}:{quality}"
        if key not in OPEN_POSITION_CHORDS:
            key = None

    if key is not None:
        positions = _shape_positions(OPEN_POSITION_CHORDS[key])
        return key, positions

    target_pcs = _pitch_classes(pitches)
    if not target_pcs:
        return None, []

    best_key = None
    best_positions: list[tuple[int, int]] = []
    best_score = None
    for cand_key, shape in OPEN_POSITION_CHORDS.items():
        positions = _shape_positions(shape)
        cand_pitches = positions_to_pitches(positions, tuning)
        cand_pcs = _pitch_classes(cand_pitches)
        if not target_pcs.issubset(cand_pcs):
            continue
        score = len(cand_pcs) - len(target_pcs)
        if best_score is None or score < best_score:
            best_key = cand_key
            best_positions = positions
            best_score = score

    if best_key is None:
        return None, []
    return best_key, best_positions


def matches_open_chord(
    pitches: list[int],
    chord_label: str,
    *,
    tuning: tuple[int, ...] = STANDARD_TUNING,
) -> tuple[bool, list[tuple[int, int]]]:
    """
    Return (True, positions) if the pitches can be played as a known open chord.
    Positions are returned in the same order as the input pitches.
    """
    if not pitches:
        return False, []

    _key, chord_positions = _best_open_chord_for_pitches(pitches, chord_label, tuning)
    if not chord_positions:
        return False, []

    chord_pitches = positions_to_pitches(chord_positions, tuning)
    pos_map: dict[int, list[tuple[int, int]]] = {}
    pos_map_pc: dict[int, list[tuple[int, int]]] = {}
    for pos, pitch in zip(chord_positions, chord_pitches):
        pos_map.setdefault(int(pitch), []).append(pos)
        pos_map_pc.setdefault(int(pitch) % 12, []).append(pos)

    positions_out: list[tuple[int, int]] = []
    used_strings: set[int] = set()
    for pitch in pitches:
        options = pos_map.get(int(pitch), [])
        if not options:
            options = pos_map_pc.get(int(pitch) % 12, [])
        picked = None
        for pos in options:
            if int(pos[0]) not in used_strings:
                picked = pos
                break
        if picked is None:
            return False, []
        used_strings.add(int(picked[0]))
        positions_out.append(picked)

    return True, positions_out
