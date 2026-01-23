from __future__ import annotations

from typing import Iterable

# Standard tuning (string 6 -> 1): E2 A2 D3 G3 B3 E4
STANDARD_TUNING = (40, 45, 50, 55, 59, 64)

TUNINGS: dict[str, tuple[int, ...]] = {
    "standard": STANDARD_TUNING,
    "drop_d": (38, 45, 50, 55, 59, 64),        # D2 A2 D3 G3 B3 E4
    "open_g": (38, 43, 50, 55, 59, 62),        # D2 G2 D3 G3 B3 D4
    "dadgad": (38, 45, 50, 55, 57, 62),        # D2 A2 D3 G3 A3 D4
    "half_step_down": (39, 44, 49, 54, 58, 63) # Eb2 Ab2 Db3 Gb3 Bb3 Eb4
}

MAX_FRET_DEFAULT = 24


def get_tuning(name: str | None) -> tuple[int, ...]:
    if not name:
        return STANDARD_TUNING
    key = str(name).strip().lower()
    return TUNINGS.get(key, STANDARD_TUNING)


def pitch_to_fret_options(
    pitch_midi: int,
    tuning: tuple[int, ...] = STANDARD_TUNING,
    *,
    max_fret: int = MAX_FRET_DEFAULT,
) -> list[tuple[int, int]]:
    """
    Return all valid (string, fret) pairs for a pitch.
    String numbers are 1..6 where 1 is the highest string.
    """
    options: list[tuple[int, int]] = []
    pitch = int(pitch_midi)
    for i, open_pitch in enumerate(tuning):
        fret = pitch - int(open_pitch)
        if 0 <= fret <= int(max_fret):
            string_num = 6 - int(i)
            options.append((int(string_num), int(fret)))
    return options


def positions_to_pitches(
    positions: Iterable[tuple[int, int]],
    tuning: tuple[int, ...] = STANDARD_TUNING,
) -> list[int]:
    """
    Convert (string, fret) positions to MIDI pitches.
    """
    pitches: list[int] = []
    tuning_list = list(tuning)
    for string_num, fret in positions:
        idx = 6 - int(string_num)
        if idx < 0 or idx >= len(tuning_list):
            continue
        pitches.append(int(tuning_list[idx]) + int(fret))
    return pitches
