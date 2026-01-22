from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Literal

import numpy as np

from app.schemas import KeySignature, ScoreData, ScoreItem, ScoreMeasure, TupletSpec
from app.services.amt.basic_pitch import NoteEvent


VF_NOTE_NAMES_SHARP = ["c", "c#", "d", "d#", "e", "f", "f#", "g", "g#", "a", "a#", "b"]
VF_NOTE_NAMES_FLAT = ["c", "db", "d", "eb", "e", "f", "gb", "g", "ab", "a", "bb", "b"]


def _midi_to_vexflow_key(pitch_midi: int, *, use_flats: bool) -> str:
    pc = int(pitch_midi) % 12
    octave = int(pitch_midi) // 12 - 1
    name = VF_NOTE_NAMES_FLAT[pc] if use_flats else VF_NOTE_NAMES_SHARP[pc]
    return f"{name}/{octave}"


def estimate_key_signature_music21(note_events: list[NoteEvent]) -> KeySignature | None:
    if not note_events:
        return None

    from music21 import note as m21note
    from music21 import stream as m21stream

    # Use a sample of events to keep analysis fast and stable on long files.
    sampled = note_events[:: max(1, len(note_events) // 1500)]

    s = m21stream.Stream()
    for ev in sampled:
        try:
            n = m21note.Note(int(ev.pitch_midi))
            n.quarterLength = 0.25
            s.append(n)
        except Exception:
            continue

    if len(s) == 0:
        return None

    k = s.analyze("key")
    tonic = str(k.tonic.name).replace("-", "b")
    mode = str(getattr(k, "mode", "major"))
    if mode not in ("major", "minor"):
        mode = "major"

    fifths = int(getattr(k, "sharps", 0))
    use_flats = fifths < 0
    vexflow = f"{tonic}{'m' if mode == 'minor' else ''}"
    name = f"{tonic} {'minor' if mode == 'minor' else 'major'}"

    return KeySignature(
        tonic=tonic,
        mode=mode,  # type: ignore[arg-type]
        fifths=fifths,
        name=name,
        vexflow=vexflow,
        use_flats=use_flats,
        score=1.0,
    )


@dataclass(frozen=True)
class QuantizeResult:
    score: ScoreData
    key_signature: KeySignature | None


# Mapping of duration in 16ths (steps) to (duration_str, dots)
_DUR_LOOKUP: dict[int, list[tuple[str, int]]] = {
    1: [("16", 0)],
    2: [("8", 0)],
    3: [("8", 1)],
    4: [("q", 0)],
    6: [("q", 1)],
    7: [("q", 2)], # Double dot is sometimes supported, but let's stick to simple
    8: [("h", 0)],
    12: [("h", 1)],
    16: [("w", 0)],
}

def decompose_steps_straight(steps: int) -> list[tuple[str, int]]:
    """
    Decomposes a duration (in 16th steps) into a list of tied notes.
    """
    if steps <= 0:
        return []

    # Try direct lookup first
    if steps in _DUR_LOOKUP:
        return _DUR_LOOKUP[steps]

    # Otherwise greedy decomposition
    # Standard units in descending order of size (steps)
    units = [
        (16, "w", 0),
        (12, "h", 1),
        (8, "h", 0),
        (6, "q", 1),
        (4, "q", 0),
        (3, "8", 1),
        (2, "8", 0),
        (1, "16", 0)
    ]

    out = []
    rem = steps
    for val, dur, dots in units:
        while rem >= val:
            out.append((dur, dots))
            rem -= val
            if rem == 0:
                break
    return out


def quantize_note_events_to_score(
    note_events: list[NoteEvent],
    *,
    tempo_bpm: float,
    time_signature: str = "4/4",
) -> QuantizeResult:
    tempo = float(tempo_bpm) if tempo_bpm and tempo_bpm > 0 else 120.0
    sec_per_q = 60.0 / tempo

    # Always use 16th note grid (0.25 beats) for stability
    grid_q = 0.25
    grid_kind = "straight"

    # Key detection (music21).
    key_sig = estimate_key_signature_music21(note_events)
    use_flats = bool(key_sig.use_flats) if key_sig else False

    # Time signature parsing (default to 4/4).
    try:
        num_s, den_s = (time_signature or "4/4").split("/")
        num = int(num_s)
        den = int(den_s)
    except Exception:
        num, den = 4, 4

    # Measure length in quarters
    measure_q = float(num) * (4.0 / float(den))

    # Steps per measure (16ths per measure)
    # e.g. 4/4 -> 4 beats -> 16 steps
    steps_per_measure = int(round(measure_q / grid_q))
    if steps_per_measure <= 0:
        steps_per_measure = 16

    # Build boundary maps (step -> pitches starting/ending at that step).
    starts: dict[int, list[int]] = defaultdict(list)
    ends: dict[int, list[int]] = defaultdict(list)
    last_step = 0

    for ev in note_events:
        if ev.end_time_s <= ev.start_time_s:
            continue
        s_q = float(ev.start_time_s) / sec_per_q
        e_q = float(ev.end_time_s) / sec_per_q

        # Snap to grid
        s = int(round(s_q / grid_q))
        e = int(round(e_q / grid_q))
        if e <= s:
            e = s + 1

        pitch = int(ev.pitch_midi)
        starts[s].append(pitch)
        ends[e].append(pitch)
        last_step = max(last_step, e)

    # Make sure we cover full measures
    if last_step % steps_per_measure != 0:
        last_step = ((last_step // steps_per_measure) + 1) * steps_per_measure

    # Sweep steps into compressed chord/rest events.
    active: dict[int, int] = {} # pitch -> count
    events: list[tuple[list[int], int]] = []  # (pitches, duration_steps)

    prev_pitches: list[int] | None = None
    prev_len = 0

    for step in range(0, last_step):
        # Process ends first
        for p in ends.get(step, []):
            active[p] = active.get(p, 0) - 1
            if active[p] <= 0:
                active.pop(p, None)

        # Process starts
        for p in starts.get(step, []):
            active[p] = active.get(p, 0) + 1

        cur = sorted(active.keys())

        # If this is the start
        if prev_pitches is None:
            prev_pitches = cur
            prev_len = 1
            continue

        if cur == prev_pitches:
            prev_len += 1
        else:
            events.append((prev_pitches, prev_len))
            prev_pitches = cur
            prev_len = 1

    if prev_pitches is not None:
        events.append((prev_pitches, prev_len))
    else:
         # Empty score
         pass

    # Split events into measures and emit ScoreData.
    measures: list[ScoreMeasure] = []
    current_measure_items: list[ScoreItem] = []
    measure_number = 1
    remaining_steps = steps_per_measure

    def flush_measure():
        nonlocal current_measure_items, measure_number
        measures.append(ScoreMeasure(number=measure_number, items=current_measure_items))
        current_measure_items = []
        measure_number += 1

    def emit_item(pitches: list[int], duration: str, dots: int, tie: str | None):
        keys = [_midi_to_vexflow_key(p, use_flats=use_flats) for p in pitches] if pitches else []
        current_measure_items.append(
            ScoreItem(
                rest=(len(pitches) == 0),
                keys=keys,
                duration=duration,
                dots=dots,
                tuplet=None, # Tupletes disabled for stability
                tie=tie,  # type: ignore[arg-type]
            )
        )

    for pitches, dur_steps in events:
        steps_left = int(dur_steps)
        while steps_left > 0:
            take = min(steps_left, remaining_steps)

            # Decompose 'take' steps into tied notes
            parts = decompose_steps_straight(take)

            for i, (dur, dots) in enumerate(parts):
                tie = None
                # Determine tie status
                # If we are splitting 'take', we need internal ties
                # AND if steps_left > take (crossing measure), we need to start a tie at the end

                is_start_of_chain = (i == 0)
                is_end_of_chain = (i == len(parts) - 1)

                # Logic:
                # If there are multiple parts, 0->start, mid->continue, last->stop/continue?
                # Wait, 'take' is just the chunk fitting in THIS measure.
                # If steps_left > take, we are crossing a measure, so the last part must TIE OUT.
                # If we came from a previous block (how do we know?), we might need to TIE IN?
                # Ah, 'events' loop iterates through contiguous blocks of same pitch.
                # But we sliced 'events' based on pitch changes.
                # So within one event iteration, it IS a single note (tied).

                # Let's refine tie logic:
                # The total note (dur_steps) is being split into potentially multiple measures (take)
                # and multiple graphic notes (parts).

                # We need to track if we are at the very beginning or very end of the *original* event.
                # We are in a while loop `steps_left > 0`.
                # Total duration is `dur_steps`.
                # Steps processed so far = dur_steps - steps_left.

                steps_processed = dur_steps - steps_left

                # Check if this specific note part is the FIRST of the whole event
                is_absolute_first = (steps_processed == 0) and is_start_of_chain
                # Check if this specific note part is the LAST of the whole event
                is_absolute_last = (steps_left == take) and is_end_of_chain

                if len(pitches) > 0: # Only tie notes, not rests
                    if is_absolute_first and is_absolute_last:
                        tie = None
                    elif is_absolute_first:
                        tie = "start"
                    elif is_absolute_last:
                        tie = "stop"
                    else:
                        tie = "continue"
                else:
                    tie = None

                emit_item(pitches, duration=dur, dots=dots, tie=tie)

            steps_left -= take
            remaining_steps -= take

            if remaining_steps <= 0:
                flush_measure()
                remaining_steps = steps_per_measure

    if current_measure_items:
        flush_measure()

    score = ScoreData(grid_q=float(grid_q), grid_kind="straight", measures=measures)
    return QuantizeResult(score=score, key_signature=key_sig)
