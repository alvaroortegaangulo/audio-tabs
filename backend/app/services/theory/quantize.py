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


def _choose_grid_q(onsets_q: np.ndarray) -> tuple[float, Literal["straight", "triplet"]]:
    if onsets_q.size == 0:
        return 0.25, "straight"

    candidates: list[tuple[float, Literal["straight", "triplet"], float]] = [
        (0.25, "straight", 1.0),  # 16th
        (1.0 / 3.0, "triplet", 1.25),  # 8th-triplet grid (penalize slightly)
    ]

    best = None
    for grid, kind, penalty in candidates:
        qn = np.round(onsets_q / grid) * grid
        err = float(np.mean(np.abs(onsets_q - qn)))
        cost = err * penalty
        if best is None or cost < best[0]:
            best = (cost, grid, kind)

    assert best is not None
    return float(best[1]), best[2]


def _snap_note_events_to_grid(
    note_events: list[NoteEvent],
    *,
    sec_per_q: float,
    grid_q: float,
) -> list[NoteEvent]:
    step_s = float(sec_per_q) * float(grid_q)
    if step_s <= 0:
        return list(note_events)

    out: list[NoteEvent] = []
    for ev in note_events:
        if ev.end_time_s <= ev.start_time_s:
            continue
        s_step = int(round(float(ev.start_time_s) / step_s))
        e_step = int(round(float(ev.end_time_s) / step_s))
        if e_step <= s_step:
            e_step = s_step + 1
        start = float(s_step) * step_s
        end = float(e_step) * step_s
        out.append(
            NoteEvent(
                start_time_s=start,
                end_time_s=end,
                pitch_midi=int(ev.pitch_midi),
                velocity=int(ev.velocity),
                amplitude=float(ev.amplitude),
            )
        )

    return sorted(out, key=lambda e: e.start_time_s)


def _merge_nearby_note_events(
    note_events: list[NoteEvent],
    *,
    gap_s: float,
) -> list[NoteEvent]:
    by_pitch: dict[int, list[NoteEvent]] = {}
    for ev in note_events:
        by_pitch.setdefault(int(ev.pitch_midi), []).append(ev)

    merged: list[NoteEvent] = []
    gap_s = max(0.0, float(gap_s))
    for pitch, events in by_pitch.items():
        events_sorted = sorted(events, key=lambda e: e.start_time_s)
        cur: NoteEvent | None = None
        for ev in events_sorted:
            if cur is None:
                cur = ev
                continue
            gap = float(ev.start_time_s) - float(cur.end_time_s)
            if gap <= gap_s:
                end_time = max(float(cur.end_time_s), float(ev.end_time_s))
                amp = max(float(cur.amplitude), float(ev.amplitude))
                vel = max(int(cur.velocity), int(ev.velocity))
                cur = NoteEvent(
                    start_time_s=float(cur.start_time_s),
                    end_time_s=float(end_time),
                    pitch_midi=int(pitch),
                    velocity=int(vel),
                    amplitude=float(amp),
                )
            else:
                merged.append(cur)
                cur = ev
        if cur is not None:
            merged.append(cur)

    return sorted(merged, key=lambda e: e.start_time_s)


_DUR_TOKENS_STRAIGHT: list[tuple[str, int, float]] = [
    ("w", 0, 4.0),
    ("h", 1, 3.0),
    ("h", 0, 2.0),
    ("q", 1, 1.5),
    ("q", 0, 1.0),
    ("8", 1, 0.75),
    ("8", 0, 0.5),
    ("16", 1, 0.375),
    ("16", 0, 0.25),
    ("32", 1, 0.1875),
    ("32", 0, 0.125),
]


def _decompose_duration_straight(duration_q: float) -> list[tuple[str, int, float]]:
    out: list[tuple[str, int, float]] = []
    rem = float(duration_q)
    eps = 1e-6

    for dur, dots, ql in _DUR_TOKENS_STRAIGHT:
        while rem + eps >= ql:
            out.append((dur, dots, ql))
            rem -= ql

    if rem > 1e-3:
        # Fallback: force a 32nd to avoid dropping time.
        out.append(("32", 0, max(0.125, rem)))

    return out


def quantize_note_events_to_score(
    note_events: list[NoteEvent],
    *,
    tempo_bpm: float,
    time_signature: str = "4/4",
    min_grid_q: float = 0.25,
    snap_to_grid: bool = True,
    merge_gap_s: float = 0.02,
) -> QuantizeResult:
    tempo = float(tempo_bpm) if tempo_bpm and tempo_bpm > 0 else 120.0
    sec_per_q = 60.0 / tempo

    # Convert onsets to quarter units for grid selection.
    onsets_q = np.asarray([ev.start_time_s / sec_per_q for ev in note_events], dtype=np.float32)
    grid_q, grid_kind = _choose_grid_q(onsets_q)
    min_grid_q = float(min_grid_q)
    if min_grid_q > 0:
        grid_q = max(float(grid_q), min_grid_q)

    if snap_to_grid:
        note_events = _snap_note_events_to_grid(note_events, sec_per_q=sec_per_q, grid_q=grid_q)
        note_events = _merge_nearby_note_events(note_events, gap_s=merge_gap_s)

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

    measure_q = float(num) * (4.0 / float(den))
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

        s = int(round(s_q / grid_q))
        e = int(round(e_q / grid_q))
        if e <= s:
            e = s + 1

        pitch = int(ev.pitch_midi)
        starts[s].append(pitch)
        ends[e].append(pitch)
        last_step = max(last_step, e)

    # Sweep steps into compressed chord/rest events.
    active: dict[int, int] = {}
    events: list[tuple[list[int], int]] = []  # (pitches, duration_steps)
    prev_pitches: list[int] | None = None
    prev_len = 0

    for step in range(0, last_step):
        for p in ends.get(step, []):
            active[p] = active.get(p, 0) - 1
            if active[p] <= 0:
                active.pop(p, None)
        for p in starts.get(step, []):
            active[p] = active.get(p, 0) + 1

        cur = sorted(active.keys())
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

    if prev_pitches is None:
        events = [([], int(steps_per_measure))]
    else:
        events.append((prev_pitches, prev_len))

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

    def emit_item(pitches: list[int], duration: str, dots: int, tuplet: TupletSpec | None, tie: str | None):
        keys = [_midi_to_vexflow_key(p, use_flats=use_flats) for p in pitches] if pitches else []
        current_measure_items.append(
            ScoreItem(
                rest=(len(pitches) == 0),
                keys=keys,
                duration=duration,
                dots=dots,
                tuplet=tuplet,
                tie=tie,  # type: ignore[arg-type]
            )
        )

    for pitches, dur_steps in events:
        steps_left = int(dur_steps)
        while steps_left > 0:
            take = min(steps_left, remaining_steps)

            dur_q = take * grid_q
            if grid_kind == "triplet":
                # Represent as 8th-notes in 3:2 tuplets (each = 1/3 quarter).
                unit_steps = int(take)
                tuplet = TupletSpec(num_notes=3, notes_occupied=2)
                for i in range(unit_steps):
                    tie = None
                    if unit_steps > 1:
                        if i == 0:
                            tie = "start"
                        elif i == unit_steps - 1:
                            tie = "stop"
                        else:
                            tie = "continue"
                    emit_item(pitches, duration="8", dots=0, tuplet=tuplet, tie=tie)
            else:
                parts = _decompose_duration_straight(dur_q)
                for i, (dur, dots, _ql) in enumerate(parts):
                    tie = None
                    if len(parts) > 1:
                        if i == 0:
                            tie = "start"
                        elif i == len(parts) - 1:
                            tie = "stop"
                        else:
                            tie = "continue"
                    emit_item(pitches, duration=dur, dots=dots, tuplet=None, tie=tie)

            steps_left -= take
            remaining_steps -= take

            if remaining_steps <= 0:
                flush_measure()
                remaining_steps = steps_per_measure

    if current_measure_items:
        flush_measure()

    score = ScoreData(grid_q=float(grid_q), grid_kind=grid_kind, measures=measures)
    return QuantizeResult(score=score, key_signature=key_sig)
