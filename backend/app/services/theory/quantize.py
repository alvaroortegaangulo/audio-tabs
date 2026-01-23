from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from app.schemas import KeySignature, ScoreData, ScoreItem, ScoreMeasure, TupletSpec
from app.services.amt.basic_pitch import NoteEvent
from app.core.config import settings
from app.services.guitar.fretboard import get_tuning
from app.services.guitar.optimizer import optimize_tab_positions_for_events


VF_NOTE_NAMES_SHARP = ["c", "c#", "d", "d#", "e", "f", "f#", "g", "g#", "a", "a#", "b"]
VF_NOTE_NAMES_FLAT = ["c", "db", "d", "eb", "e", "f", "gb", "g", "ab", "a", "bb", "b"]


def _midi_to_vexflow_key(pitch_midi: int, *, use_flats: bool) -> str:
    pc = int(pitch_midi) % 12
    octave = int(pitch_midi) // 12 - 1
    name = VF_NOTE_NAMES_FLAT[pc] if use_flats else VF_NOTE_NAMES_SHARP[pc]
    return f"{name}/{octave}"


def _vf_key_to_midi(key: str) -> int | None:
    try:
        note, octave_s = key.split("/")
        note = note.strip().lower()
        octave = int(octave_s)
        if note in VF_NOTE_NAMES_SHARP:
            pc = VF_NOTE_NAMES_SHARP.index(note)
        elif note in VF_NOTE_NAMES_FLAT:
            pc = VF_NOTE_NAMES_FLAT.index(note)
        else:
            return None
        return int((octave + 1) * 12 + int(pc))
    except Exception:
        return None


def estimate_key_signature_music21(note_events: list[NoteEvent]) -> KeySignature | None:
    if not note_events:
        return None

    from music21 import note as m21note
    from music21 import stream as m21stream

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
    pickup_quarters: float = 0.0
    score_m21: object | None = None
    tab_positions: list[list[list[tuple[int, int]]]] | None = None


@dataclass(frozen=True)
class _DurToken:
    duration: str
    dots: int
    ql: float
    tuplet: tuple[int, int] | None


_DUR_TOKENS_STRAIGHT: list[_DurToken] = [
    _DurToken("w", 0, 4.0, None),
    _DurToken("h", 1, 3.0, None),
    _DurToken("h", 0, 2.0, None),
    _DurToken("q", 1, 1.5, None),
    _DurToken("q", 0, 1.0, None),
    _DurToken("8", 1, 0.75, None),
    _DurToken("8", 0, 0.5, None),
    _DurToken("16", 1, 0.375, None),
    _DurToken("16", 0, 0.25, None),
    _DurToken("32", 1, 0.1875, None),
    _DurToken("32", 0, 0.125, None),
]

_DUR_TOKENS_TRIPLET: list[_DurToken] = [
    _DurToken("w", 0, 4.0 * 2.0 / 3.0, (3, 2)),
    _DurToken("h", 0, 2.0 * 2.0 / 3.0, (3, 2)),
    _DurToken("q", 0, 1.0 * 2.0 / 3.0, (3, 2)),
    _DurToken("8", 0, 0.5 * 2.0 / 3.0, (3, 2)),
    _DurToken("16", 0, 0.25 * 2.0 / 3.0, (3, 2)),
    _DurToken("32", 0, 0.125 * 2.0 / 3.0, (3, 2)),
]

_DUR_TOKENS_ALL: list[_DurToken] = sorted(
    _DUR_TOKENS_STRAIGHT + _DUR_TOKENS_TRIPLET,
    key=lambda t: (-t.ql, t.tuplet is not None),
)


def _decompose_duration(duration_q: float) -> list[_DurToken]:
    out: list[_DurToken] = []
    rem = float(duration_q)
    eps = 1e-6

    for token in _DUR_TOKENS_ALL:
        while rem + eps >= token.ql:
            out.append(token)
            rem -= token.ql

    if rem > 1e-3:
        out.append(_DUR_TOKENS_ALL[-1])

    return out


def _parse_time_signature(time_signature: str) -> tuple[int, int]:
    try:
        num_s, den_s = (time_signature or "4/4").split("/")
        num = int(num_s)
        den = int(den_s)
        if num <= 0 or den <= 0:
            raise ValueError
        return num, den
    except Exception:
        return 4, 4


def _duration_to_quarters(item: ScoreItem) -> float:
    base = None
    for token in _DUR_TOKENS_ALL:
        if token.duration == str(item.duration) and int(token.dots) == int(item.dots or 0):
            base = float(token.ql)
            break
    if base is None:
        base = {
            "w": 4.0,
            "h": 2.0,
            "q": 1.0,
            "8": 0.5,
            "16": 0.25,
            "32": 0.125,
        }.get(str(item.duration), 0.0)

    total = float(base)
    dots = int(item.dots or 0)
    for i in range(dots):
        total += float(base) / float(2 ** (i + 1))

    tuplet = getattr(item, "tuplet", None)
    if tuplet is not None:
        num = int(getattr(tuplet, "num_notes", 0) or 0)
        occ = int(getattr(tuplet, "notes_occupied", 0) or 0)
        if num > 0 and occ > 0:
            total *= float(occ) / float(num)

    return float(total)


def _to_beats(times_s: np.ndarray, beat_times: np.ndarray) -> np.ndarray:
    beats = np.asarray(beat_times, dtype=np.float64)
    beats = beats[np.isfinite(beats)]
    beats = np.sort(beats)
    indices = np.arange(len(beats), dtype=np.float64)
    avg_dur = float(np.mean(np.diff(beats))) if len(beats) > 1 else 0.5
    if avg_dur <= 0:
        avg_dur = 0.5
    res = np.interp(times_s, beats, indices, left=-1.0, right=-1.0)

    mask_l = times_s < beats[0]
    if np.any(mask_l):
        res[mask_l] = indices[0] - (beats[0] - times_s[mask_l]) / avg_dur

    mask_r = times_s > beats[-1]
    if np.any(mask_r):
        res[mask_r] = indices[-1] + (times_s[mask_r] - beats[-1]) / avg_dur

    return res


def _beats_to_seconds(beat_pos: float, beat_times: np.ndarray | None, tempo_bpm: float) -> float:
    if beat_times is None or len(beat_times) < 2:
        tempo = float(tempo_bpm) if tempo_bpm and tempo_bpm > 0 else 120.0
        sec_per_q = 60.0 / tempo
        return float(beat_pos) * float(sec_per_q)

    beats = np.asarray(beat_times, dtype=np.float64)
    beats = beats[np.isfinite(beats)]
    if beats.size < 2:
        tempo = float(tempo_bpm) if tempo_bpm and tempo_bpm > 0 else 120.0
        sec_per_q = 60.0 / tempo
        return float(beat_pos) * float(sec_per_q)

    indices = np.arange(len(beats), dtype=np.float64)
    avg_dur = float(np.mean(np.diff(beats))) if len(beats) > 1 else 0.5
    if avg_dur <= 0:
        avg_dur = 0.5
    res = float(np.interp([float(beat_pos)], indices, beats, left=beats[0], right=beats[-1])[0])
    if beat_pos < indices[0]:
        res = float(beats[0]) + float(beat_pos) * float(avg_dur)
    elif beat_pos > indices[-1]:
        res = float(beats[-1]) + (float(beat_pos) - float(indices[-1])) * float(avg_dur)
    return float(res)


def _warp_note_events(
    note_events: list[NoteEvent],
    *,
    tempo_bpm: float,
    beat_times: np.ndarray | None,
) -> tuple[list[NoteEvent], float, float]:
    if not note_events:
        return [], 0.0, 1.0

    if beat_times is not None and len(beat_times) > 1:
        starts = np.array([e.start_time_s for e in note_events], dtype=np.float64)
        ends = np.array([e.end_time_s for e in note_events], dtype=np.float64)
        new_starts = _to_beats(starts, beat_times)
        new_ends = _to_beats(ends, beat_times)
        warped = [
            NoteEvent(
                start_time_s=float(new_starts[i]),
                end_time_s=float(new_ends[i]),
                pitch_midi=int(ev.pitch_midi),
                velocity=int(ev.velocity),
                amplitude=float(ev.amplitude),
            )
            for i, ev in enumerate(note_events)
        ]
        sec_per_q = 1.0
    else:
        tempo = float(tempo_bpm) if tempo_bpm and tempo_bpm > 0 else 120.0
        sec_per_q = 60.0 / tempo
        warped = [
            NoteEvent(
                start_time_s=float(ev.start_time_s) / sec_per_q,
                end_time_s=float(ev.end_time_s) / sec_per_q,
                pitch_midi=int(ev.pitch_midi),
                velocity=int(ev.velocity),
                amplitude=float(ev.amplitude),
            )
            for ev in note_events
        ]

    min_start = min((ev.start_time_s for ev in warped), default=0.0)
    pickup_quarters = max(0.0, -float(min_start))
    if pickup_quarters > 0.0:
        warped = [
            NoteEvent(
                start_time_s=float(ev.start_time_s) + pickup_quarters,
                end_time_s=float(ev.end_time_s) + pickup_quarters,
                pitch_midi=int(ev.pitch_midi),
                velocity=int(ev.velocity),
                amplitude=float(ev.amplitude),
            )
            for ev in warped
        ]

    return warped, pickup_quarters, float(sec_per_q)


def _merge_nearby_note_events(
    note_events: list[NoteEvent],
    *,
    gap_q: float,
) -> list[NoteEvent]:
    by_pitch: dict[int, list[NoteEvent]] = {}
    for ev in note_events:
        by_pitch.setdefault(int(ev.pitch_midi), []).append(ev)

    merged: list[NoteEvent] = []
    gap_q = max(0.0, float(gap_q))
    for pitch, events in by_pitch.items():
        events_sorted = sorted(events, key=lambda e: e.start_time_s)
        cur: NoteEvent | None = None
        for ev in events_sorted:
            if cur is None:
                cur = ev
                continue
            gap = float(ev.start_time_s) - float(cur.end_time_s)
            if gap <= gap_q:
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


def _chordified_sequence(part: object) -> list[tuple[list[int], float]]:
    from music21 import chord as m21chord
    from music21 import note as m21note

    chordified = part.chordify()  # type: ignore[attr-defined]
    elements = list(chordified.recurse().notesAndRests)
    events: list[tuple[float, float, list[int]]] = []
    for el in elements:
        offset = float(getattr(el, "offset", 0.0))
        ql = float(getattr(el.duration, "quarterLength", 0.0))
        if ql <= 1e-6:
            continue
        if isinstance(el, m21note.Rest):
            pitches: list[int] = []
        elif isinstance(el, m21chord.Chord):
            pitches = [int(p.midi) for p in el.pitches]
        elif isinstance(el, m21note.Note):
            pitches = [int(el.pitch.midi)]
        else:
            continue
        events.append((offset, ql, pitches))

    events.sort(key=lambda e: e[0])
    seq: list[tuple[list[int], float]] = []
    eps = 1e-6
    cur = 0.0
    for offset, ql, pitches in events:
        if offset > cur + eps:
            seq.append(([], float(offset - cur)))
            cur = offset
        if offset < cur - eps:
            overlap = cur - offset
            if ql <= overlap + eps:
                continue
            ql = float(ql - overlap)
            offset = cur
        seq.append((pitches, float(ql)))
        cur = offset + ql

    merged: list[tuple[list[int], float]] = []
    for pitches, ql in seq:
        if ql <= 1e-6:
            continue
        if merged and merged[-1][0] == pitches:
            merged[-1] = (pitches, merged[-1][1] + ql)
        else:
            merged.append((pitches, ql))

    return merged


def quantize_note_events_to_score(
    note_events: list[NoteEvent],
    *,
    tempo_bpm: float,
    beat_times: np.ndarray | None = None,
    time_signature: str = "4/4",
    min_grid_q: float = 0.25,
    snap_to_grid: bool = True,
    merge_gap_s: float = 0.02,
) -> QuantizeResult:
    key_sig = estimate_key_signature_music21(note_events)
    use_flats = bool(key_sig.use_flats) if key_sig else False

    warped_events, pickup_quarters, sec_per_q = _warp_note_events(
        note_events,
        tempo_bpm=float(tempo_bpm),
        beat_times=beat_times,
    )

    if not warped_events:
        num, den = _parse_time_signature(time_signature)
        measure_q = float(num) * (4.0 / float(den))
        items = []
        for token in _decompose_duration(measure_q):
            items.append(ScoreItem(rest=True, keys=[], duration=token.duration, dots=token.dots))
        empty = ScoreMeasure(number=1, items=items)
        score = ScoreData(grid_q=1.0, grid_kind="straight", measures=[empty])
        return QuantizeResult(score=score, key_signature=key_sig, pickup_quarters=0.0, score_m21=None, tab_positions=None)

    gap_q = float(merge_gap_s)
    if beat_times is None or len(beat_times) <= 1:
        gap_q = float(merge_gap_s) / float(sec_per_q if sec_per_q > 0 else 1.0)
    warped_events = _merge_nearby_note_events(warped_events, gap_q=gap_q)

    from music21 import meter as m21meter
    from music21 import note as m21note
    from music21 import stream as m21stream

    part = m21stream.Part()
    part.insert(0, m21meter.TimeSignature(time_signature))

    for ev in warped_events:
        if ev.end_time_s <= ev.start_time_s:
            continue
        dur = float(ev.end_time_s) - float(ev.start_time_s)
        if dur <= 0:
            continue
        n = m21note.Note(int(ev.pitch_midi))
        n.duration.quarterLength = dur
        part.insert(float(ev.start_time_s), n)

    if snap_to_grid:
        part.quantize(
            quarterLengthDivisors=(4, 3),
            processOffsets=True,
            processDurations=True,
            inPlace=True,
        )
        try:
            part.makeNotation(inPlace=True)
        except Exception:
            pass

    events_seq = _chordified_sequence(part)

    num, den = _parse_time_signature(time_signature)
    measure_q = float(num) * (4.0 / float(den))
    pickup_quarters = float(pickup_quarters or 0.0)
    remaining_q = pickup_quarters if pickup_quarters > 1e-6 else measure_q

    measures: list[ScoreMeasure] = []
    current_items: list[ScoreItem] = []
    measure_number = 1
    min_token_q = None
    has_tuplet = False
    has_straight = False

    def flush_measure() -> None:
        nonlocal current_items, measure_number
        measures.append(ScoreMeasure(number=measure_number, items=current_items))
        current_items = []
        measure_number += 1

    def emit_item(
        pitches: list[int],
        token: _DurToken,
        tie: str | None,
    ) -> None:
        nonlocal min_token_q, has_tuplet, has_straight
        keys = [_midi_to_vexflow_key(p, use_flats=use_flats) for p in sorted(set(pitches))] if pitches else []
        tuplet_spec = None
        if token.tuplet is not None:
            tuplet_spec = TupletSpec(num_notes=int(token.tuplet[0]), notes_occupied=int(token.tuplet[1]))
            has_tuplet = True
        else:
            has_straight = True
        current_items.append(
            ScoreItem(
                rest=(len(keys) == 0),
                keys=keys,
                duration=token.duration,
                dots=int(token.dots),
                tuplet=tuplet_spec,
                tie=tie,  # type: ignore[arg-type]
            )
        )
        min_token_q = token.ql if min_token_q is None else min(min_token_q, token.ql)

    for pitches, dur_q in events_seq:
        remaining_event = float(dur_q)
        if remaining_event <= 1e-6:
            continue
        is_pitched = len(pitches) > 0
        event_started = False

        while remaining_event > 1e-6:
            take = min(remaining_event, remaining_q)
            tokens = _decompose_duration(take)
            for idx, token in enumerate(tokens):
                is_first = (not event_started) and (idx == 0)
                is_last = (remaining_event - take <= 1e-6) and (idx == len(tokens) - 1)
                tie = None
                if is_pitched and not (is_first and is_last):
                    if is_first:
                        tie = "start"
                    elif is_last:
                        tie = "stop"
                    else:
                        tie = "continue"
                emit_item(pitches, token, tie)
                event_started = True

            remaining_event -= take
            remaining_q -= take
            if remaining_q <= 1e-6:
                flush_measure()
                remaining_q = measure_q

    if current_items:
        flush_measure()

    grid_q = float(min_token_q if min_token_q is not None else 1.0)
    if min_grid_q and min_grid_q > 0:
        grid_q = max(grid_q, float(min_grid_q))
    grid_kind: Literal["straight", "triplet"] = "triplet" if has_tuplet and not has_straight else "straight"

    score = ScoreData(grid_q=float(grid_q), grid_kind=grid_kind, measures=measures)
    score_m21 = m21stream.Score()
    score_m21.insert(0, part)

    tab_positions: list[list[list[tuple[int, int]]]] | None = None
    try:
        tuning = get_tuning(getattr(settings, "GUITAR_TUNING", "standard"))
        events: list[tuple[float, list[int], str | None]] = []
        item_refs: list[tuple[int, int]] = []
        tab_positions = []
        offset_q = 0.0
        for m_idx, meas in enumerate(score.measures):
            measure_positions: list[list[tuple[int, int]]] = []
            for item_idx, item in enumerate(meas.items):
                dur_q = _duration_to_quarters(item)
                if item.rest or not item.keys:
                    measure_positions.append([])
                else:
                    pitches: list[int] = []
                    for key in item.keys:
                        midi = _vf_key_to_midi(str(key))
                        if midi is not None:
                            pitches.append(int(midi))
                    if pitches:
                        t_q = float(offset_q) - float(pickup_quarters or 0.0)
                        t_sec = _beats_to_seconds(t_q, beat_times, float(tempo_bpm))
                        events.append((t_sec, pitches, None))
                        item_refs.append((m_idx, item_idx))
                    measure_positions.append([])
                offset_q += float(dur_q)
            tab_positions.append(measure_positions)

        if events:
            opt_res = optimize_tab_positions_for_events(
                events,
                tuning=tuning,
                tempo_bpm=float(tempo_bpm),
            )
            for ev_idx, (m_idx, item_idx) in enumerate(item_refs):
                if ev_idx >= len(opt_res.events):
                    break
                positions = [(p.string, p.fret) for p in opt_res.events[ev_idx].positions]
                if positions and len(positions) == len(score.measures[m_idx].items[item_idx].keys):
                    tab_positions[m_idx][item_idx] = positions
    except Exception:
        tab_positions = None

    return QuantizeResult(
        score=score,
        key_signature=key_sig,
        pickup_quarters=float(pickup_quarters),
        score_m21=score_m21,
        tab_positions=tab_positions,
    )
