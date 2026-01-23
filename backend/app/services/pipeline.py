from __future__ import annotations

import json
import logging
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import librosa
import numpy as np
import soundfile as sf

from app.core.config import settings
from app.schemas import JobResult, KeySignature, ChordSegment, ScoreData, ScoreItem, ScoreMeasure
from app.services.audio import ffmpeg_to_wav_mono_44k, load_wav, peak_normalize
from app.services.chords.extract import extract_chords_template
from app.services.grid.beats import estimate_beats_librosa, normalize_beat_times
from app.services.engraving.lilypond import build_lilypond_score, render_lilypond_pdf
from app.services.musicxml.lead_sheet import export_lead_sheet_musicxml
from app.services.theory.key import estimate_key_madmom, spell_chord_label, NOTE_TO_PC

# Transcription & Export Imports
from app.services.amt.basic_pitch import transcribe_basic_pitch, NoteEvent, save_note_events_csv
from app.services.theory.quantize import quantize_note_events_to_score
from app.services.theory.musical_postprocessor import (
    remove_harmonic_duplicates,
    merge_temporal_clusters,
    apply_music_theory_rules,
)

# Optional demucs import
try:
    from app.services.separation.demucs_sep import run_demucs_4stems, select_stem_path, get_stem_path, _DEMUCS_AVAILABLE
except ImportError:
    _DEMUCS_AVAILABLE = False
    run_demucs_4stems = None  # type: ignore
    select_stem_path = None  # type: ignore
    get_stem_path = None  # type: ignore
from app.services.musicxml.export import export_musicxml
from app.services.accompaniment.strum import detect_strum_onsets
from app.services.accompaniment.shapes import (
    pick_shape_for_chord,
    shape_pitches,
    shape_positions,
    shape_to_dict,
    Shape,
)
from app.services.guitar.fretboard import STANDARD_TUNING
from app.services.analysis.content_classifier import analyze_musical_content, ContentSegment
from app.services.analysis.audio_quality import analyze_audio_characteristics, calibrate_thresholds

_CHORD_TONE_BIAS = 0.08
_CHORD_CONFIDENCE_THRESHOLD = 0.03
_SEVENTH_MIN_CONFIDENCE = 0.03
_SEVENTH_MIN_DURATION = 0.6
_SEVENTH_RATIO = 0.55
_ACC_MIN_GRID_Q = 0.5
_ACC_MIN_SEGMENT_SEC = 0.6
_ACC_MIN_CONFIDENCE = 0.05
_ACC_SWITCH_PENALTY = 4.0
_TAB_MAX_FRET = 20
_TAB_STRING_PENALTY = 0.35
_TAB_FRET_PENALTY = 0.05
_TAB_ANCHOR_PENALTY = 0.6

_LOG = logging.getLogger(__name__)

_VF_NOTE_NAMES_SHARP = ["c", "c#", "d", "d#", "e", "f", "f#", "g", "g#", "a", "a#", "b"]
_VF_NOTE_NAMES_FLAT = ["c", "db", "d", "eb", "e", "f", "gb", "g", "ab", "a", "bb", "b"]

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

def _job_title(job_dir: Path, input_path: Path) -> str:
    meta_path = job_dir / "input" / "meta.json"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        filename = str(meta.get("filename") or "").strip()
        if filename:
            return Path(filename).stem or filename
    except Exception:
        pass
    return input_path.stem or "Lead Sheet"


@dataclass(frozen=True)
class _StrumEvent:
    time_s: float
    keys: list[str]
    positions: list[tuple[int, int]]
    pitches: list[int]


@dataclass(frozen=True)
class GuitarTranscriptionResult:
    segments: list[ContentSegment]
    note_events: list[NoteEvent]
    chord_events: list[ChordSegment]
    score_data: ScoreData
    pickup_quarters: float
    score_m21: object | None = None
    tab_positions: list[list[list[tuple[int, int]]]] | None = None


def _normalize_transcription_mode(mode: str | None) -> str:
    mode = str(mode or "guitar").strip().lower()
    if mode not in ("notes", "accompaniment", "guitar"):
        return "guitar"
    return mode


def _midi_to_vf_key(pitch_midi: int, *, use_flats: bool) -> str:
    pc = int(pitch_midi) % 12
    octave = int(pitch_midi) // 12 - 1
    name = _VF_NOTE_NAMES_FLAT[pc] if use_flats else _VF_NOTE_NAMES_SHARP[pc]
    return f"{name}/{octave}"


def _decompose_duration_straight(duration_q: float) -> list[tuple[str, int, float]]:
    out: list[tuple[str, int, float]] = []
    rem = float(duration_q)
    eps = 1e-6

    for dur, dots, ql in _DUR_TOKENS_STRAIGHT:
        while rem + eps >= ql:
            out.append((dur, dots, ql))
            rem -= ql

    if rem > 1e-3:
        out.append(("32", 0, max(0.125, rem)))

    return out


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


def _choose_strum_grid(positions: np.ndarray) -> float:
    if positions.size == 0:
        return 0.5

    candidates = [
        (0.25, 1.1),
        (0.5, 1.0),
        (1.0, 1.05),
    ]
    best = None
    for grid, penalty in candidates:
        q = np.round(positions / grid) * grid
        err = float(np.mean(np.abs(positions - q)))
        cost = err * penalty
        if best is None or cost < best[0]:
            best = (cost, grid)
    assert best is not None
    return float(best[1])


def _assign_shapes(chords: list[ChordSegment]) -> list[tuple[ChordSegment, Shape | None]]:
    out: list[tuple[ChordSegment, Shape | None]] = []
    prev: Shape | None = None
    for seg in chords:
        if seg.label == "N":
            out.append((seg, None))
            continue
        shape = pick_shape_for_chord(seg.label, prev)
        if shape is not None:
            prev = shape
        out.append((seg, shape))
    return out


def _shape_at_time(
    t_sec: float,
    segments: list[tuple[ChordSegment, Shape | None]],
    idx: int,
) -> tuple[int, Shape | None, str]:
    i = idx
    while i < len(segments) and float(segments[i][0].end) <= t_sec:
        i += 1
    if i >= len(segments):
        return i, None, "N"
    seg, shape = segments[i]
    if float(seg.start) <= t_sec < float(seg.end):
        return i, shape, str(seg.label or "N")
    return i, None, "N"


def _build_strum_events(
    onsets_s: np.ndarray,
    segments: list[tuple[ChordSegment, Shape | None]],
    *,
    use_flats: bool,
) -> list[_StrumEvent]:
    if onsets_s.size == 0:
        return []

    events: list[_StrumEvent] = []
    seg_idx = 0
    for t in np.sort(onsets_s):
        seg_idx, shape, _label = _shape_at_time(float(t), segments, seg_idx)
        if shape is None:
            events.append(_StrumEvent(time_s=float(t), keys=[], positions=[], pitches=[]))
            continue
        pitches = shape_pitches(shape)
        positions = shape_positions(shape)
        keys = [_midi_to_vf_key(p, use_flats=use_flats) for p in pitches]
        events.append(_StrumEvent(time_s=float(t), keys=keys, positions=positions, pitches=pitches))
    return events


def _strum_events_to_note_events(
    events: list[_StrumEvent],
    *,
    tempo_bpm: float,
) -> list[NoteEvent]:
    tempo = float(tempo_bpm) if tempo_bpm and tempo_bpm > 0 else 120.0
    sec_per_q = 60.0 / tempo
    dur_s = max(0.08, 0.2 * sec_per_q)
    out: list[NoteEvent] = []
    for ev in events:
        for pitch in ev.pitches:
            out.append(
                NoteEvent(
                    start_time_s=float(ev.time_s),
                    end_time_s=float(ev.time_s) + float(dur_s),
                    pitch_midi=int(pitch),
                    velocity=90,
                    amplitude=1.0,
                )
            )
    return out


def _quantize_strum_events(
    events: list[_StrumEvent],
    *,
    beat_times: np.ndarray | None,
    tempo_bpm: float,
    time_signature: str,
    min_grid_q: float,
) -> tuple[ScoreData, float, list[list[list[tuple[int, int]]]]]:
    if not events:
        try:
            num_s, den_s = (time_signature or "4/4").split("/")
            num = int(num_s)
            den = int(den_s)
        except Exception:
            num, den = 4, 4
        measure_q = float(num) * (4.0 / float(den))
        items: list[ScoreItem] = []
        positions: list[list[tuple[int, int]]] = []
        for dur, dots, _ql in _decompose_duration_straight(measure_q):
            items.append(ScoreItem(rest=True, keys=[], duration=dur, dots=dots))
            positions.append([])
        empty = ScoreMeasure(number=1, items=items)
        return ScoreData(grid_q=1.0, grid_kind="straight", measures=[empty]), 0.0, [positions]

    times = np.asarray([e.time_s for e in events], dtype=np.float64)
    if beat_times is not None and len(beat_times) > 1:
        positions = _to_beats(times, np.asarray(beat_times, dtype=np.float64))
    else:
        tempo = float(tempo_bpm) if tempo_bpm and tempo_bpm > 0 else 120.0
        sec_per_q = 60.0 / tempo
        positions = times / float(sec_per_q)

    grid_q = _choose_strum_grid(positions)
    grid_q = max(float(grid_q), float(min_grid_q))

    steps = np.round(positions / grid_q).astype(int)
    step_map: dict[int, _StrumEvent] = {}
    for step, ev in zip(steps, events):
        prev = step_map.get(int(step))
        if prev is None or len(ev.keys) > len(prev.keys):
            step_map[int(step)] = ev

    steps_sorted = sorted(step_map.keys())
    if not steps_sorted:
        try:
            num_s, den_s = (time_signature or "4/4").split("/")
            num = int(num_s)
            den = int(den_s)
        except Exception:
            num, den = 4, 4
        measure_q = float(num) * (4.0 / float(den))
        items: list[ScoreItem] = []
        positions: list[list[tuple[int, int]]] = []
        for dur, dots, _ql in _decompose_duration_straight(measure_q):
            items.append(ScoreItem(rest=True, keys=[], duration=dur, dots=dots))
            positions.append([])
        empty = ScoreMeasure(number=1, items=items)
        return ScoreData(grid_q=float(grid_q), grid_kind="straight", measures=[empty]), 0.0, [positions]

    min_step = min(0, int(steps_sorted[0]))
    default_steps = max(1, int(round(1.0 / float(grid_q))))

    timeline: list[tuple[list[str], list[tuple[int, int]], int]] = []
    if steps_sorted[0] > min_step:
        timeline.append(([], [], int(steps_sorted[0] - min_step)))

    for i, step in enumerate(steps_sorted):
        ev = step_map[int(step)]
        next_step = steps_sorted[i + 1] if i + 1 < len(steps_sorted) else int(step) + default_steps
        dur = int(next_step - int(step))
        if dur <= 0:
            dur = 1
        timeline.append((list(ev.keys), list(ev.positions), dur))

    try:
        num_s, den_s = (time_signature or "4/4").split("/")
        num = int(num_s)
        den = int(den_s)
    except Exception:
        num, den = 4, 4

    measure_q = float(num) * (4.0 / float(den))
    steps_per_measure = int(round(measure_q / float(grid_q))) if grid_q > 0 else 16
    if steps_per_measure <= 0:
        steps_per_measure = 16

    pickup_steps = max(0, int(-min_step))
    if pickup_steps >= steps_per_measure:
        pickup_steps = int(pickup_steps % steps_per_measure)
    pickup_quarters = float(pickup_steps) * float(grid_q)

    measures: list[ScoreMeasure] = []
    tab_positions: list[list[list[tuple[int, int]]]] = []
    current_items: list[ScoreItem] = []
    current_positions: list[list[tuple[int, int]]] = []
    measure_number = 1
    remaining_steps = pickup_steps if pickup_steps > 0 else steps_per_measure

    def flush_measure() -> None:
        nonlocal current_items, current_positions, measure_number
        measures.append(ScoreMeasure(number=measure_number, items=current_items))
        tab_positions.append(current_positions)
        current_items = []
        current_positions = []
        measure_number += 1

    def emit_item(keys: list[str], positions: list[tuple[int, int]], duration: str, dots: int, tie: str | None) -> None:
        current_items.append(
            ScoreItem(
                rest=(len(keys) == 0),
                keys=list(keys),
                duration=duration,
                dots=dots,
                tie=tie,  # type: ignore[arg-type]
            )
        )
        current_positions.append(list(positions) if keys else [])

    for keys, positions, dur_steps in timeline:
        if dur_steps <= 0:
            continue

        count_steps = int(dur_steps)
        if keys:
            rem = int(remaining_steps)
            steps_left = int(count_steps)
            item_total = 0
            while steps_left > 0:
                take = min(steps_left, rem)
                parts = _decompose_duration_straight(float(take) * float(grid_q))
                item_total += len(parts)
                steps_left -= take
                rem -= take
                if rem <= 0:
                    rem = int(steps_per_measure)
        else:
            item_total = 0

        steps_left = int(count_steps)
        item_idx = 0
        while steps_left > 0:
            take = min(steps_left, remaining_steps)
            dur_q = float(take) * float(grid_q)
            parts = _decompose_duration_straight(dur_q)
            for i, (dur, dots, _ql) in enumerate(parts):
                item_idx += 1
                tie = None
                if keys and item_total > 1:
                    if item_idx == 1:
                        tie = "start"
                    elif item_idx == item_total:
                        tie = "stop"
                    else:
                        tie = "continue"
                emit_item(keys, positions, duration=dur, dots=dots, tie=tie)
            steps_left -= take
            remaining_steps -= take
            if remaining_steps <= 0:
                flush_measure()
                remaining_steps = int(steps_per_measure)

    if current_items:
        flush_measure()

    score = ScoreData(grid_q=float(grid_q), grid_kind="straight", measures=measures)
    return score, pickup_quarters, tab_positions

# Chords imports
from app.services.chords.template import (
    build_chord_library,
    chroma_features,
    emission_probs,
    frames_to_segments,
    finalize_segments,
    Segment
)
from app.services.chords.viterbi import viterbi_decode


def detect_chords(y, sr) -> list[Segment]:
    """
    Runs the Viterbi-based chord detection pipeline.
    """
    # 1. Compute Chroma
    chroma, harm_rms = chroma_features(y, sr)

    # 2. Build Library (Maj/Min/7)
    labels, T = build_chord_library(vocab="majmin7")

    # 3. Compute Probabilities
    probs = emission_probs(chroma, harm_rms, labels, T)

    # 4. Viterbi Decode
    # switch_penalty tunes how often chords change.
    # -5.0 is a reasonable starting point for log-prob costs.
    path, conf = viterbi_decode(probs, switch_penalty=-5.0)

    # 5. Convert to Segments
    hop_length = 512
    times = [i * hop_length / sr for i in range(len(path))]

    # Minimum chord length in seconds (e.g. 0.5s)
    raw_segs = frames_to_segments(path, conf, times, min_len=0.5)

    # 6. Finalize (map indices to labels)
    segs = finalize_segments(raw_segs, labels)

    return segs


def _chord_tone_pcs(label: str) -> set[int] | None:
    if not label or label == "N":
        return None

    label = label.split("/", 1)[0].strip()
    root = None
    qual = ""
    if ":" in label:
        root, qual = label.split(":", 1)
    else:
        match = re.match(r"^([A-Ga-g])([#b]?)(.*)$", label)
        if match:
            root = f"{match.group(1).upper()}{match.group(2)}"
            qual = match.group(3) or ""
    root = (root or "").strip()
    qual = (qual or "").strip().lower().replace("(", "").replace(")", "").replace(" ", "")

    pc = NOTE_TO_PC.get(root)
    if pc is None:
        return None

    if qual in ("", "maj", "major"):
        intervals = [0, 4, 7]
    elif qual in ("min", "m", "minor"):
        intervals = [0, 3, 7]
    elif "min7b5" in qual or "m7b5" in qual or "hdim" in qual:
        intervals = [0, 3, 6]
    elif "dim" in qual:
        intervals = [0, 3, 6]
    elif "aug" in qual:
        intervals = [0, 4, 8]
    elif "sus2" in qual:
        intervals = [0, 2, 7]
    elif "sus4" in qual or "sus" in qual:
        intervals = [0, 5, 7]
    else:
        intervals = [0, 4, 7]

    has_add9 = "add9" in qual
    if "maj7" in qual or "maj9" in qual or "maj13" in qual:
        intervals.append(11)
    elif "dim7" in qual:
        intervals.append(9)
    elif "7" in qual or (("9" in qual or "11" in qual or "13" in qual) and not has_add9):
        intervals.append(10)
    elif "6" in qual:
        intervals.append(9)

    if "b9" in qual:
        intervals.append(13)
    if "#9" in qual:
        intervals.append(15)
    if "9" in qual and "b9" not in qual and "#9" not in qual:
        intervals.append(14)
    if "#11" in qual:
        intervals.append(18)
    elif "11" in qual:
        intervals.append(17)
    if "b13" in qual:
        intervals.append(20)
    elif "13" in qual:
        intervals.append(21)

    return {int((pc + i) % 12) for i in intervals}


def _merge_overlapping_notes(
    note_events: Iterable[NoteEvent],
    *,
    gap_s: float = 0.03,
) -> list[NoteEvent]:
    by_pitch: dict[int, list[NoteEvent]] = {}
    for ev in note_events:
        by_pitch.setdefault(int(ev.pitch_midi), []).append(ev)

    merged: list[NoteEvent] = []
    for pitch, events in by_pitch.items():
        events_sorted = sorted(events, key=lambda e: e.start_time_s)
        cur: NoteEvent | None = None
        for ev in events_sorted:
            if cur is None:
                cur = ev
                continue
            if float(ev.start_time_s) <= float(cur.end_time_s) + float(gap_s):
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


def _limit_onset_polyphony(
    note_events: Iterable[NoteEvent],
    *,
    max_notes: int = 6,
    onset_window_s: float = 0.03,
) -> list[NoteEvent]:
    events = sorted(note_events, key=lambda e: e.start_time_s)
    if not events:
        return []

    def pick_group(group: list[NoteEvent]) -> list[NoteEvent]:
        by_pitch: dict[int, NoteEvent] = {}
        for ev in group:
            prev = by_pitch.get(int(ev.pitch_midi))
            if prev is None or float(ev.amplitude) > float(prev.amplitude):
                by_pitch[int(ev.pitch_midi)] = ev
        candidates = list(by_pitch.values())
        candidates.sort(key=lambda e: float(e.amplitude), reverse=True)
        return candidates[: max(1, int(max_notes))]

    out: list[NoteEvent] = []
    group: list[NoteEvent] = [events[0]]
    last_start = float(events[0].start_time_s)
    for ev in events[1:]:
        if float(ev.start_time_s) - last_start <= float(onset_window_s):
            group.append(ev)
        else:
            out.extend(pick_group(group))
            group = [ev]
            last_start = float(ev.start_time_s)
    out.extend(pick_group(group))

    return sorted(out, key=lambda e: e.start_time_s)


def _filter_note_events(
    note_events: Iterable[NoteEvent],
    *,
    chords: Iterable[ChordSegment],
    min_amp: float,
    min_dur_s: float,
    min_pitch: int,
    max_pitch: int,
    chord_tone_bias: float = 0.08,
    chord_confidence_threshold: float | None = None,
) -> list[NoteEvent]:
    events = sorted(note_events, key=lambda e: e.start_time_s)
    chords_sorted = sorted(chords, key=lambda c: float(c.start))
    seg_idx = 0

    def label_at(t_sec: float) -> tuple[str, float]:
        nonlocal seg_idx
        while seg_idx < len(chords_sorted) and float(chords_sorted[seg_idx].end) <= t_sec:
            seg_idx += 1
        if seg_idx >= len(chords_sorted):
            return "N", 0.0
        seg = chords_sorted[seg_idx]
        if float(seg.start) <= t_sec < float(seg.end):
            return seg.label or "N", float(seg.confidence)
        return "N", 0.0

    out: list[NoteEvent] = []
    min_amp = float(min_amp)
    min_dur_s = float(min_dur_s)
    min_pitch = int(min_pitch)
    max_pitch = int(max_pitch)
    chord_tone_bias = float(chord_tone_bias)

    for ev in events:
        dur = float(ev.end_time_s) - float(ev.start_time_s)
        if dur < min_dur_s:
            continue
        if float(ev.amplitude) < min_amp:
            continue
        pitch = int(ev.pitch_midi)
        if pitch < min_pitch or pitch > max_pitch:
            continue

        if chords_sorted:
            mid = 0.5 * (float(ev.start_time_s) + float(ev.end_time_s))
            label, conf = label_at(mid)
            if chord_confidence_threshold is not None and conf < float(chord_confidence_threshold):
                label = "N"
            pcs = _chord_tone_pcs(label)
            if pcs is not None and (pitch % 12) not in pcs:
                if float(ev.amplitude) < (min_amp + chord_tone_bias):
                    continue

        out.append(ev)

    return out


def _post_process_note_events(
    note_events: list[NoteEvent],
    *,
    chords: list[ChordSegment],
    tempo_bpm: float,
) -> list[NoteEvent]:
    if not note_events:
        return []

    note_events = remove_harmonic_duplicates(note_events)
    if not note_events:
        return []

    note_events = merge_temporal_clusters(
        note_events,
        window_ms=float(getattr(settings, "TEMPORAL_CLUSTER_WINDOW_MS", 80.0)),
    )
    if not note_events:
        return []

    note_events = _merge_overlapping_notes(note_events, gap_s=0.03)

    amps = np.asarray([float(ev.amplitude) for ev in note_events], dtype=np.float32)
    if amps.size > 0:
        min_amp = max(0.2, float(np.percentile(amps, 35)))
    else:
        min_amp = 0.2

    sec_per_q = 60.0 / float(tempo_bpm if tempo_bpm else 120.0)
    min_dur_s = max(0.08, 0.2 * sec_per_q)

    chord_conf_threshold = None
    if chords:
        confs = np.asarray([float(c.confidence) for c in chords], dtype=np.float32)
        if confs.size > 0:
            chord_conf_threshold = max(float(_CHORD_CONFIDENCE_THRESHOLD), float(np.median(confs)) * 0.9)

    note_events = _filter_note_events(
        note_events,
        chords=chords,
        min_amp=min_amp,
        min_dur_s=min_dur_s,
        min_pitch=40,
        max_pitch=88,
        chord_tone_bias=float(_CHORD_TONE_BIAS),
        chord_confidence_threshold=chord_conf_threshold,
    )
    note_events = _limit_onset_polyphony(
        note_events,
        max_notes=6,
        onset_window_s=0.06,
    )
    note_events = apply_music_theory_rules(
        note_events,
        chords=chords,
        key_sig=None,
    )
    return note_events


def _shift_note_events(note_events: Iterable[NoteEvent], offset_s: float) -> list[NoteEvent]:
    offset_s = float(offset_s or 0.0)
    if abs(offset_s) <= 1e-9:
        return list(note_events)
    out: list[NoteEvent] = []
    for ev in note_events:
        out.append(
            NoteEvent(
                start_time_s=float(ev.start_time_s) - offset_s,
                end_time_s=float(ev.end_time_s) - offset_s,
                pitch_midi=int(ev.pitch_midi),
                velocity=int(ev.velocity),
                amplitude=float(ev.amplitude),
            )
        )
    return out


def _shift_chords(chords: Iterable[ChordSegment], offset_s: float) -> list[ChordSegment]:
    offset_s = float(offset_s or 0.0)
    if abs(offset_s) <= 1e-9:
        return list(chords)
    out: list[ChordSegment] = []
    for c in chords:
        out.append(
            ChordSegment(
                start=float(c.start) - offset_s,
                end=float(c.end) - offset_s,
                label=str(c.label),
                confidence=float(c.confidence),
            )
        )
    return out


def _shift_content_segments(
    segments: Iterable[ContentSegment],
    offset_s: float,
) -> list[ContentSegment]:
    offset_s = float(offset_s or 0.0)
    if abs(offset_s) <= 1e-9:
        return list(segments)
    out: list[ContentSegment] = []
    for seg in segments:
        out.append(
            ContentSegment(
                start_time_s=float(seg.start_time_s) - offset_s,
                end_time_s=float(seg.end_time_s) - offset_s,
                content_type=seg.content_type,
                confidence=float(seg.confidence),
                metrics=dict(seg.metrics or {}),
            )
        )
    return out


def _vf_key_to_midi(key: str) -> int | None:
    try:
        note, octave_s = key.split("/")
        note = note.strip()
        octave = int(octave_s)
        if not note:
            return None
        note_name = note[0].upper() + note[1:]
        pc = NOTE_TO_PC.get(note_name)
        if pc is None:
            return None
        return int((octave + 1) * 12 + int(pc))
    except Exception:
        return None


def _pitch_to_tab_positions(pitch_midi: int, *, max_fret: int = _TAB_MAX_FRET) -> list[tuple[int, int]]:
    positions: list[tuple[int, int]] = []
    for i, open_pitch in enumerate(STANDARD_TUNING):
        fret = int(pitch_midi) - int(open_pitch)
        if 0 <= fret <= int(max_fret):
            string_num = 6 - int(i)
            positions.append((int(string_num), int(fret)))
    return positions


def _tab_position_cost(
    string_num: int,
    fret: int,
    *,
    prev_pos: tuple[int, int] | None,
    anchor_fret: float | None,
) -> float:
    cost = 0.0
    if prev_pos is not None:
        prev_string, prev_fret = prev_pos
        cost += abs(int(fret) - int(prev_fret))
        cost += float(_TAB_STRING_PENALTY) * abs(int(string_num) - int(prev_string))
    if anchor_fret is not None:
        cost += float(_TAB_ANCHOR_PENALTY) * abs(int(fret) - float(anchor_fret))
    cost += float(_TAB_FRET_PENALTY) * float(fret)
    return float(cost)


def _select_tab_position(
    pitch_midi: int,
    *,
    prev_pos: tuple[int, int] | None,
    anchor_fret: float | None,
) -> tuple[int, int] | None:
    candidates = _pitch_to_tab_positions(pitch_midi)
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda p: _tab_position_cost(p[0], p[1], prev_pos=prev_pos, anchor_fret=anchor_fret),
    )


def _assign_chord_positions(
    pitches: list[int],
    *,
    prev_pos: tuple[int, int] | None,
    anchor_fret: float | None,
) -> list[tuple[int, int]] | None:
    if not pitches:
        return None
    pitches_sorted = sorted(pitches)
    candidates: dict[int, list[tuple[int, int]]] = {
        pitch: _pitch_to_tab_positions(pitch) for pitch in pitches_sorted
    }
    if any(not cand for cand in candidates.values()):
        return None

    best_cost: float | None = None
    best_positions: list[tuple[int, int]] | None = None

    def backtrack(
        idx: int,
        used_strings: set[int],
        current: list[tuple[int, int]],
        cost_so_far: float,
    ) -> None:
        nonlocal best_cost, best_positions
        if idx >= len(pitches_sorted):
            if best_cost is None or cost_so_far < best_cost:
                best_cost = cost_so_far
                best_positions = list(current)
            return
        if best_cost is not None and cost_so_far >= best_cost:
            return

        pitch = pitches_sorted[idx]
        for string_num, fret in candidates[pitch]:
            if string_num in used_strings:
                continue
            cost = _tab_position_cost(
                string_num,
                fret,
                prev_pos=prev_pos,
                anchor_fret=anchor_fret,
            )
            used_strings.add(string_num)
            current.append((string_num, fret))
            backtrack(idx + 1, used_strings, current, cost_so_far + cost)
            current.pop()
            used_strings.remove(string_num)

    backtrack(0, set(), [], 0.0)
    if best_positions is not None:
        return best_positions

    fallback: list[tuple[int, int]] = []
    for pitch in pitches_sorted:
        pos = _select_tab_position(pitch, prev_pos=prev_pos, anchor_fret=anchor_fret)
        if pos is None:
            return None
        fallback.append(pos)
    return fallback


_DUR_QUARTERS = {
    "w": 4.0,
    "h": 2.0,
    "q": 1.0,
    "8": 0.5,
    "16": 0.25,
    "32": 0.125,
}


def _duration_to_quarters(item: ScoreItem) -> float:
    base = _DUR_QUARTERS.get(str(item.duration))
    if base is None:
        return 0.0
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


def _content_type_at_time(
    t_sec: float,
    segments: list[ContentSegment],
    idx: int,
) -> tuple[int, str]:
    i = idx
    while i < len(segments) and float(segments[i].end_time_s) <= t_sec:
        i += 1
    if i < len(segments):
        seg = segments[i]
        if float(seg.start_time_s) <= t_sec < float(seg.end_time_s):
            return i, seg.content_type
    return i, "hybrid"


def _shape_positions_for_pitches(shape: Shape, pitches: list[int]) -> list[tuple[int, int]] | None:
    if not pitches:
        return None
    shape_p = shape_pitches(shape)
    shape_pos = shape_positions(shape)
    if len(shape_p) != len(shape_pos):
        return None

    mapping: dict[int, list[tuple[int, int]]] = {}
    for pitch, pos in zip(shape_p, shape_pos):
        mapping.setdefault(int(pitch), []).append(pos)

    positions: list[tuple[int, int]] = []
    for pitch in pitches:
        options = mapping.get(int(pitch))
        if not options:
            return None
        positions.append(options.pop(0))
    return positions


def _build_tab_positions_for_guitar(
    score: ScoreData,
    *,
    content_segments: list[ContentSegment],
    chords: list[ChordSegment],
    beat_times: np.ndarray | None,
    tempo_bpm: float,
    pickup_quarters: float,
) -> list[list[list[tuple[int, int]]]]:
    if not score.measures:
        return []

    segments_sorted = sorted(content_segments, key=lambda s: float(s.start_time_s))
    segments_idx = 0
    chord_shapes = _assign_shapes(chords)
    chord_idx = 0

    tab_positions: list[list[list[tuple[int, int]]]] = []
    prev_pos: tuple[int, int] | None = None
    offset_q = 0.0

    for measure in score.measures:
        measure_positions: list[list[tuple[int, int]]] = []
        for item in measure.items:
            dur_q = _duration_to_quarters(item)
            t_q = float(offset_q) - float(pickup_quarters or 0.0)
            t_sec = _beats_to_seconds(t_q, beat_times, tempo_bpm)

            segments_idx, content_type = _content_type_at_time(t_sec, segments_sorted, segments_idx)
            chord_idx, shape, _label = _shape_at_time(t_sec, chord_shapes, chord_idx)
            anchor_fret = float(shape.position) if shape is not None else None

            if item.rest or not item.keys:
                measure_positions.append([])
                offset_q += dur_q
                continue

            pitches: list[int] = []
            for key in item.keys:
                midi = _vf_key_to_midi(str(key))
                if midi is not None:
                    pitches.append(int(midi))
            pitches = sorted(set(pitches))
            if not pitches:
                measure_positions.append([])
                offset_q += dur_q
                continue

            positions: list[tuple[int, int]] | None = None
            if content_type == "chordal":
                if shape is not None:
                    positions = _shape_positions_for_pitches(shape, pitches)
                if positions is None:
                    positions = _assign_chord_positions(pitches, prev_pos=prev_pos, anchor_fret=anchor_fret)
            elif content_type == "hybrid":
                if len(pitches) > 1 and shape is not None:
                    positions = _shape_positions_for_pitches(shape, pitches)
                if positions is None:
                    if len(pitches) == 1:
                        pos = _select_tab_position(pitches[0], prev_pos=prev_pos, anchor_fret=anchor_fret)
                        positions = [pos] if pos is not None else None
                    else:
                        positions = _assign_chord_positions(pitches, prev_pos=prev_pos, anchor_fret=anchor_fret)
            else:  # melodic
                if len(pitches) == 1:
                    pos = _select_tab_position(pitches[0], prev_pos=prev_pos, anchor_fret=None)
                    positions = [pos] if pos is not None else None
                else:
                    positions = _assign_chord_positions(pitches, prev_pos=prev_pos, anchor_fret=None)

            if positions is None or len(positions) != len(pitches):
                measure_positions.append([])
            else:
                measure_positions.append(positions)
                avg_string = sum(p[0] for p in positions) / float(len(positions))
                avg_fret = sum(p[1] for p in positions) / float(len(positions))
                prev_pos = (int(round(avg_string)), int(round(avg_fret)))

            offset_q += dur_q

        tab_positions.append(measure_positions)

    return tab_positions


def _parse_chord_label(label: str) -> tuple[str | None, str | None]:
    if not label or label == "N":
        return None, None
    label = label.split("/", 1)[0].strip()
    if ":" in label:
        root, qual = label.split(":", 1)
        root = root.strip()
        qual = qual.strip().lower() or "maj"
    else:
        match = re.match(r"^([A-Ga-g])([#b]?)(.*)$", label)
        if match:
            root = f"{match.group(1).upper()}{match.group(2)}"
            qual = (match.group(3) or "").strip().lower() or "maj"
        else:
            root, qual = label.strip(), "maj"
    if not root:
        return None, None
    qual = qual.replace("(", "").replace(")", "").replace(" ", "")
    if qual in ("major", ""):
        qual = "maj"
    if qual in ("minor", "m"):
        qual = "min"
    if "min7b5" in qual or "m7b5" in qual or "hdim" in qual:
        qual = "min"
    if "dim" in qual:
        qual = "min"
    if "aug" in qual:
        qual = "maj"
    return root, qual


def _triad_label(root: str, qual: str) -> str:
    if qual in ("min", "m", "min7", "m7", "dim", "min7b5", "m7b5", "hdim"):
        return f"{root}:min"
    return f"{root}:maj"


def _segment_chroma_energy(
    chroma: np.ndarray | None,
    times: np.ndarray | None,
    start: float,
    end: float,
) -> np.ndarray | None:
    if chroma is None or times is None:
        return None
    if chroma.ndim != 2 or times.ndim != 1:
        return None
    if chroma.shape[1] != times.shape[0]:
        return None
    if end <= start:
        return None
    mask = (times >= float(start)) & (times < float(end))
    if not np.any(mask):
        return None
    return np.mean(chroma[:, mask], axis=1)


def _simplify_chord_segments(
    chords: list[ChordSegment],
    *,
    chroma: np.ndarray | None,
    times: np.ndarray | None,
    min_confidence: float,
    min_duration: float,
    seventh_ratio: float,
) -> list[ChordSegment]:
    if not chords:
        return []

    confs = np.asarray([float(c.confidence) for c in chords], dtype=np.float32)
    conf_baseline = float(np.median(confs)) if confs.size > 0 else float(min_confidence)
    conf_threshold = max(float(min_confidence), conf_baseline * 0.9)

    out: list[ChordSegment] = []
    for i, c in enumerate(chords):
        label = str(c.label or "N")
        root, qual = _parse_chord_label(label)
        if root is None or qual is None:
            out.append(c)
            continue

        is_seventh = qual in ("7", "min7", "m7", "maj7")
        if not is_seventh:
            out.append(c)
            continue

        collapse = False
        dur = float(c.end) - float(c.start)
        if dur < float(min_duration) or float(c.confidence) < float(conf_threshold):
            collapse = True

        if not collapse:
            energy = _segment_chroma_energy(chroma, times, float(c.start), float(c.end))
            if energy is not None:
                root_pc = NOTE_TO_PC.get(root)
                if root_pc is not None:
                    third = 3 if qual in ("min7", "m7") else 4
                    triad_pcs = [
                        int((root_pc + 0) % 12),
                        int((root_pc + third) % 12),
                        int((root_pc + 7) % 12),
                    ]
                    triad_energy = float(np.mean([energy[pc] for pc in triad_pcs]))
                    seventh_pc = int((root_pc + (11 if qual == "maj7" else 10)) % 12)
                    seventh_energy = float(energy[seventh_pc])
                    if triad_energy > 1e-6 and seventh_energy < triad_energy * float(seventh_ratio):
                        collapse = True

        if not collapse and 0 < i < (len(chords) - 1):
            prev_root, prev_qual = _parse_chord_label(str(chords[i - 1].label))
            next_root, next_qual = _parse_chord_label(str(chords[i + 1].label))
            if prev_root == root and next_root == root:
                if _triad_label(prev_root, prev_qual or "maj") == _triad_label(root, qual) == _triad_label(next_root, next_qual or "maj"):
                    collapse = True

        if collapse:
            label = _triad_label(root, qual)

        out.append(
            ChordSegment(
                start=float(c.start),
                end=float(c.end),
                label=label,
                confidence=float(c.confidence),
            )
        )

    return out


def _simplify_chords_for_accompaniment(
    chords: list[ChordSegment],
    *,
    min_duration: float,
    min_confidence: float,
) -> list[ChordSegment]:
    if not chords:
        return []

    triads: list[ChordSegment] = []
    for c in chords:
        label = str(c.label or "N")
        root, qual = _parse_chord_label(label)
        if root is None or qual is None:
            triads.append(c)
            continue
        triads.append(
            ChordSegment(
                start=float(c.start),
                end=float(c.end),
                label=_triad_label(root, qual),
                confidence=float(c.confidence),
            )
        )

    out: list[ChordSegment] = []
    i = 0
    while i < len(triads):
        seg = triads[i]
        dur = float(seg.end) - float(seg.start)
        if dur < float(min_duration) or float(seg.confidence) < float(min_confidence):
            if i + 1 < len(triads):
                nxt = triads[i + 1]
                out.append(
                    ChordSegment(
                        start=float(seg.start),
                        end=float(nxt.end),
                        label=str(nxt.label),
                        confidence=max(float(seg.confidence), float(nxt.confidence)),
                    )
                )
                i += 2
                continue
            if out:
                prev = out[-1]
                out[-1] = ChordSegment(
                    start=float(prev.start),
                    end=float(seg.end),
                    label=str(prev.label),
                    confidence=max(float(prev.confidence), float(seg.confidence)),
                )
                i += 1
                continue
        out.append(seg)
        i += 1

    merged: list[ChordSegment] = []
    for seg in out:
        if merged and str(seg.label) == str(merged[-1].label):
            prev = merged[-1]
            merged[-1] = ChordSegment(
                start=float(prev.start),
                end=float(seg.end),
                label=str(prev.label),
                confidence=max(float(prev.confidence), float(seg.confidence)),
            )
        else:
            merged.append(seg)
    return merged


def _tempo_from_beat_times(beat_times: np.ndarray | None) -> float:
    if beat_times is None or len(beat_times) < 2:
        return 0.0
    diffs = np.diff(np.asarray(beat_times, dtype=np.float64))
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size == 0:
        return 0.0
    # Median is more robust than mean when beat_track jitters.
    return float(60.0 / float(np.median(diffs)))


def _extract_audio_segment(
    y: np.ndarray,
    sr: int,
    start_s: float,
    end_s: float,
) -> np.ndarray:
    """Extract a segment of audio between start and end times."""
    start_sample = int(start_s * sr)
    end_sample = int(end_s * sr)
    start_sample = max(0, min(start_sample, len(y)))
    end_sample = max(start_sample, min(end_sample, len(y)))
    return y[start_sample:end_sample]


def _run_guitar_mode(
    y: np.ndarray,
    sr: int,
    audio_path: Path,
    chords: list[ChordSegment],
    beat_times: np.ndarray | None,
    tempo_bpm: float,
    *,
    base_note_events: list[NoteEvent] | None = None,
    use_flats: bool = False,
    onset_threshold: float = 0.5,
    frame_threshold: float = 0.3,
    min_note_ms: float = 127.70,
    window_sec: float = 3.0,
    hop_sec: float = 1.5,
) -> tuple[list[NoteEvent], list[_StrumEvent], list[ContentSegment]]:
    """
    Run guitar mode: hybrid transcription that applies melodic or chordal
    processing based on content analysis.

    Args:
        y: Audio signal (mono, normalized)
        sr: Sample rate
        audio_path: Path to audio file for Basic Pitch
        chords: Detected chord segments
        beat_times: Beat times array
        tempo_bpm: Detected tempo
        base_note_events: Precomputed Basic Pitch note events (original timeline)
        use_flats: Whether to use flat notation
        onset_threshold: Basic Pitch onset threshold
        frame_threshold: Basic Pitch frame threshold
        min_note_ms: Minimum note duration in ms
        window_sec: Content analysis window size
        hop_sec: Content analysis hop size

    Returns:
        Tuple of (note_events, strum_events, content_segments)
    """
    # Step 1: Analyze content to classify segments
    content_segments = analyze_musical_content(
        y, sr,
        window_sec=window_sec,
        hop_sec=hop_sec,
        min_segment_sec=1.0,
    )

    all_note_events: list[NoteEvent] = []
    all_strum_events: list[_StrumEvent] = []

    if base_note_events is None:
        try:
            _, base_note_events = transcribe_basic_pitch(
                audio_path,
                onset_threshold=onset_threshold,
                frame_threshold=frame_threshold,
                minimum_note_length_ms=min_note_ms,
                melodia_trick=True,
            )
        except Exception as e:
            print(f"Error executing basic_pitch for guitar mode: {e}")
            base_note_events = []

    # Assign chord shapes for strum events
    segment_shapes = _assign_shapes(chords)

    for seg in content_segments:
        start_s = seg.start_time_s
        end_s = seg.end_time_s

        if seg.content_type in ("melodic", "hybrid"):
            seg_notes = [
                n
                for n in (base_note_events or [])
                if n.start_time_s >= start_s and n.start_time_s < end_s
            ]
            all_note_events.extend(seg_notes)

        if seg.content_type in ("chordal", "hybrid"):
            # For chordal segments: detect strum onsets
            try:
                y_seg = _extract_audio_segment(y, sr, start_s, end_s)
                if len(y_seg) > sr * 0.2:  # At least 200ms of audio
                    beat_times_seg = None
                    if beat_times is not None and len(beat_times) > 1:
                        bt = np.asarray(beat_times, dtype=np.float32)
                        mask = (bt >= float(start_s)) & (bt < float(end_s))
                        if np.count_nonzero(mask) >= 2:
                            beat_times_seg = (bt[mask] - float(start_s)).astype(np.float32)
                    min_interval = 0.12 if seg.content_type == "chordal" else 0.2
                    onset_delta = 0.2 if seg.content_type == "chordal" else 0.25
                    strum_onsets = detect_strum_onsets(
                        y_seg,
                        sr,
                        beat_times=beat_times_seg,
                        tempo_bpm=tempo_bpm,
                        min_interval_s=min_interval,
                        onset_delta=onset_delta,
                        backtrack=False,
                    )
                    # Offset onsets back to global time
                    strum_onsets = strum_onsets + start_s

                    # Build strum events with chord shapes
                    strum_evts = _build_strum_events(
                        strum_onsets, segment_shapes, use_flats=use_flats
                    )
                    all_strum_events.extend(strum_evts)
            except Exception as e:
                print(f"Error in chordal detection for segment {start_s:.2f}-{end_s:.2f}: {e}")

    return all_note_events, all_strum_events, content_segments


def _merge_note_events_for_guitar(
    note_events: list[NoteEvent],
    strum_events: list[_StrumEvent],
    content_segments: list[ContentSegment],
    *,
    tempo_bpm: float,
) -> list[NoteEvent]:
    """
    Merge note events and strum events into a unified list of NoteEvents.

    For segments classified as chordal, converts strum events to note events.
    For melodic segments, keeps note events as-is.
    For hybrid segments, combines both with deduplication.

    Args:
        note_events: Note events from melodic transcription
        strum_events: Strum events from chordal detection
        content_segments: Content classification segments
        tempo_bpm: Tempo in BPM

    Returns:
        Merged list of NoteEvent objects
    """
    # Convert strum events to note events
    strum_notes = _strum_events_to_note_events(strum_events, tempo_bpm=tempo_bpm)

    # Build a lookup for segment types by time
    def get_content_type(t: float) -> str:
        for seg in content_segments:
            if seg.start_time_s <= t < seg.end_time_s:
                return seg.content_type
        return "hybrid"

    merged: list[NoteEvent] = []

    # Add melodic notes from melodic/hybrid segments
    for note in note_events:
        ctype = get_content_type(note.start_time_s)
        if ctype in ("melodic", "hybrid"):
            merged.append(note)

    # Add strum notes from chordal segments
    for note in strum_notes:
        ctype = get_content_type(note.start_time_s)
        if ctype == "chordal":
            merged.append(note)
        elif ctype == "hybrid":
            # For hybrid, add strum notes that don't overlap with existing notes
            overlaps = False
            for existing in merged:
                if (abs(existing.start_time_s - note.start_time_s) < 0.05 and
                    existing.pitch_midi == note.pitch_midi):
                    overlaps = True
                    break
            if not overlaps:
                merged.append(note)

    # Sort by start time
    merged.sort(key=lambda n: n.start_time_s)

    return merged


def merge_transcription_results(
    note_events: list[NoteEvent],
    strum_events: list[_StrumEvent],
    content_segments: list[ContentSegment],
    chords: list[ChordSegment],
    *,
    tempo_bpm: float,
    beat_times: np.ndarray | None,
    time_signature: str,
) -> GuitarTranscriptionResult:
    """
    Merge melodic and chordal transcription outputs and return a unified score.

    This removes duplicates between melodic notes and chord tones, keeps chord
    metadata, and produces a ScoreData ready for export.
    """
    merged = _merge_note_events_for_guitar(
        note_events,
        strum_events,
        content_segments,
        tempo_bpm=tempo_bpm,
    )
    merged = _post_process_note_events(merged, chords=chords, tempo_bpm=float(tempo_bpm))

    quant_res = quantize_note_events_to_score(
        merged,
        tempo_bpm=float(tempo_bpm),
        beat_times=beat_times,
        time_signature=time_signature,
    )

    tab_positions = quant_res.tab_positions
    if tab_positions is None:
        tab_positions = _build_tab_positions_for_guitar(
            quant_res.score,
            content_segments=content_segments,
            chords=chords,
            beat_times=beat_times,
            tempo_bpm=float(tempo_bpm),
            pickup_quarters=float(quant_res.pickup_quarters),
        )

    return GuitarTranscriptionResult(
        segments=content_segments,
        note_events=merged,
        chord_events=chords,
        score_data=quant_res.score,
        pickup_quarters=float(quant_res.pickup_quarters),
        score_m21=quant_res.score_m21,
        tab_positions=tab_positions,
    )


def _score_complexity_cost(score: ScoreData) -> float:
    """
    Heuristic to choose a beat grid that yields more readable notation.
    Lower is better.
    """
    items = [it for m in (score.measures or []) for it in (m.items or [])]
    if not items:
        return 1e9

    n_items = float(len(items))
    n_measures = float(len(score.measures or []))
    n_short = float(sum(1 for it in items if str(it.duration) in ("16", "32")))
    n_ties = float(sum(1 for it in items if it.tie))
    non_rest = [it for it in items if not it.rest]
    avg_poly = float(np.mean([len(it.keys or []) for it in non_rest])) if non_rest else 0.0

    # Bias toward ~4-8 measures for short clips; avoid excessive fragmentation.
    return (
        n_items
        + 0.85 * n_short
        + 0.25 * n_ties
        + 0.35 * avg_poly
        + 0.6 * abs(n_measures - 6.0)
    )


def _pick_best_beat_times(
    note_events: list[NoteEvent],
    beat_times: np.ndarray | None,
    *,
    time_signature: str,
) -> np.ndarray | None:
    if beat_times is None or len(beat_times) < 2 or not note_events:
        return beat_times

    beats = np.asarray(beat_times, dtype=np.float32)
    beats = beats[np.isfinite(beats)]
    if beats.size < 2:
        return beat_times

    # Keep selection fast on long jobs.
    events = sorted(note_events, key=lambda e: float(e.start_time_s))
    if len(events) > 600:
        # Bias toward higher-confidence events but keep temporal ordering.
        top = sorted(events, key=lambda e: float(e.amplitude), reverse=True)[:600]
        events = sorted(top, key=lambda e: float(e.start_time_s))

    candidates: list[np.ndarray] = [beats]
    if beats.size >= 4:
        candidates.append(beats[::2])
        candidates.append(beats[1::2])

    best_cost = None
    best = beats
    for cand in candidates:
        if cand.size < 2:
            continue
        try:
            q = quantize_note_events_to_score(
                events,
                tempo_bpm=120.0,  # ignored when beat_times is provided
                beat_times=cand,
                time_signature=time_signature,
            )
            cost = float(_score_complexity_cost(q.score))
        except Exception:
            continue

        if best_cost is None or cost < best_cost:
            best_cost = cost
            best = cand

    return best.astype(np.float32)


def run_pipeline(job_dir: Path, input_path: Path) -> JobResult:
    work = job_dir / "work"
    out = job_dir / "out"
    work.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)

    transcription_mode = _normalize_transcription_mode(settings.TRANSCRIPTION_MODE)
    is_accompaniment = transcription_mode == "accompaniment"
    is_guitar_mode = transcription_mode == "guitar"

    wav_path = work / "audio_mono_44k.wav"
    ffmpeg_to_wav_mono_44k(input_path, wav_path)

    mix_path = wav_path
    transcription_path = mix_path
    beat_path = mix_path
    demucs_error = None
    stems_dir: Path | None = None
    transcription_source = "mix"
    beat_source = "mix"

    if bool(settings.ENABLE_DEMUCS) and _DEMUCS_AVAILABLE and run_demucs_4stems is not None:
        try:
            stems_dir = run_demucs_4stems(
                mix_path,
                work / "demucs",
                model=str(settings.DEMUCS_MODEL),
                return_stem=False,
            )
            # Parse stem priority from config (comma-separated) or use default
            stem_priority_str = getattr(settings, "TRANSCRIPTION_STEM_PRIORITY", "guitar,other,vocals")
            stem_priority = tuple(s.strip() for s in stem_priority_str.split(",") if s.strip())
            if not stem_priority:
                stem_priority = ("guitar", "other", "vocals")

            transcription_path = select_stem_path(stems_dir, stem_priority)
            transcription_source = transcription_path.stem
            # Use drums stem for beat tracking if available
            drums_path = get_stem_path(stems_dir, "drums")
            if drums_path is not None:
                beat_path = drums_path
                beat_source = drums_path.stem
        except Exception as e:
            demucs_error = str(e)
            stems_dir = None
            transcription_path = mix_path
            beat_path = mix_path
            transcription_source = "mix"
            beat_source = "mix"
    elif bool(settings.ENABLE_DEMUCS) and not _DEMUCS_AVAILABLE:
        demucs_error = "Demucs not available (torch/demucs not installed)"

    y_trans, sr_trans = load_wav(transcription_path)
    y_trans = peak_normalize(y_trans)

    harmonic_path = transcription_path
    y_harm = y_trans
    try:
        y_harm = librosa.effects.harmonic(y_trans)
        y_harm = peak_normalize(y_harm)
        harmonic_path = work / "audio_harmonic.wav"
        sf.write(str(harmonic_path), y_harm, int(sr_trans))
    except Exception as e:
        print(f"Error computing harmonic stem: {e}")
        y_harm = y_trans
        harmonic_path = transcription_path

    y_beats, sr_beats = load_wav(beat_path)
    y_beats = peak_normalize(y_beats)

    time_sig = "4/4"
    tempo_raw, beat_times_raw = estimate_beats_librosa(
        y_beats,
        sr_beats,
        use_harmonic=(beat_source != "drums"),
    )
    tempo_for_bp = float(tempo_raw) if tempo_raw and tempo_raw > 0 else 120.0

    onset_threshold = float(settings.BASIC_PITCH_ONSET_THRESHOLD)
    frame_threshold = float(settings.BASIC_PITCH_FRAME_THRESHOLD)
    characteristics: dict[str, float] | None = None
    if bool(getattr(settings, "ENABLE_AUTO_THRESHOLD_CALIBRATION", False)):
        try:
            characteristics = analyze_audio_characteristics(harmonic_path, cache_dir=work)
            onset_threshold, frame_threshold = calibrate_thresholds(characteristics)
            _LOG.info(
                "Auto-calibrated thresholds: onset=%.3f, frame=%.3f",
                float(onset_threshold),
                float(frame_threshold),
            )
        except Exception as exc:
            _LOG.warning("Threshold calibration failed, using defaults: %s", exc)
            characteristics = None
            onset_threshold = float(settings.BASIC_PITCH_ONSET_THRESHOLD)
            frame_threshold = float(settings.BASIC_PITCH_FRAME_THRESHOLD)

    if characteristics is not None:
        try:
            calibration_info = {
                "characteristics": characteristics,
                "thresholds": {
                    "onset": float(onset_threshold),
                    "frame": float(frame_threshold),
                },
                "defaults": {
                    "onset": float(settings.BASIC_PITCH_ONSET_THRESHOLD),
                    "frame": float(settings.BASIC_PITCH_FRAME_THRESHOLD),
                },
            }
            (work / "threshold_calibration.json").write_text(
                json.dumps(calibration_info, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            _LOG.warning("Failed to write threshold calibration info: %s", exc)

    # --- 1. Basic Pitch Transcription ---
    note_events: list[NoteEvent] = []
    midi_data = None
    if not is_accompaniment and bool(settings.ENABLE_BASIC_PITCH):
        try:
            midi_data, note_events = transcribe_basic_pitch(
                harmonic_path,
                midi_tempo=float(tempo_for_bp),
                onset_threshold=float(onset_threshold),
                frame_threshold=float(frame_threshold),
                minimum_note_length_ms=float(settings.BASIC_PITCH_MIN_NOTE_MS),
                melodia_trick=True,
            )
        except Exception as e:
            print(f"Error executing basic_pitch: {e}")
            note_events = []
            midi_data = None

    note_events_raw = list(note_events)

    # Decide whether to keep beats as-is (double-time) or halved (half-time) based on
    # readability of the quantized score. This keeps `tempo_bpm`, `beat_times`, and
    # downstream exports consistent.
    beat_times_sel = _pick_best_beat_times(note_events, beat_times_raw, time_signature=time_sig)
    tempo_bpm = _tempo_from_beat_times(beat_times_sel)
    if not tempo_bpm or tempo_bpm <= 0:
        tempo_bpm = float(tempo_for_bp)

    beat_times_norm, beat_offset = normalize_beat_times(beat_times_sel)
    note_events = _shift_note_events(note_events, beat_offset)

    # --- 2. Chord Detection ---
    chord_vocab = str(settings.CHORD_VOCAB)
    switch_penalty = float(settings.SWITCH_PENALTY)
    min_segment_sec = float(settings.MIN_SEGMENT_SEC)
    if is_accompaniment:
        chord_vocab = "majmin"
        switch_penalty = max(switch_penalty, float(_ACC_SWITCH_PENALTY))
        min_segment_sec = max(min_segment_sec, float(_ACC_MIN_SEGMENT_SEC))

    chroma, _times, chords = extract_chords_template(
        y_harm,
        sr_trans,
        vocab=chord_vocab,
        switch_penalty=float(switch_penalty),
        min_segment_sec=float(min_segment_sec),
        beat_times=beat_times_sel,
    )

    key_est = estimate_key_madmom(harmonic_path)
    key_sig = None
    if key_est is not None:
        key_sig = KeySignature(
            tonic=key_est.tonic,
            mode=key_est.mode,
            fifths=int(key_est.fifths),
            name=key_est.name,
            vexflow=key_est.vexflow,
            use_flats=bool(key_est.use_flats),
            score=float(key_est.score),
        )

    use_flats = bool(key_sig.use_flats) if key_sig else False
    spelled_chords: list[ChordSegment] = []
    for c in chords:
        spelled_chords.append(
            ChordSegment(
                start=float(c.start),
                end=float(c.end),
                label=spell_chord_label(str(c.label), use_flats=use_flats),
                confidence=float(c.confidence),
            )
        )

    if is_accompaniment:
        spelled_chords = _simplify_chords_for_accompaniment(
            spelled_chords,
            min_duration=float(_ACC_MIN_SEGMENT_SEC),
            min_confidence=float(_ACC_MIN_CONFIDENCE),
        )
    else:
        spelled_chords = _simplify_chord_segments(
            spelled_chords,
            chroma=chroma,
            times=_times,
            min_confidence=float(_SEVENTH_MIN_CONFIDENCE),
            min_duration=float(_SEVENTH_MIN_DURATION),
            seventh_ratio=float(_SEVENTH_RATIO),
        )
    spelled_chords_raw = list(spelled_chords)
    spelled_chords = _shift_chords(spelled_chords, beat_offset)

    # --- Guitar Mode: Hybrid transcription based on content analysis ---
    content_segments: list[ContentSegment] = []
    content_segments_raw: list[ContentSegment] = []
    strum_events_guitar: list[_StrumEvent] = []
    guitar_result: GuitarTranscriptionResult | None = None
    if is_guitar_mode:
        try:
            guitar_notes, strum_events_guitar, content_segments = _run_guitar_mode(
                y_trans,
                sr_trans,
                harmonic_path,
                spelled_chords_raw,
                beat_times_sel,
                tempo_bpm,
                base_note_events=note_events_raw,
                use_flats=use_flats,
                onset_threshold=float(onset_threshold),
                frame_threshold=float(frame_threshold),
                min_note_ms=float(settings.BASIC_PITCH_MIN_NOTE_MS),
                window_sec=float(settings.CONTENT_ANALYSIS_WINDOW_SEC),
                hop_sec=float(settings.CONTENT_ANALYSIS_HOP_SEC),
            )
            # Shift events to account for beat offset
            guitar_notes = _shift_note_events(guitar_notes, beat_offset)
            strum_events_guitar = [
                _StrumEvent(
                    time_s=e.time_s - beat_offset,
                    keys=e.keys,
                    positions=e.positions,
                    pitches=e.pitches,
                )
                for e in strum_events_guitar
            ]
            content_segments_raw = list(content_segments)
            content_segments = _shift_content_segments(content_segments, beat_offset)
            # Merge melodic and chordal results (includes quantization)
            guitar_result = merge_transcription_results(
                guitar_notes,
                strum_events_guitar,
                content_segments,
                spelled_chords,
                tempo_bpm=tempo_bpm,
                beat_times=beat_times_norm,
                time_signature=time_sig,
            )
            note_events = guitar_result.note_events
        except Exception as e:
            print(f"Error in guitar mode: {e}")
            # Fall back to standard note transcription
            is_guitar_mode = False

    if note_events and not is_accompaniment and guitar_result is None:
        note_events = _post_process_note_events(
            note_events,
            chords=spelled_chords,
            tempo_bpm=float(tempo_bpm),
        )

    strum_onsets = np.asarray([], dtype=np.float32)
    strum_events: list[_StrumEvent] = []
    tab_positions: list[list[list[tuple[int, int]]]] | None = None
    segment_shapes: list[tuple[ChordSegment, Shape | None]] = []
    note_events_debug = note_events
    score_m21: object | None = None

    if is_accompaniment:
        try:
            strum_onsets = detect_strum_onsets(
                y_trans,
                sr_trans,
                beat_times=beat_times_sel,
                tempo_bpm=float(tempo_bpm),
                min_interval_s=0.12,
                onset_delta=0.2,
                backtrack=False,
            )
            strum_onsets = (strum_onsets.astype(np.float32) - float(beat_offset)).astype(np.float32)
        except Exception as e:
            print(f"Error detecting strum onsets: {e}")
            strum_onsets = np.asarray([], dtype=np.float32)

        segment_shapes = _assign_shapes(spelled_chords)
        strum_events = _build_strum_events(strum_onsets, segment_shapes, use_flats=use_flats)
        score_data, pickup_quarters, tab_positions = _quantize_strum_events(
            strum_events,
            beat_times=beat_times_norm,
            tempo_bpm=float(tempo_bpm),
            time_signature=time_sig,
            min_grid_q=float(_ACC_MIN_GRID_Q),
        )
        note_events_debug = _strum_events_to_note_events(strum_events, tempo_bpm=float(tempo_bpm))
    elif guitar_result is not None:
        score_data = guitar_result.score_data
        pickup_quarters = float(guitar_result.pickup_quarters)
        score_m21 = guitar_result.score_m21
        tab_positions = guitar_result.tab_positions
        note_events_debug = guitar_result.note_events
    else:
        # --- 3. Quantization to ScoreData ---
        quant_res = quantize_note_events_to_score(
            note_events,
            tempo_bpm=float(tempo_bpm),
            beat_times=beat_times_norm,
            time_signature=time_sig
        )
        score_data = quant_res.score
        pickup_quarters = float(quant_res.pickup_quarters)
        score_m21 = quant_res.score_m21
        tab_positions = quant_res.tab_positions

    # --- Debug artifacts ---
    try:
        beat_payload = {
            "tempo_bpm": float(tempo_bpm),
            "tempo_raw_bpm": float(tempo_raw or 0.0),
            "beat_times_s": (beat_times_norm.tolist() if beat_times_norm is not None else []),
            "beat_times_raw_s": (beat_times_sel.tolist() if beat_times_sel is not None else []),
            "beat_offset_s": float(beat_offset),
            "beat_source": beat_source,
            "transcription_source": transcription_source,
            "transcription_mode": transcription_mode,
            "demucs_enabled": bool(settings.ENABLE_DEMUCS),
            "demucs_error": demucs_error,
        }
        (out / "beat_times.json").write_text(
            json.dumps(beat_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        save_note_events_csv(note_events_debug, out / "note_events.csv")
        chords_payload = [c.model_dump() for c in spelled_chords]
        (out / "chords.json").write_text(
            json.dumps(chords_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        segments_to_dump = content_segments_raw or content_segments
        if segments_to_dump:
            segments_payload = []
            for seg in segments_to_dump:
                segments_payload.append(
                    {
                        "start_time_s": float(seg.start_time_s),
                        "end_time_s": float(seg.end_time_s),
                        "content_type": str(seg.content_type),
                        "confidence": float(seg.confidence),
                        "metrics": dict(seg.metrics or {}),
                    }
                )
            (out / "content_segments.json").write_text(
                json.dumps(segments_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        if is_accompaniment:
            (out / "strum_onsets.json").write_text(
                json.dumps({"onsets_s": strum_onsets.tolist()}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            shapes_payload = []
            for seg, shape in segment_shapes:
                shapes_payload.append(
                    {
                        "start": float(seg.start),
                        "end": float(seg.end),
                        "label": str(seg.label),
                        "confidence": float(seg.confidence),
                        "shape": (shape_to_dict(shape) if shape is not None else None),
                    }
                )
            (out / "chosen_shapes.json").write_text(
                json.dumps(shapes_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
    except Exception as e:
        print(f"Error writing debug artifacts: {e}")

    title = _job_title(job_dir, input_path)
    musicxml_path = out / "result.musicxml"

    # --- 4. Export MusicXML (Full Lead Sheet with Tab) ---
    score_payload = score_m21 if (score_m21 is not None and tab_positions is None) else score_data
    export_musicxml(
        musicxml_path,
        score_payload,
        tempo_bpm=float(tempo_bpm),
        time_signature=time_sig,
        key_signature_fifths=(key_sig.fifths if key_sig else None),
        title=title,
        instrument="guitar",
        chords=[Segment(c.start, c.end, c.label, c.confidence) for c in spelled_chords],
        beat_times=beat_times_norm,
        pickup_quarters=float(pickup_quarters),
        slash_notation=bool(is_accompaniment),
        tab_positions=tab_positions,
        midi_path=(out / "transcription.mid"),
    )

    # PDF (LilyPond) generation - kept as fallback/supplement
    # Ideally LilyPond service should also be updated to support full score,
    # but for now we focus on MusicXML/OSMD as requested.
    pdf_error = None
    if shutil.which("lilypond") is not None:
        try:
            ly = build_lilypond_score(
                spelled_chords,
                tempo_bpm=float(tempo_bpm),
                time_signature=time_sig,
                key_tonic=(key_sig.tonic if key_sig else None),
                key_mode=(key_sig.mode if key_sig else "major"),
                title=title,
            )
            render_lilypond_pdf(ly, out, basename="score")
        except Exception as e:
            pdf_error = f"No se pudo generar PDF (LilyPond): {e}"

    backend_name = "basic_pitch+chords_viterbi"
    if guitar_result is not None:
        backend_name = "guitar_hybrid+chords_viterbi"
    if is_accompaniment:
        backend_name = "accompaniment+chords_viterbi"

    return JobResult(
        job_id=job_dir.name,
        tempo_bpm=float(tempo_bpm),
        time_signature=time_sig,
        key_signature=key_sig,
        chords=spelled_chords,
        transcription_backend=backend_name,
        transcription_error=pdf_error,
        score=score_data,
    )
