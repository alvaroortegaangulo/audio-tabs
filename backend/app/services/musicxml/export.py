from __future__ import annotations

from pathlib import Path
from typing import List, Optional
import numpy as np

from music21 import chord as m21chord
from music21 import instrument as m21instrument
from music21 import key as m21key
from music21 import meter as m21meter
from music21 import note as m21note
from music21 import stream as m21stream
from music21 import tempo as m21tempo
from music21 import harmony as m21harmony
from music21 import clef as m21clef
from music21 import layout as m21layout
from music21 import articulations as m21articulations
from music21 import duration as m21duration
from music21 import interval as m21interval
from music21 import pitch as m21pitch

from app.schemas import ScoreData
from app.services.chords.template import Segment

_DURATION_Q = {
    "w": 4.0,
    "h": 2.0,
    "q": 1.0,
    "8": 0.5,
    "16": 0.25,
    "32": 0.125,
}

_DURATION_M21 = {
    "w": "whole",
    "h": "half",
    "q": "quarter",
    "8": "eighth",
    "16": "16th",
    "32": "32nd",
}


def _m21_duration(duration: str, dots: int, tuplet: tuple[int, int] | None) -> m21duration.Duration:
    """
    Construye un Duration explícito para evitar que music21 infiera duraciones
    "complex" (no exportables a MusicXML) por errores de redondeo.
    """
    d_type = _DURATION_M21.get(duration)
    if d_type is None:
        raise ValueError(f"Unsupported duration: {duration}")

    d = m21duration.Duration(d_type)
    d.dots = int(dots or 0)

    if tuplet is not None:
        num_notes, notes_occupied = tuplet
        # Ej: 3:2 para corcheas tresillo (cada nota = 1/3 de negra)
        d.appendTuplet(m21duration.Tuplet(int(num_notes), int(notes_occupied)))

    return d


def _quarter_length(duration: str, dots: int, tuplet: tuple[int, int] | None) -> float:
    base = _DURATION_Q.get(duration)
    if base is None:
        raise ValueError(f"Unsupported duration: {duration}")

    q = float(base)
    if dots:
        add = q / 2.0
        for _ in range(int(dots)):
            q += add
            add /= 2.0

    if tuplet is not None:
        num_notes, notes_occupied = tuplet
        q *= float(notes_occupied) / float(num_notes)

    return float(q)


def _vf_key_to_m21_pitch(key: str) -> str:
    """
    VexFlow key: 'c#/4' -> music21 pitch: 'C#4'
    """
    pitch, octave = key.split("/")
    pitch = pitch.strip()
    octave = octave.strip()
    if not pitch or not octave:
        raise ValueError(f"Invalid vexflow key: {key}")
    return pitch[0].upper() + pitch[1:] + octave


def _get_preferred_tab_position(midi_note: int) -> tuple[int, int]:
    """
    Returns (string, fret) for a given MIDI note.
    Prioritizes open position (first 5 frets).
    Standard Tuning: E2(40), A2(45), D3(50), G3(55), B3(59), E4(64).
    """
    tuning = [(6, 40), (5, 45), (4, 50), (3, 55), (2, 59), (1, 64)]

    max_fret = 24  # cover up to E6 on a 24-fret guitar

    candidates = []
    for string_num, open_pitch in tuning:
        fret = midi_note - open_pitch
        if 0 <= fret <= max_fret:
            candidates.append((string_num, int(fret)))

    if not candidates:
        # Fallback for very high/low notes not covered (unlikely for guitar range)
        return (0, 0)

    # Prioritize Open Position (Frets 0-4 usually, let's say 0-5)
    # Strategy:
    # 1. Look for open strings (fret 0)
    # 2. Look for frets 1-5
    # 3. Use lowest fret available otherwise.

    # Check for 0 (Open strings)
    zeros = [c for c in candidates if c[1] == 0]
    if zeros:
        return zeros[0] # Prefer lower string number? doesn't matter much for same note

    # Check for frets 1-5
    low_pos = [c for c in candidates if 1 <= c[1] <= 5]
    if low_pos:
        # If multiple, prefer the one on the lower string (higher string_num) -> thicker string?
        # Or higher string (lower string_num) -> brighter tone?
        # For beginners, maybe lower fret is king.
        return min(low_pos, key=lambda x: x[1])

    # Otherwise pick lowest fret available
    return min(candidates, key=lambda x: x[1])


def _chord_label_to_figure(label: str) -> str | None:
    """
    Convert internal labels like "A:min7" into a standard chord figure that
    music21 can parse (and export) into valid MusicXML <harmony> kinds.
    """
    if not label or label == "N":
        return None

    if ":" in label:
        root, qual = label.split(":", 1)
        root = root.strip()
        qual = qual.strip().lower()
    else:
        root, qual = label.strip(), "maj"

    if not root:
        return None

    # music21 prefers flats as "-" (e.g. "B-"), but also parses common "b".
    # Use "-" to be safe for MusicXML export.
    root_m21 = root.replace("b", "-")

    if qual in ("maj", ""):
        suffix = ""
    elif qual in ("min", "m"):
        suffix = "m"
    elif qual == "7":
        suffix = "7"
    elif qual in ("min7", "m7"):
        suffix = "m7"
    elif qual in ("maj7",):
        suffix = "maj7"
    else:
        suffix = ""

    return f"{root_m21}{suffix}"


def export_musicxml(
    out_path: Path,
    score_data: ScoreData,
    *,
    tempo_bpm: float,
    time_signature: str = "4/4",
    key_signature_fifths: int | None = None,
    title: str = "Transcription",
    instrument: str = "piano",
    chords: List[Segment] | None = None,
    beat_times: np.ndarray | None = None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    score = m21stream.Score()
    score.metadata = None  # Could add title here if needed

    # Part 1: Standard Notation + Chords
    part_notation = m21stream.Part()
    part_notation.id = "P1"

    # Instrument Setup
    # Note: m21instrument.Guitar() implies transposition (sounding vs written).
    # If we manually transpose notes, we should be careful.
    # However, setting it to Guitar helps OSMD pick the right icon/sound.
    inst_not = m21instrument.Guitar() if instrument == "guitar" else m21instrument.Piano()
    part_notation.insert(0, inst_not)

    part_notation.insert(0, m21meter.TimeSignature(time_signature))
    if key_signature_fifths is not None:
        part_notation.insert(0, m21key.KeySignature(int(key_signature_fifths)))
    part_notation.insert(0, m21tempo.MetronomeMark(number=float(tempo_bpm)))

    # Part 2: Tablature
    part_tab = m21stream.Part()
    part_tab.id = "P2"
    # Setup Tab Staff
    tab_clef = m21clef.TabClef()
    staff_layout = m21layout.StaffLayout(staffLines=6)
    part_tab.insert(0, tab_clef)
    part_tab.insert(0, staff_layout)
    inst_tab = m21instrument.Guitar()
    part_tab.insert(0, inst_tab)

    part_tab.insert(0, m21meter.TimeSignature(time_signature))
    if key_signature_fifths is not None:
        part_tab.insert(0, m21key.KeySignature(int(key_signature_fifths)))

    # Calculate measure duration for chord mapping
    try:
        ts_num, ts_den = map(int, time_signature.split("/"))
    except:
        ts_num, ts_den = 4, 4

    quarter_length_per_measure = float(ts_num) * (4.0 / float(ts_den))
    beat_quarters = 4.0 / float(ts_den)

    sec_per_q = 60.0 / float(tempo_bpm if tempo_bpm else 120.0)
    seconds_per_measure = sec_per_q * quarter_length_per_measure
    seconds_per_beat = sec_per_q * beat_quarters
    chords_sorted = sorted(chords or [], key=lambda c: float(c.start))

    # Prepare beat lookup for chords if beat_times is available
    beat_times_lookup = None
    if beat_times is not None and len(beat_times) > 1:
        # We need a function that maps (measure_index, beat_index) -> time_s
        beat_times_lookup = np.array(beat_times)

        # If beat_times ends early, we extrapolate
        avg_dur = float(np.mean(np.diff(beat_times_lookup)))
        if avg_dur <= 0: avg_dur = 0.5

        def get_time_at_beat_abs(beat_abs: float) -> float:
            idx = int(beat_abs)
            frac = beat_abs - idx

            # Simple linear interp
            if idx < 0:
                # extrapolate left
                return beat_times_lookup[0] + beat_abs * avg_dur
            if idx >= len(beat_times_lookup) - 1:
                # extrapolate right
                return beat_times_lookup[-1] + (beat_abs - (len(beat_times_lookup)-1)) * avg_dur

            return beat_times_lookup[idx] + frac * (beat_times_lookup[idx+1] - beat_times_lookup[idx])
    else:
        # Fallback to constant tempo
        def get_time_at_beat_abs(beat_abs: float) -> float:
            return beat_abs * seconds_per_beat


    def get_chords_for_measure(measure_idx: int) -> List[tuple[float, str]]:
        if not chords_sorted:
            return []

        # Find time range for this measure using beats
        # Measure start beat (0-indexed) = measure_idx * ts_num (assuming beat=quarter for denom=4)
        # Actually ts_num is beats per measure.
        beats_per_measure = float(ts_num)
        start_beat_abs = measure_idx * beats_per_measure
        end_beat_abs = (measure_idx + 1) * beats_per_measure

        m_start_time = get_time_at_beat_abs(start_beat_abs)
        m_end_time = get_time_at_beat_abs(end_beat_abs)

        # IMPORTANTE: Insertar ChordSymbol en offsets arbitrarios hace que music21
        # parta notas/rests en duraciones no expresables (e.g. 2048th), rompiendo
        # la exportación a MusicXML. Cuantizamos a inicios de beat.
        def label_at(t: float) -> str:
            for c in chords_sorted:
                if float(c.end) <= t:
                    continue
                if float(c.start) <= t < float(c.end):
                    return str(c.label or "N")
                if float(c.start) > t:
                    break
            return "N"

        by_beat: dict[int, str] = {}

        # Chord al inicio del compás (si aplica)
        start_lbl = label_at(m_start_time + 1e-4) # small epsilon
        if start_lbl != "N":
            by_beat[0] = start_lbl

        # New strategy: Query the chord at each beat start
        for b in range(ts_num):
            beat_time = get_time_at_beat_abs(start_beat_abs + b)
            # Add a small offset to be inside the beat
            lbl = label_at(beat_time + 0.05)
            if lbl != "N":
                by_beat[b] = lbl

        out: list[tuple[float, str]] = []
        prev = ""
        for b in range(ts_num):
            lbl = by_beat.get(b)
            if not lbl:
                continue
            if b == 0 or lbl != prev:
                # offset in quarters
                out.append((float(b) * beat_quarters, lbl))
            prev = lbl

        return out

    # Transposition interval for Guitar Notation (Sounding -> Written: Up 1 octave)
    transpose_interval = m21interval.Interval('P8')

    for i, meas in enumerate(score_data.measures):
        # --- Standard Staff Measure ---
        m_not = m21stream.Measure(number=int(meas.number))

        # Inject Chords
        measure_chords = get_chords_for_measure(i)
        for offset, label in measure_chords:
            try:
                fig = _chord_label_to_figure(str(label))
                if fig:
                    h = m21harmony.ChordSymbol(fig)
                    m_not.insert(float(offset), h)
            except Exception:
                pass

        # --- Tab Staff Measure ---
        m_tab = m21stream.Measure(number=int(meas.number))

        offset_ql = 0.0
        for item in meas.items:
            tuplet = None
            if item.tuplet is not None:
                tuplet = (int(item.tuplet.num_notes), int(item.tuplet.notes_occupied))
            dur = _m21_duration(item.duration, int(item.dots), tuplet)
            dur_ql = float(dur.quarterLength)

            if item.rest or not item.keys:
                obj_not = m21note.Rest()
                obj_not.duration = dur
                obj_tab = m21note.Rest()
                obj_tab.duration = dur
            else:
                pitches_str = [_vf_key_to_m21_pitch(k) for k in item.keys]

                # --- Standard Notation Object ---
                # We need to transpose pitches up by an octave for standard guitar notation
                written_pitches = []
                for p_str in pitches_str:
                    p = m21pitch.Pitch(p_str)
                    p.transpose(transpose_interval, inPlace=True)
                    written_pitches.append(p)

                if len(written_pitches) == 1:
                    obj_not = m21note.Note(written_pitches[0])
                    obj_not.duration = dur
                else:
                    obj_not = m21chord.Chord(written_pitches)
                    obj_not.duration = dur

                obj_tab = None

                # --- Tab Notation Objects ---
                # Use original (sounding) pitches to calculate fret positions.
                # NOTE: music21 chord.Chord does not reliably export per-note
                # technical indications (string/fret). Insert individual Notes
                # at the same offset so each <note> gets its own <technical>.
                sounding_pitches = [m21pitch.Pitch(p_str) for p_str in pitches_str]
                for p in sounding_pitches:
                    n_tab = m21note.Note(p)
                    n_tab.duration = dur
                    s, f = _get_preferred_tab_position(int(p.midi))
                    if s > 0:
                        n_tab.articulations.append(m21articulations.StringIndication(int(s)))
                        n_tab.articulations.append(m21articulations.FretIndication(int(f)))
                    m_tab.insert(float(offset_ql), n_tab)

            m_not.insert(float(offset_ql), obj_not)
            if obj_tab is not None:
                m_tab.insert(float(offset_ql), obj_tab)

            offset_ql += dur_ql

        part_notation.append(m_not)
        part_tab.append(m_tab)

    try:
        part_notation.makeBeams(inPlace=True)
        part_tab.makeBeams(inPlace=True)
    except Exception:
        pass

    score.insert(0, part_notation)
    score.insert(0, part_tab)

    # Add StaffGroup to link them visually
    # barlineSpan=True ensures barlines go through both staves
    staff_group = m21layout.StaffGroup(
        [part_notation, part_tab],
        name="Guitar",
        abbreviation="Gtr.",
        symbol="bracket"
    )
    staff_group.barlineSpan = True
    score.insert(0, staff_group)

    score.write("musicxml", fp=str(out_path))
