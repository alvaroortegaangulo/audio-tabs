from __future__ import annotations

from pathlib import Path
from typing import List, Optional

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

    candidates = []
    for string_num, open_pitch in tuning:
        fret = midi_note - open_pitch
        if 0 <= fret <= 19:  # Assuming 19 frets max for standard view
            candidates.append((string_num, fret))

    if not candidates:
        # Fallback for very high/low notes not covered (unlikely for guitar range)
        return (0, 0)

    # Prefer low frets (<= 5)
    low_pos = [c for c in candidates if c[1] <= 5]
    if low_pos:
        # Pick the one with the lowest fret number
        return min(low_pos, key=lambda x: x[1])

    # Otherwise pick lowest fret available
    return min(candidates, key=lambda x: x[1])


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
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    score = m21stream.Score()
    score.metadata = None  # Could add title here if needed

    # Part 1: Standard Notation + Chords
    part_notation = m21stream.Part()
    part_notation.id = "P1"
    part_notation.insert(0, m21instrument.Guitar() if instrument == "guitar" else m21instrument.Piano())
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
    part_tab.insert(0, m21meter.TimeSignature(time_signature))
    if key_signature_fifths is not None:
        part_tab.insert(0, m21key.KeySignature(int(key_signature_fifths)))

    # Calculate measure duration for chord mapping
    # 4/4 time -> 4 beats per measure.
    # Note: This is a simplification assuming 4/4.
    # Real robustness requires parsing time_signature string.
    try:
        ts_num, ts_den = map(int, time_signature.split("/"))
    except:
        ts_num, ts_den = 4, 4

    seconds_per_beat = 60.0 / tempo_bpm
    seconds_per_measure = seconds_per_beat * ts_num # e.g. 4 beats
    quarter_length_per_measure = float(ts_num) * (4.0 / float(ts_den)) # e.g. 4 * (4/4) = 4.0

    # Helper to find chords in a measure
    def get_chords_for_measure(measure_idx: int) -> List[tuple[float, str]]:
        if not chords:
            return []

        m_start_time = measure_idx * seconds_per_measure
        m_end_time = m_start_time + seconds_per_measure

        found = []
        for c in chords:
            # Check overlap or containment
            # We treat chords as events at their start time mostly for placement
            if m_start_time <= c.start < m_end_time:
                # Calculate offset in quarter lengths
                rel_time = c.start - m_start_time
                offset_q = (rel_time / seconds_per_measure) * quarter_length_per_measure
                found.append((offset_q, c.label))
        return found

    for i, meas in enumerate(score_data.measures):
        # --- Standard Staff Measure ---
        m_not = m21stream.Measure(number=int(meas.number))

        # Inject Chords
        measure_chords = get_chords_for_measure(i)
        for offset, label in measure_chords:
            # Label format "C:maj", "A#:min"
            try:
                if ":" in label:
                    root_str, kind_str = label.split(":")
                    # Map kind to music21 compatibility if needed
                    # basic types: maj, min, dim, aug, etc.
                    # music21 understands many.
                    if kind_str == "7": kind_str = "dominant" # "C:7" -> dominant

                    # Create ChordSymbol
                    h = m21harmony.ChordSymbol(root=root_str, kind=kind_str)
                    # Insert at specific offset
                    # Note: inserting directly into Measure at offset requires care with existing elements?
                    # m21stream.Measure.insert(offset, obj) works.
                    m_not.insert(offset, h)
            except Exception:
                pass # Ignore invalid chord parsing

        # --- Tab Staff Measure ---
        m_tab = m21stream.Measure(number=int(meas.number))

        for item in meas.items:
            tuplet = None
            if item.tuplet is not None:
                tuplet = (int(item.tuplet.num_notes), int(item.tuplet.notes_occupied))
            ql = _quarter_length(item.duration, int(item.dots), tuplet)

            # Create objects for both staves
            if item.rest or not item.keys:
                obj_not = m21note.Rest(quarterLength=ql)
                obj_tab = m21note.Rest(quarterLength=ql)
            else:
                pitches_str = [_vf_key_to_m21_pitch(k) for k in item.keys]

                # Standard Notation Object
                if len(pitches_str) == 1:
                    obj_not = m21note.Note(pitches_str[0], quarterLength=ql)
                else:
                    obj_not = m21chord.Chord(pitches_str, quarterLength=ql)

                # Tab Notation Object
                # For Tab, we need separate notes to attach string/fret info effectively
                # or a Chord with individual note heads having articulations.
                if len(pitches_str) == 1:
                    obj_tab = m21note.Note(pitches_str[0], quarterLength=ql)
                    s, f = _get_preferred_tab_position(obj_tab.pitch.midi)
                    if s > 0:
                        obj_tab.articulations.append(m21articulations.StringIndication(s))
                        obj_tab.articulations.append(m21articulations.FretIndication(f))
                else:
                    obj_tab = m21chord.Chord(pitches_str, quarterLength=ql)
                    # For chords, we iterate notes and add articulations?
                    # Music21 chord structure is complex for tab.
                    # Often easier to represent as a chord, and music21 might serialize it
                    # if we attach string/fret to the notes inside the chord.
                    for cn in obj_tab.notes:
                        s, f = _get_preferred_tab_position(cn.pitch.midi)
                        if s > 0:
                            cn.articulations.append(m21articulations.StringIndication(s))
                            cn.articulations.append(m21articulations.FretIndication(f))

            m_not.append(obj_not)
            m_tab.append(obj_tab)

        part_notation.append(m_not)
        part_tab.append(m_tab)

    try:
        part_notation.makeBeams(inPlace=True)
        part_tab.makeBeams(inPlace=True)
    except Exception:
        pass

    # Group the parts
    score.insert(0, part_notation)
    score.insert(0, part_tab)

    # Add StaffGroup to link them visually
    staff_group = m21layout.StaffGroup(
        [part_notation, part_tab],
        name="Guitar",
        abbreviation="Gtr.",
        symbol="bracket"
    )
    score.insert(0, staff_group)

    score.write("musicxml", fp=str(out_path))
