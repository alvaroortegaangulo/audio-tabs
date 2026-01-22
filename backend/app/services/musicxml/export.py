from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np

from app.schemas import ScoreData
from app.services.chords.template import Segment

_DURATION_M21 = {
    "w": "whole",
    "h": "half",
    "q": "quarter",
    "8": "eighth",
    "16": "16th",
    "32": "32nd",
}


def _m21_duration(duration: str, dots: int, tuplet: tuple[int, int] | None):
    from music21 import duration as m21duration

    d_type = _DURATION_M21.get(duration)
    if d_type is None:
        raise ValueError(f"Unsupported duration: {duration}")

    d = m21duration.Duration(d_type)
    d.dots = int(dots or 0)
    if tuplet is not None:
        num_notes, notes_occupied = tuplet
        d.appendTuplet(m21duration.Tuplet(int(num_notes), int(notes_occupied)))
    return d


def _vf_key_to_m21_pitch(key: str) -> str:
    pitch, octave = key.split("/")
    pitch = pitch.strip()
    octave = octave.strip()
    if not pitch or not octave:
        raise ValueError(f"Invalid vexflow key: {key}")
    return pitch[0].upper() + pitch[1:] + octave


def _chord_label_to_figure(label: str) -> str | None:
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


def _time_to_beats(t_sec: float, beat_times: np.ndarray) -> float:
    beats = np.asarray(beat_times, dtype=np.float64)
    beats = beats[np.isfinite(beats)]
    beats = np.sort(beats)
    indices = np.arange(len(beats), dtype=np.float64)
    avg_dur = float(np.mean(np.diff(beats))) if len(beats) > 1 else 0.5
    if avg_dur <= 0:
        avg_dur = 0.5
    res = float(np.interp([t_sec], beats, indices, left=-1.0, right=-1.0)[0])
    if t_sec < beats[0]:
        res = indices[0] - (beats[0] - t_sec) / avg_dur
    elif t_sec > beats[-1]:
        res = indices[-1] + (t_sec - beats[-1]) / avg_dur
    return float(res)


def _ensure_part_header(
    part,
    *,
    tempo_bpm: float,
    time_signature: str,
    key_signature_fifths: int | None,
    instrument: str,
) -> None:
    from music21 import instrument as m21instrument
    from music21 import key as m21key
    from music21 import meter as m21meter
    from music21 import tempo as m21tempo

    if not part.getElementsByClass(m21instrument.Instrument):
        inst = m21instrument.Guitar() if instrument == "guitar" else m21instrument.Piano()
        part.insert(0, inst)
    if not part.getElementsByClass(m21meter.TimeSignature):
        part.insert(0, m21meter.TimeSignature(time_signature))
    if key_signature_fifths is not None and not part.getElementsByClass(m21key.KeySignature):
        part.insert(0, m21key.KeySignature(int(key_signature_fifths)))
    if not part.getElementsByClass(m21tempo.MetronomeMark):
        part.insert(0, m21tempo.MetronomeMark(number=float(tempo_bpm)))


def _score_from_score_data(
    score_data: ScoreData,
    *,
    tempo_bpm: float,
    time_signature: str,
    key_signature_fifths: int | None,
    instrument: str,
    slash_notation: bool,
    tab_positions: list[list[list[tuple[int, int]]]] | None,
) -> object:
    from music21 import articulations as m21articulations
    from music21 import chord as m21chord
    from music21 import clef as m21clef
    from music21 import layout as m21layout
    from music21 import note as m21note
    from music21 import pitch as m21pitch
    from music21 import stream as m21stream
    from music21 import tie as m21tie

    score = m21stream.Score()
    part_notation = m21stream.Part()
    _ensure_part_header(
        part_notation,
        tempo_bpm=float(tempo_bpm),
        time_signature=time_signature,
        key_signature_fifths=key_signature_fifths,
        instrument=instrument,
    )

    part_tab = None
    if tab_positions is not None:
        part_tab = m21stream.Part()
        part_tab.insert(0, m21clef.TabClef())
        part_tab.insert(0, m21layout.StaffLayout(staffLines=6))
        _ensure_part_header(
            part_tab,
            tempo_bpm=float(tempo_bpm),
            time_signature=time_signature,
            key_signature_fifths=key_signature_fifths,
            instrument="guitar",
        )

    for i, meas in enumerate(score_data.measures):
        m_not = m21stream.Measure(number=int(meas.number))
        m_tab = m21stream.Measure(number=int(meas.number)) if part_tab is not None else None
        measure_positions = None
        if tab_positions is not None and i < len(tab_positions):
            measure_positions = tab_positions[i]

        offset_ql = 0.0
        for item_idx, item in enumerate(meas.items):
            tuplet = None
            if item.tuplet is not None:
                tuplet = (int(item.tuplet.num_notes), int(item.tuplet.notes_occupied))
            dur = _m21_duration(item.duration, int(item.dots), tuplet)
            dur_ql = float(dur.quarterLength)

            if item.rest or not item.keys:
                obj_not = m21note.Rest()
                obj_not.duration = dur
                m_not.insert(float(offset_ql), obj_not)
                if m_tab is not None:
                    obj_tab = m21note.Rest()
                    obj_tab.duration = dur
                    m_tab.insert(float(offset_ql), obj_tab)
                offset_ql += dur_ql
                continue

            pitches_str = [_vf_key_to_m21_pitch(k) for k in item.keys]
            written_pitches = [m21pitch.Pitch(p) for p in pitches_str]

            if len(written_pitches) == 1:
                obj_not = m21note.Note(written_pitches[0])
            else:
                obj_not = m21chord.Chord(written_pitches)
            obj_not.duration = dur

            if slash_notation:
                if isinstance(obj_not, m21note.Note):
                    obj_not.notehead = "slash"
                else:
                    for n in obj_not.notes:
                        n.notehead = "slash"

            tie_obj = None
            if item.tie is not None:
                try:
                    tie_obj = m21tie.Tie(str(item.tie))
                except Exception:
                    tie_obj = None
            if tie_obj is not None:
                if isinstance(obj_not, m21note.Note):
                    obj_not.tie = tie_obj
                else:
                    for n in obj_not.notes:
                        n.tie = tie_obj

            m_not.insert(float(offset_ql), obj_not)

            if m_tab is not None:
                positions = None
                if measure_positions is not None and item_idx < len(measure_positions):
                    override = measure_positions[item_idx]
                    if override and len(override) == len(pitches_str):
                        positions = list(override)
                if positions is not None:
                    for pitch_str, (s, f) in zip(pitches_str, positions):
                        n_tab = m21note.Note(m21pitch.Pitch(pitch_str))
                        n_tab.duration = dur
                        if tie_obj is not None:
                            n_tab.tie = tie_obj
                        if s > 0:
                            n_tab.articulations.append(m21articulations.StringIndication(int(s)))
                            n_tab.articulations.append(m21articulations.FretIndication(int(f)))
                        m_tab.insert(float(offset_ql), n_tab)
                else:
                    obj_tab = m21note.Rest()
                    obj_tab.duration = dur
                    m_tab.insert(float(offset_ql), obj_tab)

            offset_ql += dur_ql

        part_notation.append(m_not)
        if part_tab is not None and m_tab is not None:
            part_tab.append(m_tab)

    score.insert(0, part_notation)
    if part_tab is not None:
        score.insert(0, part_tab)

        from music21 import layout as m21layout

        staff_group = m21layout.StaffGroup(
            [part_notation, part_tab],
            name="Guitar",
            abbreviation="Gtr.",
            symbol="bracket",
        )
        staff_group.barlineSpan = True
        score.insert(0, staff_group)

    return score


def _inject_metadata(score, *, title: str) -> None:
    from music21 import metadata as m21metadata

    if score.metadata is None:
        score.metadata = m21metadata.Metadata()
    score.metadata.title = title
    score.metadata.composer = "Audio Tabs AI"


def _inject_chords(
    score,
    chords: List[Segment] | None,
    *,
    tempo_bpm: float,
    beat_times: np.ndarray | None,
    pickup_quarters: float,
) -> None:
    if not chords:
        return
    try:
        part = score.parts[0]
    except Exception:
        return

    sec_per_q = 60.0 / float(tempo_bpm if tempo_bpm else 120.0)
    beat_times_arr = None
    if beat_times is not None and len(beat_times) > 1:
        beat_times_arr = np.asarray(beat_times, dtype=np.float32)

    from music21 import harmony as m21harmony

    for seg in sorted(chords, key=lambda c: float(c.start)):
        label = str(seg.label or "N")
        fig = _chord_label_to_figure(label)
        if not fig:
            continue
        t_sec = float(seg.start)
        if beat_times_arr is not None:
            beat_pos = _time_to_beats(t_sec, beat_times_arr)
            offset_q = float(beat_pos)
        else:
            offset_q = float(t_sec) / float(sec_per_q)
        offset_q += float(pickup_quarters or 0.0)
        cs = m21harmony.ChordSymbol(fig)
        part.insert(float(max(0.0, offset_q)), cs)


def export_musicxml(
    out_path: Path,
    score_data: ScoreData | object,
    *,
    tempo_bpm: float,
    time_signature: str = "4/4",
    key_signature_fifths: int | None = None,
    title: str = "Transcription",
    instrument: str = "piano",
    chords: List[Segment] | None = None,
    beat_times: np.ndarray | None = None,
    pickup_quarters: float = 0.0,
    slash_notation: bool = False,
    tab_positions: list[list[list[tuple[int, int]]]] | None = None,
    midi_path: Path | None = None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    score = None
    try:
        from music21 import stream as m21stream

        if isinstance(score_data, m21stream.Score):
            score = score_data
    except Exception:
        score = None

    if score is None:
        score = _score_from_score_data(
            score_data,
            tempo_bpm=float(tempo_bpm),
            time_signature=time_signature,
            key_signature_fifths=key_signature_fifths,
            instrument=instrument,
            slash_notation=bool(slash_notation),
            tab_positions=tab_positions,
        )
    else:
        try:
            part = score.parts[0]
            _ensure_part_header(
                part,
                tempo_bpm=float(tempo_bpm),
                time_signature=time_signature,
                key_signature_fifths=key_signature_fifths,
                instrument=instrument,
            )
        except Exception:
            pass

    _inject_metadata(score, title=title)
    _inject_chords(
        score,
        chords,
        tempo_bpm=float(tempo_bpm),
        beat_times=beat_times,
        pickup_quarters=float(pickup_quarters or 0.0),
    )

    score.write("musicxml", fp=str(out_path))
    if midi_path is not None:
        midi_path.parent.mkdir(parents=True, exist_ok=True)
        score.write("midi", fp=str(midi_path))
