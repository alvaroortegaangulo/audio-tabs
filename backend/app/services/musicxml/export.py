from __future__ import annotations

from pathlib import Path

from music21 import chord as m21chord
from music21 import instrument as m21instrument
from music21 import key as m21key
from music21 import meter as m21meter
from music21 import note as m21note
from music21 import stream as m21stream
from music21 import tempo as m21tempo

from app.schemas import ScoreData


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


def export_musicxml(
    out_path: Path,
    score_data: ScoreData,
    *,
    tempo_bpm: float,
    time_signature: str = "4/4",
    key_signature_fifths: int | None = None,
    title: str = "Transcription",
    instrument: str = "piano",
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    score = m21stream.Score()
    score.metadata = None

    part = m21stream.Part()
    if instrument == "guitar":
        part.insert(0, m21instrument.Guitar())
    else:
        part.insert(0, m21instrument.Piano())

    part.insert(0, m21meter.TimeSignature(time_signature))
    if key_signature_fifths is not None:
        part.insert(0, m21key.KeySignature(int(key_signature_fifths)))
    part.insert(0, m21tempo.MetronomeMark(number=float(tempo_bpm)))

    for meas in score_data.measures:
        m = m21stream.Measure(number=int(meas.number))
        for item in meas.items:
            tuplet = None
            if item.tuplet is not None:
                tuplet = (int(item.tuplet.num_notes), int(item.tuplet.notes_occupied))
            ql = _quarter_length(item.duration, int(item.dots), tuplet)

            if item.rest or not item.keys:
                obj = m21note.Rest(quarterLength=ql)
            else:
                pitches = [_vf_key_to_m21_pitch(k) for k in item.keys]
                if len(pitches) == 1:
                    obj = m21note.Note(pitches[0], quarterLength=ql)
                else:
                    obj = m21chord.Chord(pitches, quarterLength=ql)

            m.append(obj)
        part.append(m)

    try:
        part.makeBeams(inPlace=True)
    except Exception:
        pass

    score.append(part)
    score.write("musicxml", fp=str(out_path))

