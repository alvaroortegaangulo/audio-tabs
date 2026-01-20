from __future__ import annotations
from pathlib import Path
from typing import List
from music21 import stream, meter, tempo as m21tempo, harmony, note, instrument

from app.schemas import ChordSegment

def export_musicxml(
    out_path: Path,
    chords: List[ChordSegment],
    tempo_bpm: float,
    time_signature: str = "4/4",
    title: str = "Chord Extraction"
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    score = stream.Score()
    score.metadata = None

    part = stream.Part()
    part.insert(0, instrument.Guitar())
    part.insert(0, meter.TimeSignature(time_signature))
    part.insert(0, m21tempo.MetronomeMark(number=tempo_bpm))

    # Duración total aproximada
    total_sec = max((c.end for c in chords), default=0.0)
    # Aprox beats
    beats_per_sec = tempo_bpm / 60.0
    total_beats = max(1, int(total_sec * beats_per_sec) + 1)

    beats_per_measure = int(time_signature.split("/")[0])
    num_measures = int((total_beats + beats_per_measure - 1) / beats_per_measure)

    # Crea compases con silencios
    for m in range(num_measures):
        meas = stream.Measure(number=m+1)
        meas.append(note.Rest(quarterLength=beats_per_measure))
        part.append(meas)

    # Inserta acordes como "harmony" en offsets de compás
    for c in chords:
        if c.label == "N":
            continue
        start_beat = (c.start * beats_per_sec)
        measure_idx = int(start_beat // beats_per_measure)
        offset_in_measure = float(start_beat - measure_idx * beats_per_measure)

        # clamp
        if measure_idx < 0:
            continue
        if measure_idx >= len(part.getElementsByClass(stream.Measure)):
            continue

        cs = harmony.ChordSymbol(c.label.replace(":", ""))  # "Cmaj", "Amin", "E7"
        part.measure(measure_idx+1).insert(offset_in_measure, cs)

    score.append(part)
    score.write("musicxml", fp=str(out_path))
