from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional

from app.schemas import ChordSegment
from app.services.theory.key import NOTE_TO_PC


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


def _pitches_for_label(label: str, *, root_octave: int = 3) -> Optional[list[int]]:
    """
    label backend: "G:maj", "A:min", "D:7", "N"
    Returns MIDI note numbers for a simple block chord.
    """
    if not label or label == "N":
        return None

    if ":" in label:
        root, qual = label.split(":", 1)
        root = root.strip()
        qual = qual.strip() or "maj"
    else:
        root, qual = label.strip(), "maj"

    pc = NOTE_TO_PC.get(root)
    if pc is None:
        # normalize sharps/flats just in case
        pc = NOTE_TO_PC.get(root.replace("♯", "#").replace("♭", "b"))
    if pc is None:
        return None

    intervals: list[int]
    if qual in ("maj", ""):
        intervals = [0, 4, 7]
    elif qual in ("min", "m"):
        intervals = [0, 3, 7]
    elif qual == "7":
        intervals = [0, 4, 7, 10]
    elif qual == "maj7":
        intervals = [0, 4, 7, 11]
    elif qual == "min7":
        intervals = [0, 3, 7, 10]
    else:
        # fallback: try major triad
        intervals = [0, 4, 7]

    base = 12 * (int(root_octave) + 1) + int(pc)
    pitches = [base + i for i in intervals]
    return pitches


def export_chords_midi(
    out_path: Path,
    chords: List[ChordSegment],
    *,
    tempo_bpm: float,
    time_signature: str = "4/4",
    root_octave: int = 3,
    start_time_s: float = 0.0,
    onset_times_s: List[float] | None = None,
) -> None:
    """
    Create a simple MIDI file that plays block chords on each beat or on
    specific onset times. start_time_s can be negative to include a pickup
    before beat 1.
    """
    from music21 import chord as m21chord
    from music21 import instrument as m21instrument
    from music21 import meter as m21meter
    from music21 import note as m21note
    from music21 import stream as m21stream
    from music21 import tempo as m21tempo

    out_path.parent.mkdir(parents=True, exist_ok=True)

    tempo = max(30.0, float(tempo_bpm) if tempo_bpm else 120.0)
    num, den = _parse_time_signature(time_signature)
    beat_quarters = 4.0 / float(den)
    measure_quarters = float(num) * beat_quarters
    sec_per_q = 60.0 / tempo

    chords_sorted = sorted(chords or [], key=lambda c: float(c.start))
    start_time_s = float(start_time_s or 0.0)
    onsets = None
    if onset_times_s:
        onsets = sorted({float(t) for t in onset_times_s})

    part = m21stream.Part()
    part.insert(0, m21instrument.AcousticGuitar())
    part.insert(0, m21meter.TimeSignature(f"{num}/{den}"))
    part.insert(0, m21tempo.MetronomeMark(number=float(tempo)))

    seg_idx = 0

    def label_at(t_sec: float) -> str:
        nonlocal seg_idx
        while seg_idx < len(chords_sorted) and float(chords_sorted[seg_idx].end) <= t_sec:
            seg_idx += 1
        if seg_idx >= len(chords_sorted):
            return "N"
        seg = chords_sorted[seg_idx]
        if float(seg.start) <= t_sec < float(seg.end):
            return seg.label or "N"
        return "N"

    if onsets is not None and len(onsets) > 0:
        shift_s = -float(start_time_s) if start_time_s < 0 else 0.0
        for i, t in enumerate(onsets):
            t_rel = float(t) + float(shift_s)
            if t_rel < 0:
                continue
            if i + 1 < len(onsets):
                dur_s = max(0.05, float(onsets[i + 1]) - float(t))
            else:
                dur_s = max(0.05, float(beat_quarters) * float(sec_per_q))
            offset_q = t_rel / sec_per_q if sec_per_q > 0 else 0.0
            dur_q = dur_s / sec_per_q if sec_per_q > 0 else float(beat_quarters)
            lbl = label_at(float(t))
            pitches = _pitches_for_label(lbl, root_octave=int(root_octave))
            if not pitches:
                obj = m21note.Rest(quarterLength=float(dur_q))
            else:
                obj = m21chord.Chord(pitches, quarterLength=float(dur_q))
            part.insert(float(offset_q), obj)
    else:
        total_sec = max((float(c.end) for c in chords_sorted), default=0.0)
        total_sec_rel = max(0.0, total_sec - start_time_s)
        total_quarters = total_sec_rel / sec_per_q if sec_per_q > 0 else 0.0
        measures_count = max(1, int(math.ceil(total_quarters / measure_quarters - 1e-9)))

        for mi in range(measures_count):
            m = m21stream.Measure(number=mi + 1)
            for b in range(num):
                t_q = float(mi) * measure_quarters + float(b) * beat_quarters
                t_sec = start_time_s + t_q * sec_per_q
                lbl = label_at(t_sec)
                pitches = _pitches_for_label(lbl, root_octave=int(root_octave))
                if not pitches:
                    obj = m21note.Rest(quarterLength=float(beat_quarters))
                else:
                    obj = m21chord.Chord(pitches, quarterLength=float(beat_quarters))
                m.append(obj)
            part.append(m)

    score = m21stream.Score()
    score.metadata = None
    score.append(part)
    score.write("midi", fp=str(out_path))
