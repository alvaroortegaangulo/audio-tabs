from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional

from app.schemas import ChordSegment


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


def _figure_from_label(label: str) -> Optional[str]:
    """
    label backend: "G:maj", "A:min", "D:7", "N"
    figure music21: "G", "Am", "D7", None
    """
    if not label or label == "N":
        return None

    if ":" in label:
        root, qual = label.split(":", 1)
        root = root.strip()
        qual = qual.strip() or "maj"
    else:
        root, qual = label.strip(), "maj"

    if not root:
        return None

    if qual in ("maj", ""):
        return root
    if qual in ("min", "m"):
        return f"{root}m"
    if qual == "7":
        return f"{root}7"
    if qual == "maj7":
        return f"{root}maj7"
    if qual == "min7":
        return f"{root}m7"

    # fallback: best-effort
    return f"{root}{qual}"


def export_lead_sheet_musicxml(
    out_path: Path,
    chords: List[ChordSegment],
    *,
    tempo_bpm: float,
    time_signature: str = "4/4",
    key_signature_fifths: int | None = None,
    title: str = "Lead Sheet",
    instrument: str = "guitar",
) -> None:
    """
    Generate a simple lead sheet (rests + harmony symbols) in MusicXML.
    """
    from music21 import harmony as m21harmony
    from music21 import instrument as m21instrument
    from music21 import key as m21key
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
    total_sec = max((float(c.end) for c in chords_sorted), default=0.0)
    total_quarters = total_sec / sec_per_q if sec_per_q > 0 else 0.0
    measures_count = max(1, int(math.ceil(total_quarters / measure_quarters - 1e-9)))

    score = m21stream.Score()
    score.metadata = None

    part = m21stream.Part()
    if instrument == "guitar":
        part.insert(0, m21instrument.Guitar())
    else:
        part.insert(0, m21instrument.Piano())

    part.insert(0, m21meter.TimeSignature(f"{num}/{den}"))
    if key_signature_fifths is not None:
        part.insert(0, m21key.KeySignature(int(key_signature_fifths)))
    part.insert(0, m21tempo.MetronomeMark(number=float(tempo)))

    measures: list[m21stream.Measure] = []
    for mi in range(measures_count):
        m = m21stream.Measure(number=mi + 1)
        for _ in range(num):
            m.append(m21note.Rest(quarterLength=float(beat_quarters)))
        measures.append(m)
        part.append(m)

    seg_idx = 0
    last_label = ""

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

    for mi in range(measures_count):
        for b in range(num):
            t_q = float(mi) * measure_quarters + float(b) * beat_quarters
            t_sec = t_q * sec_per_q
            lbl = label_at(t_sec)
            if lbl == "N":
                last_label = ""
                continue

            # show at bar start or when changes
            should_show = b == 0 or lbl != last_label
            if should_show:
                fig = _figure_from_label(lbl)
                if fig:
                    cs = m21harmony.ChordSymbol(fig)
                    measures[mi].insert(float(b) * beat_quarters, cs)
            last_label = lbl

    score.append(part)
    score.write("musicxml", fp=str(out_path))

