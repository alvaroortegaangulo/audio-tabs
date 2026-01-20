from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class NoteEvent:
    start_time_s: float
    end_time_s: float
    pitch_midi: int
    velocity: int
    amplitude: float


_MODEL = None


def _get_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    from basic_pitch import ICASSP_2022_MODEL_PATH
    from basic_pitch.inference import Model

    _MODEL = Model(ICASSP_2022_MODEL_PATH)
    return _MODEL


def transcribe_basic_pitch(
    audio_path: Path,
    *,
    midi_tempo: float,
    onset_threshold: float = 0.5,
    frame_threshold: float = 0.3,
    minimum_note_length_ms: float = 127.70,
    melodia_trick: bool = True,
) -> tuple[object, list[NoteEvent]]:
    """
    Run Spotify Basic Pitch on an audio file.

    Returns a tuple (pretty_midi_object, note_events).
    """
    from basic_pitch.inference import predict

    model = _get_model()

    # basic_pitch.predict prints to stdout; avoid polluting worker logs.
    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    with redirect_stdout(buf):
        _, midi_data, raw_events = predict(
            str(audio_path),
            model_or_model_path=model,
            onset_threshold=float(onset_threshold),
            frame_threshold=float(frame_threshold),
            minimum_note_length=float(minimum_note_length_ms),
            melodia_trick=bool(melodia_trick),
            midi_tempo=float(midi_tempo),
        )

    events: list[NoteEvent] = []
    for start_s, end_s, pitch, amplitude, _pitch_bend in raw_events:
        events.append(
            NoteEvent(
                start_time_s=float(start_s),
                end_time_s=float(end_s),
                pitch_midi=int(pitch),
                velocity=int(round(127 * float(amplitude))),
                amplitude=float(amplitude),
            )
        )

    return midi_data, events


def save_note_events_csv(note_events: list[NoteEvent], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["start_time_s,end_time_s,pitch_midi,velocity,amplitude"]
    for ev in note_events:
        lines.append(
            f"{ev.start_time_s:.6f},{ev.end_time_s:.6f},{ev.pitch_midi},{ev.velocity},{ev.amplitude:.6f}"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def chroma_from_note_events(
    note_events: list[NoteEvent],
    *,
    hop_sec: float = 0.05,
    total_sec: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert note events into a chroma matrix [12, frames] and a times vector [frames] (seconds).
    """
    hop = float(hop_sec)
    if hop <= 0:
        raise ValueError("hop_sec must be > 0")

    if total_sec is None:
        total_sec = max((ev.end_time_s for ev in note_events), default=0.0)
    total_sec = max(0.0, float(total_sec))

    frames = int(np.ceil(total_sec / hop)) + 1
    chroma = np.zeros((12, frames), dtype=np.float32)

    for ev in note_events:
        if ev.end_time_s <= ev.start_time_s:
            continue
        pc = int(ev.pitch_midi) % 12
        s = int(np.floor(ev.start_time_s / hop))
        e = int(np.ceil(ev.end_time_s / hop))
        s = max(0, min(frames - 1, s))
        e = max(s + 1, min(frames, e))
        chroma[pc, s:e] += float(ev.amplitude) if ev.amplitude > 0 else 1.0

    chroma /= (np.linalg.norm(chroma, axis=0, keepdims=True) + 1e-9)
    times = (np.arange(frames, dtype=np.float32) * hop).astype(np.float32)
    return chroma, times

