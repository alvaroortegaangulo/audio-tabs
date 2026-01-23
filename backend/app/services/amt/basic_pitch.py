"""
Basic Pitch transcription module using Spotify's Basic Pitch model.

This module provides automatic music transcription (AMT) functionality
for converting audio to MIDI and note events.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class NoteEvent:
    """Represents a single note event from transcription."""
    start_time_s: float
    end_time_s: float
    pitch_midi: int
    velocity: int
    amplitude: float


def transcribe_basic_pitch(
    audio_path: Path,
    *,
    midi_tempo: float = 120.0,
    onset_threshold: float = 0.5,
    frame_threshold: float = 0.3,
    minimum_note_length_ms: float = 127.70,
    melodia_trick: bool = True,
    min_velocity: int = 1,
    keep_melody: bool = True,
    melody_window_s: float = 0.08,
) -> tuple[object | None, list[NoteEvent]]:
    """
    Run Basic Pitch transcription on an audio file.

    Args:
        audio_path: Path to the audio file (WAV, MP3, etc.)
        midi_tempo: Tempo for MIDI output (unused by Basic Pitch but kept for compatibility)
        onset_threshold: Threshold for note onset detection (0.0-1.0)
        frame_threshold: Threshold for frame-level note detection (0.0-1.0)
        minimum_note_length_ms: Minimum note duration in milliseconds
        melodia_trick: Enable melodia trick for better melody detection
        min_velocity: Minimum velocity threshold (unused, kept for compatibility)
        keep_melody: Keep melody notes even if below threshold (unused, kept for compatibility)
        melody_window_s: Window size for melody detection (unused, kept for compatibility)

    Returns:
        Tuple of (PrettyMIDI object or None, list of NoteEvent)
    """
    from basic_pitch.inference import predict
    from basic_pitch import ICASSP_2022_MODEL_PATH

    # Enforce threshold bounds
    onset_threshold = min(1.0, max(0.0, float(onset_threshold)))
    frame_threshold = min(1.0, max(0.0, float(frame_threshold)))
    minimum_note_length_s = max(0.02, float(minimum_note_length_ms) / 1000.0)

    # Run Basic Pitch inference
    model_output, midi_data, note_events_raw = predict(
        audio_path=str(audio_path),
        model_or_model_path=ICASSP_2022_MODEL_PATH,
        onset_threshold=onset_threshold,
        frame_threshold=frame_threshold,
        minimum_note_length=minimum_note_length_s,
        melodia_trick=melodia_trick,
    )

    # Convert Basic Pitch note events to our NoteEvent format
    # Basic Pitch returns: list of tuples (start_time, end_time, pitch, amplitude, pitch_bends)
    events: list[NoteEvent] = []
    for note in note_events_raw:
        start_time = float(note[0])
        end_time = float(note[1])
        pitch = int(note[2])
        amplitude = float(note[3]) if len(note) > 3 else 1.0

        # Skip invalid notes
        if end_time <= start_time:
            continue

        # Convert amplitude (0-1) to velocity (1-127)
        velocity = max(1, min(127, int(round(amplitude * 127.0))))

        events.append(
            NoteEvent(
                start_time_s=start_time,
                end_time_s=end_time,
                pitch_midi=pitch,
                velocity=velocity,
                amplitude=amplitude,
            )
        )

    # Sort by start time
    events = sorted(events, key=lambda e: e.start_time_s)

    return midi_data, events


def save_note_events_csv(note_events: list[NoteEvent], out_path: Path) -> None:
    """Save note events to a CSV file."""
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

    Args:
        note_events: List of NoteEvent objects
        hop_sec: Time step between frames in seconds
        total_sec: Total duration in seconds (if None, computed from events)

    Returns:
        Tuple of (chroma matrix [12, frames], times vector [frames])
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
