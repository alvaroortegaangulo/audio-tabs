from __future__ import annotations

# NOTE: This module now wraps Omnizart; consider renaming to transcribe.py.

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import inspect
import os
import tempfile

import numpy as np


@dataclass(frozen=True)
class NoteEvent:
    start_time_s: float
    end_time_s: float
    pitch_midi: int
    velocity: int
    amplitude: float


_MODEL = None
_MODEL_PATH: str | None = None
_MODEL_NAME = "music-v1"


def _configure_omnizart_env() -> None:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    try:
        import tensorflow as tf  # type: ignore

        try:
            tf.config.set_visible_devices([], "GPU")
        except Exception:
            pass
        count = int(os.cpu_count() or 1)
        try:
            tf.config.threading.set_intra_op_parallelism_threads(count)
            tf.config.threading.set_inter_op_parallelism_threads(max(1, count // 2))
        except Exception:
            pass
    except Exception:
        return


def _get_model() -> tuple[object | None, str | None]:
    global _MODEL, _MODEL_PATH
    if _MODEL is not None or _MODEL_PATH is not None:
        return _MODEL, _MODEL_PATH

    _configure_omnizart_env()

    from omnizart.music import app as music_app  # type: ignore

    model_obj = None
    model_path: str | None = None

    if hasattr(music_app, "load_model"):
        try:
            model_obj = music_app.load_model(_MODEL_NAME)
        except TypeError:
            model_obj = music_app.load_model()

    if isinstance(model_obj, (str, Path)):
        model_path = str(model_obj)
        model_obj = None
    elif isinstance(model_obj, tuple):
        for item in model_obj:
            if isinstance(item, (str, Path)):
                model_path = str(item)
            else:
                model_obj = item

    _MODEL = model_obj
    _MODEL_PATH = model_path
    return _MODEL, _MODEL_PATH


def _resolve_transcribe() -> tuple[object, object]:
    from omnizart.music import app as music_app  # type: ignore

    if hasattr(music_app, "transcribe"):
        return music_app.transcribe, music_app

    for attr in ("MusicTranscription", "Transcriber", "Transcription"):
        if hasattr(music_app, attr):
            cls = getattr(music_app, attr)
            try:
                obj = cls()
            except Exception:
                continue
            if hasattr(obj, "transcribe"):
                return obj.transcribe, obj

    raise RuntimeError("Omnizart transcribe entrypoint not found")


def _enforce_thresholds(
    onset_threshold: float,
    frame_threshold: float,
    minimum_note_length_ms: float,
) -> tuple[float, float, float]:
    onset = min(1.0, max(0.0, float(onset_threshold)))
    frame = min(1.0, max(0.0, float(frame_threshold)))
    min_len_ms = max(20.0, float(minimum_note_length_ms))
    return onset, frame, min_len_ms


def _melody_anchor_pitches(
    note_events: list[NoteEvent],
    *,
    window_s: float,
) -> dict[int, int]:
    window = max(0.01, float(window_s))
    anchors: dict[int, int] = {}
    for ev in note_events:
        mid = 0.5 * (float(ev.start_time_s) + float(ev.end_time_s))
        bucket = int(mid / window)
        pitch = int(ev.pitch_midi)
        prev = anchors.get(bucket)
        if prev is None or pitch > prev:
            anchors[bucket] = pitch
    return anchors


def _clean_note_events(
    note_events: list[NoteEvent],
    *,
    min_duration_s: float,
    min_velocity: int,
    keep_melody: bool,
    melody_window_s: float,
) -> list[NoteEvent]:
    if not note_events:
        return []

    min_duration_s = max(0.0, float(min_duration_s))
    min_velocity = int(min_velocity)
    keep_melody = bool(keep_melody)
    melody_window_s = max(0.01, float(melody_window_s))

    anchors = _melody_anchor_pitches(note_events, window_s=melody_window_s) if keep_melody else {}
    out: list[NoteEvent] = []
    for ev in note_events:
        start = float(ev.start_time_s)
        end = float(ev.end_time_s)
        if end <= start:
            continue
        if (end - start) < min_duration_s:
            continue

        velocity = int(ev.velocity)
        if velocity < min_velocity:
            if not keep_melody:
                continue
            mid = 0.5 * (start + end)
            bucket = int(mid / melody_window_s)
            if anchors.get(bucket) != int(ev.pitch_midi):
                continue

        out.append(ev)

    return sorted(out, key=lambda e: e.start_time_s)


def _normalize_velocity(value: float | int | None) -> tuple[int, float]:
    if value is None:
        return 80, 0.63
    try:
        v = float(value)
    except Exception:
        return 80, 0.63
    if v <= 0:
        return 1, 0.01
    if v <= 1.0:
        vel = max(1, min(127, int(round(v * 127.0))))
        return vel, max(0.01, min(1.0, float(v)))
    vel = max(1, min(127, int(round(v))))
    amp = max(0.01, min(1.0, vel / 127.0))
    return vel, amp


def _note_fields(note: object) -> tuple[float | None, float | None, int | None, float | int | None]:
    if isinstance(note, dict):
        start = note.get("start_time", note.get("start", note.get("onset")))
        end = note.get("end_time", note.get("end", note.get("offset")))
        pitch = note.get("pitch", note.get("note", note.get("pitch_midi")))
        velocity = note.get("velocity", note.get("velocity_midi", note.get("confidence")))
        return start, end, pitch, velocity

    for attr in ("start_time", "start", "onset"):
        if hasattr(note, attr):
            start = getattr(note, attr)
            break
    else:
        start = None

    for attr in ("end_time", "end", "offset"):
        if hasattr(note, attr):
            end = getattr(note, attr)
            break
    else:
        end = None

    for attr in ("pitch", "note", "pitch_midi"):
        if hasattr(note, attr):
            pitch = getattr(note, attr)
            break
    else:
        pitch = None

    for attr in ("velocity", "velocity_midi", "confidence"):
        if hasattr(note, attr):
            velocity = getattr(note, attr)
            break
    else:
        velocity = None

    return start, end, pitch, velocity


def _extract_notes(result: object) -> list[object]:
    if result is None:
        return []
    if isinstance(result, (list, tuple)):
        # If it's a list of notes, return directly.
        if len(result) == 0:
            return []
        if not isinstance(result, tuple):
            return list(result)
        for item in result:
            if isinstance(item, list):
                return list(item)
            if isinstance(item, tuple) and item and not isinstance(item[0], (str, Path)):
                return list(item)
        return list(result)
    if hasattr(result, "notes"):
        return list(getattr(result, "notes"))
    if isinstance(result, dict):
        for key in ("notes", "note_events", "events"):
            if key in result:
                return list(result[key])
    return []


def _find_midi_file(output_dir: Path) -> Path | None:
    for ext in ("*.mid", "*.midi"):
        for p in output_dir.rglob(ext):
            return p
    return None


def _notes_from_midi(midi_path: Path) -> tuple[object | None, list[NoteEvent]]:
    try:
        import pretty_midi  # type: ignore
    except Exception:
        return None, []

    pm = pretty_midi.PrettyMIDI(str(midi_path))
    events: list[NoteEvent] = []
    for inst in pm.instruments:
        for note in inst.notes:
            vel, amp = _normalize_velocity(note.velocity)
            events.append(
                NoteEvent(
                    start_time_s=float(note.start),
                    end_time_s=float(note.end),
                    pitch_midi=int(note.pitch),
                    velocity=int(vel),
                    amplitude=float(amp),
                )
            )
    return pm, events


def transcribe_basic_pitch(
    audio_path: Path,
    *,
    midi_tempo: float,
    onset_threshold: float = 0.6,
    frame_threshold: float = 0.4,
    minimum_note_length_ms: float = 60.0,
    melodia_trick: bool = True,
    min_velocity: int = 30,
    keep_melody: bool = True,
    melody_window_s: float = 0.08,
) -> tuple[object | None, list[NoteEvent]]:
    """
    Run Omnizart music transcription on an audio file.

    Returns a tuple (pretty_midi_object_or_None, note_events).
    """
    _ = melodia_trick
    _ = midi_tempo

    onset_threshold, frame_threshold, minimum_note_length_ms = _enforce_thresholds(
        onset_threshold,
        frame_threshold,
        minimum_note_length_ms,
    )

    model_obj, model_path = _get_model()
    transcribe_fn, _ctx = _resolve_transcribe()

    kwargs: dict[str, object] = {}
    sig = inspect.signature(transcribe_fn)
    if model_path is not None and "model_path" in sig.parameters:
        kwargs["model_path"] = model_path
    if model_obj is not None and "model" in sig.parameters:
        kwargs["model"] = model_obj

    events: list[NoteEvent] = []
    midi_data: object | None = None

    with tempfile.TemporaryDirectory() as tmp_dir:
        if "output" in sig.parameters:
            kwargs["output"] = str(tmp_dir)
        result = transcribe_fn(str(audio_path), **kwargs)

        notes = _extract_notes(result)
        if not notes:
            midi_path = _find_midi_file(Path(tmp_dir))
            if midi_path is not None:
                midi_data, events = _notes_from_midi(midi_path)

        if notes:
            for note in notes:
                if isinstance(note, (list, tuple)) and len(note) >= 3:
                    start_s, end_s, pitch = note[:3]
                    velocity = note[3] if len(note) > 3 else None
                else:
                    start_s, end_s, pitch, velocity = _note_fields(note)
                if start_s is None or end_s is None or pitch is None:
                    continue
                if float(end_s) <= float(start_s):
                    continue
                vel, amp = _normalize_velocity(velocity)
                events.append(
                    NoteEvent(
                        start_time_s=float(start_s),
                        end_time_s=float(end_s),
                        pitch_midi=int(pitch),
                        velocity=int(vel),
                        amplitude=float(amp),
                    )
                )

    min_duration_s = float(minimum_note_length_ms) / 1000.0
    if frame_threshold > 0:
        min_velocity = max(int(min_velocity), int(round(float(frame_threshold) * 127.0)))

    events = _clean_note_events(
        events,
        min_duration_s=float(min_duration_s),
        min_velocity=int(min_velocity),
        keep_melody=bool(keep_melody),
        melody_window_s=float(melody_window_s),
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
