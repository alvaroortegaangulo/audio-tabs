from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np

from app.core.config import settings
from app.schemas import ChordSegment, KeySignature
from app.services.amt.basic_pitch import NoteEvent

_LOG = logging.getLogger(__name__)

_HARMONIC_RATIOS = (
    (2.0, "even"),       # octave
    (3.0 / 2.0, "odd"),  # perfect fifth
    (4.0 / 3.0, "odd"),  # perfect fourth
    (5.0 / 4.0, "odd"),  # major third
)

_CHUNK_SEC = 30.0


def _setting_float(name: str, default: float) -> float:
    return float(getattr(settings, name, default) or default)


def _setting_int(name: str, default: int) -> int:
    return int(getattr(settings, name, default) or default)


def _freq_from_midi(pitch_midi: int) -> float:
    return 440.0 * (2.0 ** ((float(pitch_midi) - 69.0) / 12.0))


def _cents_diff(ratio: float, target: float) -> float:
    return float(1200.0 * math.log2(float(ratio) / float(target)))


def _group_by_onset(
    events: list[NoteEvent],
    *,
    window_s: float,
) -> list[list[NoteEvent]]:
    if not events:
        return []
    groups: list[list[NoteEvent]] = []
    current: list[NoteEvent] = [events[0]]
    start_t = float(events[0].start_time_s)
    for ev in events[1:]:
        t = float(ev.start_time_s)
        if t - start_t <= float(window_s):
            current.append(ev)
        else:
            groups.append(current)
            current = [ev]
            start_t = t
    groups.append(current)
    return groups


def remove_harmonic_duplicates(note_events: list[NoteEvent]) -> list[NoteEvent]:
    """
    Remove likely harmonic duplicates within onset windows.
    """
    if not note_events:
        return []

    window_s = _setting_float("HARMONIC_DUPLICATE_WINDOW_MS", 100.0) / 1000.0
    tol_cents = _setting_float("HARMONIC_TOLERANCE_CENTS", 50.0)
    even_thresh = _setting_float("HARMONIC_EVEN_THRESHOLD", 0.7)
    odd_thresh = _setting_float("HARMONIC_ODD_THRESHOLD", 0.55)

    events = sorted(note_events, key=lambda e: float(e.start_time_s))
    cleaned: list[NoteEvent] = []
    removed = 0

    idx = 0
    while idx < len(events):
        chunk_start = float(events[idx].start_time_s)
        chunk_end = chunk_start + float(_CHUNK_SEC)
        chunk: list[NoteEvent] = []
        while idx < len(events) and float(events[idx].start_time_s) < chunk_end:
            chunk.append(events[idx])
            idx += 1

        groups = _group_by_onset(chunk, window_s=window_s)
        for group in groups:
            if len(group) < 2:
                cleaned.extend(group)
                continue
            freqs = np.array([_freq_from_midi(int(ev.pitch_midi)) for ev in group], dtype=np.float64)
            amps = np.array([float(ev.amplitude) for ev in group], dtype=np.float64)
            drop: set[int] = set()
            for i in range(len(group)):
                if i in drop:
                    continue
                for j in range(i + 1, len(group)):
                    if j in drop:
                        continue
                    f_i = float(freqs[i])
                    f_j = float(freqs[j])
                    if f_i <= 0 or f_j <= 0:
                        continue
                    if abs(f_i - f_j) < 1e-6:
                        continue
                    if f_i < f_j:
                        low_idx, high_idx = i, j
                    else:
                        low_idx, high_idx = j, i
                    ratio = float(freqs[high_idx]) / float(freqs[low_idx])

                    matched = False
                    for target, kind in _HARMONIC_RATIOS:
                        if abs(_cents_diff(ratio, target)) <= float(tol_cents):
                            thresh = float(even_thresh) if kind == "even" else float(odd_thresh)
                            if float(amps[high_idx]) < float(amps[low_idx]) * thresh:
                                drop.add(high_idx)
                                removed += 1
                            matched = True
                            break
                    if matched:
                        continue

            for k, ev in enumerate(group):
                if k not in drop:
                    cleaned.append(ev)

    _LOG.info("Removed %d harmonic duplicates", int(removed))
    return sorted(cleaned, key=lambda e: float(e.start_time_s))


def merge_temporal_clusters(
    note_events: list[NoteEvent],
    window_ms: float = 80.0,
) -> list[NoteEvent]:
    """
    Merge nearby detections of the same (or nearly same) pitch.
    """
    if not note_events:
        return []

    window_s = float(window_ms) / 1000.0
    gap_s = _setting_float("TEMPORAL_CLUSTER_GAP_MS", 50.0) / 1000.0
    events = sorted(note_events, key=lambda e: float(e.start_time_s))

    groups: list[dict[str, object]] = []
    last_by_pitch: dict[int, int] = {}

    merged = 0
    for ev in events:
        pitch = int(ev.pitch_midi)
        candidates = []
        for p in (pitch - 1, pitch, pitch + 1):
            idx = last_by_pitch.get(int(p))
            if idx is not None:
                candidates.append(idx)

        best_idx = None
        best_score = None
        for idx in candidates:
            group = groups[idx]
            group_pitch = int(group["pitch"])
            start_t = float(group["start"])
            end_t = float(group["end"])
            if abs(int(pitch) - int(group_pitch)) > 1:
                continue
            if float(ev.start_time_s) - start_t > window_s:
                continue
            if float(ev.start_time_s) - end_t > float(gap_s):
                continue
            score = abs(int(pitch) - int(group_pitch)) + abs(float(ev.start_time_s) - end_t)
            if best_score is None or score < best_score:
                best_score = score
                best_idx = idx

        if best_idx is None:
            groups.append(
                {
                    "start": float(ev.start_time_s),
                    "end": float(ev.end_time_s),
                    "pitch": int(ev.pitch_midi),
                    "amp": float(ev.amplitude),
                    "vel": int(ev.velocity),
                }
            )
            last_by_pitch[int(pitch)] = len(groups) - 1
        else:
            group = groups[best_idx]
            group["end"] = max(float(group["end"]), float(ev.end_time_s))
            if float(ev.amplitude) >= float(group["amp"]):
                group["amp"] = float(ev.amplitude)
                group["vel"] = int(ev.velocity)
                group["pitch"] = int(ev.pitch_midi)
            groups[best_idx] = group
            last_by_pitch[int(pitch)] = best_idx
            merged += 1

    out: list[NoteEvent] = []
    for group in groups:
        out.append(
            NoteEvent(
                start_time_s=float(group["start"]),
                end_time_s=float(group["end"]),
                pitch_midi=int(group["pitch"]),
                velocity=int(group["vel"]),
                amplitude=float(group["amp"]),
            )
        )

    _LOG.info("Merged %d temporal clusters", int(merged))
    return sorted(out, key=lambda e: float(e.start_time_s))


def _get_chord_tone_pcs(label: str) -> set[int] | None:
    try:
        from app.services import pipeline as pipeline_mod

        return pipeline_mod._chord_tone_pcs(label)
    except Exception:
        return None


def _chord_at_time(chords: list[ChordSegment], t: float, idx: int) -> tuple[int, str | None]:
    i = idx
    while i < len(chords) and float(chords[i].end) <= t:
        i += 1
    if i < len(chords):
        seg = chords[i]
        if float(seg.start) <= t < float(seg.end):
            return i, str(seg.label or "N")
    return i, None


def _group_onsets_indices(
    note_events: list[NoteEvent],
    window_s: float,
) -> list[list[int]]:
    if not note_events:
        return []
    groups: list[list[int]] = []
    current = [0]
    start_t = float(note_events[0].start_time_s)
    for idx, ev in enumerate(note_events[1:], start=1):
        t = float(ev.start_time_s)
        if t - start_t <= float(window_s):
            current.append(idx)
        else:
            groups.append(current)
            current = [idx]
            start_t = t
    groups.append(current)
    return groups


@dataclass
class _VoiceState:
    last_pitch: int
    min_pitch: int
    max_pitch: int
    indices: list[int]


def _assign_voices(
    note_events: list[NoteEvent],
    *,
    onset_window_s: float,
) -> dict[int, list[int]]:
    groups = _group_onsets_indices(note_events, window_s=onset_window_s)
    voices: list[_VoiceState] = []

    for group in groups:
        pitches = [(idx, int(note_events[idx].pitch_midi)) for idx in group]
        pitches.sort(key=lambda p: p[1])

        if not voices:
            for idx, pitch in pitches:
                voices.append(_VoiceState(last_pitch=pitch, min_pitch=pitch, max_pitch=pitch, indices=[idx]))
            continue

        used = set()
        assignments: list[tuple[int, int]] = []

        for idx, pitch in pitches:
            best = None
            best_cost = None
            for v_idx, voice in enumerate(voices):
                if v_idx in used:
                    continue
                jump = abs(int(pitch) - int(voice.last_pitch))
                cost = float(jump)
                if jump > 7:
                    cost += float(math.exp((jump - 7) / 5.0))
                range_next = max(int(voice.max_pitch), int(pitch)) - min(int(voice.min_pitch), int(pitch))
                if range_next > 24:
                    cost += 4.0
                if best_cost is None or cost < best_cost:
                    best_cost = cost
                    best = v_idx
            if best is None:
                voices.append(_VoiceState(last_pitch=pitch, min_pitch=pitch, max_pitch=pitch, indices=[idx]))
                best = len(voices) - 1
            else:
                used.add(best)
            assignments.append((best, idx))

        for v_idx, idx in assignments:
            voice = voices[v_idx]
            pitch = int(note_events[idx].pitch_midi)
            voice.last_pitch = pitch
            voice.min_pitch = min(int(voice.min_pitch), pitch)
            voice.max_pitch = max(int(voice.max_pitch), pitch)
            voice.indices.append(idx)
            voices[v_idx] = voice

        voices.sort(key=lambda v: int(v.last_pitch))

    out: dict[int, list[int]] = {}
    for i, voice in enumerate(voices):
        out[i] = sorted(voice.indices, key=lambda idx: float(note_events[idx].start_time_s))
    return out


def _melodic_score(
    pitch: int,
    prev_pitch: int | None,
) -> float:
    if prev_pitch is None:
        return 0.6
    jump = abs(int(pitch) - int(prev_pitch))
    if jump > 12:
        return 0.2
    return max(0.2, 1.0 - float(jump) / 12.0 * 0.6)


def apply_music_theory_rules(
    note_events: list[NoteEvent],
    chords: list[ChordSegment],
    key_sig: KeySignature | None = None,
) -> list[NoteEvent]:
    _ = key_sig
    if not note_events:
        return []

    dissonance_window_s = _setting_float("DISSONANCE_WINDOW_MS", 60.0) / 1000.0
    aggressiveness = float(_setting_float("DISSONANCE_CORRECTION_AGGRESSIVENESS", 0.5))
    aggressiveness = max(0.0, min(1.0, aggressiveness))
    voice_window_s = _setting_float("VOICE_ASSIGN_WINDOW_MS", 60.0) / 1000.0

    events = sorted(note_events, key=lambda e: float(e.start_time_s))
    voices = _assign_voices(events, onset_window_s=voice_window_s)
    prev_pitch: dict[int, int] = {}
    for indices in voices.values():
        for i, idx in enumerate(indices):
            if i == 0:
                continue
            prev_pitch[idx] = int(events[indices[i - 1]].pitch_midi)

    groups = _group_onsets_indices(events, window_s=dissonance_window_s)
    chord_idx = 0
    remove: set[int] = set()
    removed_dissonance = 0

    for group in groups:
        if len(group) < 2:
            continue
        pitches = [int(events[idx].pitch_midi) for idx in group]
        amps = [float(events[idx].amplitude) for idx in group]
        avg_amp = float(np.mean(amps)) if amps else 0.0
        if len(pitches) >= 3 and (max(pitches) - min(pitches) <= 2):
            continue

        t = float(events[group[0]].start_time_s)
        chord_idx, label = _chord_at_time(chords, t, chord_idx)
        chord_pcs = _get_chord_tone_pcs(label or "N") if label else None

        for i, idx_i in enumerate(group):
            if idx_i in remove:
                continue
            for idx_j in group[i + 1:]:
                if idx_j in remove:
                    continue
                diff = abs(int(events[idx_i].pitch_midi) - int(events[idx_j].pitch_midi))
                if diff % 12 != 1:
                    continue

                def credibility(idx: int) -> float:
                    amp = float(events[idx].amplitude)
                    amp_score = min(1.0, amp / (avg_amp + 1e-6)) if avg_amp > 0 else 0.5
                    pitch = int(events[idx].pitch_midi)
                    chord_score = 0.6
                    if chord_pcs is not None:
                        chord_score = 1.0 if (pitch % 12) in chord_pcs else 0.2
                    melodic = _melodic_score(pitch, prev_pitch.get(idx))
                    return 0.5 * amp_score + 0.3 * chord_score + 0.2 * melodic

                score_i = credibility(idx_i)
                score_j = credibility(idx_j)
                if score_i == score_j:
                    continue

                high_idx, low_idx = (idx_i, idx_j) if score_i > score_j else (idx_j, idx_i)
                diff_score = abs(score_i - score_j)
                threshold = 0.2 - 0.1 * aggressiveness
                if diff_score >= threshold:
                    remove.add(low_idx)
                    removed_dissonance += 1

    filtered = [ev for i, ev in enumerate(events) if i not in remove]

    # Voice range sanity pass to remove low-amplitude outliers.
    voices = _assign_voices(filtered, onset_window_s=voice_window_s)
    removed_outliers = 0
    to_remove: set[int] = set()
    for indices in voices.values():
        pitches = [int(filtered[idx].pitch_midi) for idx in indices]
        if not pitches:
            continue
        if max(pitches) - min(pitches) <= 24:
            continue
        median_pitch = int(np.median(pitches))
        avg_amp = float(np.mean([float(filtered[idx].amplitude) for idx in indices]))
        for idx in indices:
            pitch = int(filtered[idx].pitch_midi)
            if abs(pitch - median_pitch) > 12 and float(filtered[idx].amplitude) < avg_amp * 0.4:
                to_remove.add(idx)
                removed_outliers += 1

    final = [ev for i, ev in enumerate(filtered) if i not in to_remove]
    _LOG.info(
        "Applied music theory rules: removed %d dissonances, %d voice outliers",
        int(removed_dissonance),
        int(removed_outliers),
    )
    return final
