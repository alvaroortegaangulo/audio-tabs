from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import librosa
import numpy as np
import soundfile as sf

from app.core.config import settings
from app.schemas import JobResult, KeySignature, ChordSegment, ScoreData
from app.services.audio import ffmpeg_to_wav_mono_44k, load_wav, peak_normalize
from app.services.chords.extract import extract_chords_template
from app.services.grid.beats import estimate_beats_librosa
from app.services.engraving.lilypond import build_lilypond_score, render_lilypond_pdf
from app.services.midi.export import export_chords_midi
from app.services.musicxml.lead_sheet import export_lead_sheet_musicxml
from app.services.theory.key import estimate_key_from_chroma, spell_chord_label, NOTE_TO_PC

# Transcription & Export Imports
from app.services.amt.basic_pitch import transcribe_basic_pitch, NoteEvent
from app.services.theory.quantize import quantize_note_events_to_score
from app.services.musicxml.export import export_musicxml

def _job_title(job_dir: Path, input_path: Path) -> str:
    meta_path = job_dir / "input" / "meta.json"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        filename = str(meta.get("filename") or "").strip()
        if filename:
            return Path(filename).stem or filename
    except Exception:
        pass
    return input_path.stem or "Lead Sheet"

# Chords imports
from app.services.chords.template import (
    build_chord_library,
    chroma_features,
    emission_probs,
    frames_to_segments,
    finalize_segments,
    Segment
)
from app.services.chords.viterbi import viterbi_decode


def detect_chords(y, sr) -> list[Segment]:
    """
    Runs the Viterbi-based chord detection pipeline.
    """
    # 1. Compute Chroma
    chroma = chroma_features(y, sr)

    # 2. Build Library (Maj/Min/7)
    labels, T = build_chord_library(vocab="majmin7")

    # 3. Compute Probabilities
    probs = emission_probs(chroma, labels, T)

    # 4. Viterbi Decode
    # switch_penalty tunes how often chords change.
    # -5.0 is a reasonable starting point for log-prob costs.
    path, conf = viterbi_decode(probs, switch_penalty=-5.0)

    # 5. Convert to Segments
    hop_length = 512
    times = [i * hop_length / sr for i in range(len(path))]

    # Minimum chord length in seconds (e.g. 0.5s)
    raw_segs = frames_to_segments(path, conf, times, min_len=0.5)

    # 6. Finalize (map indices to labels)
    segs = finalize_segments(raw_segs, labels)

    return segs


def _chord_tone_pcs(label: str) -> set[int] | None:
    if not label or label == "N":
        return None

    if ":" in label:
        root, qual = label.split(":", 1)
        root = root.strip()
        qual = qual.strip().lower() or "maj"
    else:
        root, qual = label.strip(), "maj"

    pc = NOTE_TO_PC.get(root)
    if pc is None:
        return None

    if qual in ("maj", ""):
        intervals = [0, 4, 7]
    elif qual in ("min", "m"):
        intervals = [0, 3, 7]
    elif qual == "7":
        intervals = [0, 4, 7, 10]
    elif qual in ("min7", "m7"):
        intervals = [0, 3, 7, 10]
    elif qual == "maj7":
        intervals = [0, 4, 7, 11]
    elif qual == "dim":
        intervals = [0, 3, 6]
    elif qual == "aug":
        intervals = [0, 4, 8]
    elif qual == "sus2":
        intervals = [0, 2, 7]
    elif qual == "sus4":
        intervals = [0, 5, 7]
    else:
        intervals = [0, 4, 7]

    return {int((pc + i) % 12) for i in intervals}


def _merge_overlapping_notes(
    note_events: Iterable[NoteEvent],
    *,
    gap_s: float = 0.03,
) -> list[NoteEvent]:
    by_pitch: dict[int, list[NoteEvent]] = {}
    for ev in note_events:
        by_pitch.setdefault(int(ev.pitch_midi), []).append(ev)

    merged: list[NoteEvent] = []
    for pitch, events in by_pitch.items():
        events_sorted = sorted(events, key=lambda e: e.start_time_s)
        cur: NoteEvent | None = None
        for ev in events_sorted:
            if cur is None:
                cur = ev
                continue
            if float(ev.start_time_s) <= float(cur.end_time_s) + float(gap_s):
                end_time = max(float(cur.end_time_s), float(ev.end_time_s))
                amp = max(float(cur.amplitude), float(ev.amplitude))
                vel = max(int(cur.velocity), int(ev.velocity))
                cur = NoteEvent(
                    start_time_s=float(cur.start_time_s),
                    end_time_s=float(end_time),
                    pitch_midi=int(pitch),
                    velocity=int(vel),
                    amplitude=float(amp),
                )
            else:
                merged.append(cur)
                cur = ev
        if cur is not None:
            merged.append(cur)

    return sorted(merged, key=lambda e: e.start_time_s)


def _limit_onset_polyphony(
    note_events: Iterable[NoteEvent],
    *,
    max_notes: int = 6,
    onset_window_s: float = 0.03,
) -> list[NoteEvent]:
    events = sorted(note_events, key=lambda e: e.start_time_s)
    if not events:
        return []

    def pick_group(group: list[NoteEvent]) -> list[NoteEvent]:
        by_pitch: dict[int, NoteEvent] = {}
        for ev in group:
            prev = by_pitch.get(int(ev.pitch_midi))
            if prev is None or float(ev.amplitude) > float(prev.amplitude):
                by_pitch[int(ev.pitch_midi)] = ev
        candidates = list(by_pitch.values())
        candidates.sort(key=lambda e: float(e.amplitude), reverse=True)
        return candidates[: max(1, int(max_notes))]

    out: list[NoteEvent] = []
    group: list[NoteEvent] = [events[0]]
    last_start = float(events[0].start_time_s)
    for ev in events[1:]:
        if float(ev.start_time_s) - last_start <= float(onset_window_s):
            group.append(ev)
        else:
            out.extend(pick_group(group))
            group = [ev]
            last_start = float(ev.start_time_s)
    out.extend(pick_group(group))

    return sorted(out, key=lambda e: e.start_time_s)


def _filter_note_events(
    note_events: Iterable[NoteEvent],
    *,
    chords: Iterable[ChordSegment],
    min_amp: float,
    min_dur_s: float,
    min_pitch: int,
    max_pitch: int,
    chord_tone_bias: float = 0.15,
) -> list[NoteEvent]:
    events = sorted(note_events, key=lambda e: e.start_time_s)
    chords_sorted = sorted(chords, key=lambda c: float(c.start))
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

    out: list[NoteEvent] = []
    min_amp = float(min_amp)
    min_dur_s = float(min_dur_s)
    min_pitch = int(min_pitch)
    max_pitch = int(max_pitch)
    chord_tone_bias = float(chord_tone_bias)

    for ev in events:
        dur = float(ev.end_time_s) - float(ev.start_time_s)
        if dur < min_dur_s:
            continue
        if float(ev.amplitude) < min_amp:
            continue
        pitch = int(ev.pitch_midi)
        if pitch < min_pitch or pitch > max_pitch:
            continue

        if chords_sorted:
            mid = 0.5 * (float(ev.start_time_s) + float(ev.end_time_s))
            label = label_at(mid)
            pcs = _chord_tone_pcs(label)
            if pcs is not None and (pitch % 12) not in pcs:
                if float(ev.amplitude) < (min_amp + chord_tone_bias):
                    continue

        out.append(ev)

    return out


def run_pipeline(job_dir: Path, input_path: Path) -> JobResult:
    work = job_dir / "work"
    out = job_dir / "out"
    work.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)

    wav_path = work / "audio_mono_44k.wav"
    ffmpeg_to_wav_mono_44k(input_path, wav_path)

    y, sr = load_wav(wav_path)
    y = peak_normalize(y)

    harmonic_path = wav_path
    y_harm = y
    try:
        y_harm = librosa.effects.harmonic(y)
        y_harm = peak_normalize(y_harm)
        harmonic_path = work / "audio_harmonic.wav"
        sf.write(str(harmonic_path), y_harm, int(sr))
    except Exception as e:
        print(f"Error computing harmonic stem: {e}")
        y_harm = y
        harmonic_path = wav_path

    tempo_bpm, _beat_times = estimate_beats_librosa(y, sr)
    if not tempo_bpm or tempo_bpm <= 0:
        tempo_bpm = 120.0
    # Librosa a veces devuelve double-time; normaliza a un rango más musical.
    while tempo_bpm >= 200.0:
        tempo_bpm /= 2.0
    if tempo_bpm > 130.0:
        half = tempo_bpm / 2.0
        if 55.0 <= half <= 110.0:
            tempo_bpm = half
    while tempo_bpm < 50.0:
        tempo_bpm *= 2.0

    time_sig = "4/4"

    # --- 1. Basic Pitch Transcription ---
    try:
        midi_data, note_events = transcribe_basic_pitch(
            harmonic_path,
            midi_tempo=float(tempo_bpm),
            onset_threshold=float(settings.BASIC_PITCH_ONSET_THRESHOLD),
            frame_threshold=float(settings.BASIC_PITCH_FRAME_THRESHOLD),
            minimum_note_length_ms=float(settings.BASIC_PITCH_MIN_NOTE_MS),
            melodia_trick=True,
        )
    except Exception as e:
        print(f"Error executing basic_pitch: {e}")
        note_events = []
        midi_data = None

    # --- 2. Chord Detection ---
    chroma, _times, chords = extract_chords_template(
        y_harm,
        sr,
        vocab=str(settings.CHORD_VOCAB),
        switch_penalty=float(settings.SWITCH_PENALTY),
        min_segment_sec=float(settings.MIN_SEGMENT_SEC),
    )

    key_est = estimate_key_from_chroma(chroma)
    key_sig = None
    if key_est is not None:
        key_sig = KeySignature(
            tonic=key_est.tonic,
            mode=key_est.mode,
            fifths=int(key_est.fifths),
            name=key_est.name,
            vexflow=key_est.vexflow,
            use_flats=bool(key_est.use_flats),
            score=float(key_est.score),
        )

    use_flats = bool(key_sig.use_flats) if key_sig else False
    spelled_chords: list[ChordSegment] = []
    for c in chords:
        spelled_chords.append(
            ChordSegment(
                start=float(c.start),
                end=float(c.end),
                label=spell_chord_label(str(c.label), use_flats=use_flats),
                confidence=float(c.confidence),
            )
        )

    if note_events:
        note_events = _merge_overlapping_notes(note_events, gap_s=0.03)

        amps = np.asarray([float(ev.amplitude) for ev in note_events], dtype=np.float32)
        if amps.size > 0:
            min_amp = max(0.2, float(np.percentile(amps, 35)))
        else:
            min_amp = 0.2

        sec_per_q = 60.0 / float(tempo_bpm if tempo_bpm else 120.0)
        min_dur_s = max(0.08, 0.2 * sec_per_q)

        note_events = _filter_note_events(
            note_events,
            chords=spelled_chords,
            min_amp=min_amp,
            min_dur_s=min_dur_s,
            min_pitch=40,
            max_pitch=88,
            chord_tone_bias=0.15,
        )
        note_events = _limit_onset_polyphony(
            note_events,
            max_notes=6,
            onset_window_s=0.03,
        )

    # --- 3. Quantization to ScoreData ---
    quant_res = quantize_note_events_to_score(
        note_events,
        tempo_bpm=float(tempo_bpm),
        time_signature=time_sig
    )
    score_data = quant_res.score

    title = _job_title(job_dir, input_path)
    musicxml_path = out / "result.musicxml"

    # --- 4. Export MusicXML (Full Lead Sheet with Tab) ---
    export_musicxml(
        musicxml_path,
        score_data,
        tempo_bpm=float(tempo_bpm),
        time_signature=time_sig,
        key_signature_fifths=(key_sig.fifths if key_sig else None),
        title=title,
        instrument="guitar",
        chords=[Segment(c.start, c.end, c.label, c.confidence) for c in spelled_chords]
    )

    midi_path = out / "transcription.mid"
    export_chords_midi(
        midi_path,
        spelled_chords,
        tempo_bpm=float(tempo_bpm),
        time_signature=time_sig,
        root_octave=3,
    )

    # PDF (LilyPond) generation - kept as fallback/supplement
    # Ideally LilyPond service should also be updated to support full score,
    # but for now we focus on MusicXML/OSMD as requested.
    pdf_error = None
    try:
        ly = build_lilypond_score(
            spelled_chords,
            tempo_bpm=float(tempo_bpm),
            time_signature=time_sig,
            key_tonic=(key_sig.tonic if key_sig else None),
            key_mode=(key_sig.mode if key_sig else "major"),
            title=title,
        )
        render_lilypond_pdf(ly, out, basename="score")
    except Exception as e:
        pdf_error = f"No se pudo generar PDF (LilyPond): {e}"

    return JobResult(
        job_id=job_dir.name,
        tempo_bpm=float(tempo_bpm),
        time_signature=time_sig,
        key_signature=key_sig,
        chords=spelled_chords,
        transcription_backend="basic_pitch+chords_viterbi",
        transcription_error=pdf_error,
        score=score_data,
    )
