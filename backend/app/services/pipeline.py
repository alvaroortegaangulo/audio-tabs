from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Iterable

import librosa
import numpy as np
import soundfile as sf

from app.core.config import settings
from app.schemas import JobResult, KeySignature, ChordSegment, ScoreData
from app.services.audio import ffmpeg_to_wav_mono_44k, load_wav, peak_normalize
from app.services.chords.extract import extract_chords_template
from app.services.grid.beats import estimate_beats_librosa, normalize_beat_times
from app.services.engraving.lilypond import build_lilypond_score, render_lilypond_pdf
from app.services.midi.export import export_chords_midi
from app.services.musicxml.lead_sheet import export_lead_sheet_musicxml
from app.services.theory.key import estimate_key_from_chroma, spell_chord_label, NOTE_TO_PC

# Transcription & Export Imports
from app.services.amt.basic_pitch import transcribe_basic_pitch, NoteEvent, save_note_events_csv
from app.services.separation.demucs_sep import run_demucs_4stems, select_stem_path, get_stem_path
from app.services.theory.quantize import quantize_note_events_to_score
from app.services.musicxml.export import export_musicxml

_CHORD_TONE_BIAS = 0.08
_CHORD_CONFIDENCE_THRESHOLD = 0.03
_SEVENTH_MIN_CONFIDENCE = 0.03
_SEVENTH_MIN_DURATION = 0.6
_SEVENTH_RATIO = 0.55

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
    chroma, harm_rms = chroma_features(y, sr)

    # 2. Build Library (Maj/Min/7)
    labels, T = build_chord_library(vocab="majmin7")

    # 3. Compute Probabilities
    probs = emission_probs(chroma, harm_rms, labels, T)

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
    chord_tone_bias: float = 0.08,
    chord_confidence_threshold: float | None = None,
) -> list[NoteEvent]:
    events = sorted(note_events, key=lambda e: e.start_time_s)
    chords_sorted = sorted(chords, key=lambda c: float(c.start))
    seg_idx = 0

    def label_at(t_sec: float) -> tuple[str, float]:
        nonlocal seg_idx
        while seg_idx < len(chords_sorted) and float(chords_sorted[seg_idx].end) <= t_sec:
            seg_idx += 1
        if seg_idx >= len(chords_sorted):
            return "N", 0.0
        seg = chords_sorted[seg_idx]
        if float(seg.start) <= t_sec < float(seg.end):
            return seg.label or "N", float(seg.confidence)
        return "N", 0.0

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
            label, conf = label_at(mid)
            if chord_confidence_threshold is not None and conf < float(chord_confidence_threshold):
                label = "N"
            pcs = _chord_tone_pcs(label)
            if pcs is not None and (pitch % 12) not in pcs:
                if float(ev.amplitude) < (min_amp + chord_tone_bias):
                    continue

        out.append(ev)

    return out


def _shift_note_events(note_events: Iterable[NoteEvent], offset_s: float) -> list[NoteEvent]:
    offset_s = float(offset_s or 0.0)
    if abs(offset_s) <= 1e-9:
        return list(note_events)
    out: list[NoteEvent] = []
    for ev in note_events:
        out.append(
            NoteEvent(
                start_time_s=float(ev.start_time_s) - offset_s,
                end_time_s=float(ev.end_time_s) - offset_s,
                pitch_midi=int(ev.pitch_midi),
                velocity=int(ev.velocity),
                amplitude=float(ev.amplitude),
            )
        )
    return out


def _shift_chords(chords: Iterable[ChordSegment], offset_s: float) -> list[ChordSegment]:
    offset_s = float(offset_s or 0.0)
    if abs(offset_s) <= 1e-9:
        return list(chords)
    out: list[ChordSegment] = []
    for c in chords:
        out.append(
            ChordSegment(
                start=float(c.start) - offset_s,
                end=float(c.end) - offset_s,
                label=str(c.label),
                confidence=float(c.confidence),
            )
        )
    return out


def _parse_chord_label(label: str) -> tuple[str | None, str | None]:
    if not label or label == "N":
        return None, None
    if ":" in label:
        root, qual = label.split(":", 1)
        root = root.strip()
        qual = qual.strip().lower() or "maj"
    else:
        root, qual = label.strip(), "maj"
    if not root:
        return None, None
    return root, qual


def _triad_label(root: str, qual: str) -> str:
    if qual in ("min", "m", "min7", "m7"):
        return f"{root}:min"
    return f"{root}:maj"


def _segment_chroma_energy(
    chroma: np.ndarray | None,
    times: np.ndarray | None,
    start: float,
    end: float,
) -> np.ndarray | None:
    if chroma is None or times is None:
        return None
    if chroma.ndim != 2 or times.ndim != 1:
        return None
    if chroma.shape[1] != times.shape[0]:
        return None
    if end <= start:
        return None
    mask = (times >= float(start)) & (times < float(end))
    if not np.any(mask):
        return None
    return np.mean(chroma[:, mask], axis=1)


def _simplify_chord_segments(
    chords: list[ChordSegment],
    *,
    chroma: np.ndarray | None,
    times: np.ndarray | None,
    min_confidence: float,
    min_duration: float,
    seventh_ratio: float,
) -> list[ChordSegment]:
    if not chords:
        return []

    confs = np.asarray([float(c.confidence) for c in chords], dtype=np.float32)
    conf_baseline = float(np.median(confs)) if confs.size > 0 else float(min_confidence)
    conf_threshold = max(float(min_confidence), conf_baseline * 0.9)

    out: list[ChordSegment] = []
    for i, c in enumerate(chords):
        label = str(c.label or "N")
        root, qual = _parse_chord_label(label)
        if root is None or qual is None:
            out.append(c)
            continue

        is_seventh = qual in ("7", "min7", "m7", "maj7")
        if not is_seventh:
            out.append(c)
            continue

        collapse = False
        dur = float(c.end) - float(c.start)
        if dur < float(min_duration) or float(c.confidence) < float(conf_threshold):
            collapse = True

        if not collapse:
            energy = _segment_chroma_energy(chroma, times, float(c.start), float(c.end))
            if energy is not None:
                root_pc = NOTE_TO_PC.get(root)
                if root_pc is not None:
                    third = 3 if qual in ("min7", "m7") else 4
                    triad_pcs = [
                        int((root_pc + 0) % 12),
                        int((root_pc + third) % 12),
                        int((root_pc + 7) % 12),
                    ]
                    triad_energy = float(np.mean([energy[pc] for pc in triad_pcs]))
                    seventh_pc = int((root_pc + (11 if qual == "maj7" else 10)) % 12)
                    seventh_energy = float(energy[seventh_pc])
                    if triad_energy > 1e-6 and seventh_energy < triad_energy * float(seventh_ratio):
                        collapse = True

        if not collapse and 0 < i < (len(chords) - 1):
            prev_root, prev_qual = _parse_chord_label(str(chords[i - 1].label))
            next_root, next_qual = _parse_chord_label(str(chords[i + 1].label))
            if prev_root == root and next_root == root:
                if _triad_label(prev_root, prev_qual or "maj") == _triad_label(root, qual) == _triad_label(next_root, next_qual or "maj"):
                    collapse = True

        if collapse:
            label = _triad_label(root, qual)

        out.append(
            ChordSegment(
                start=float(c.start),
                end=float(c.end),
                label=label,
                confidence=float(c.confidence),
            )
        )

    return out


def _tempo_from_beat_times(beat_times: np.ndarray | None) -> float:
    if beat_times is None or len(beat_times) < 2:
        return 0.0
    diffs = np.diff(np.asarray(beat_times, dtype=np.float64))
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size == 0:
        return 0.0
    # Median is more robust than mean when beat_track jitters.
    return float(60.0 / float(np.median(diffs)))


def _score_complexity_cost(score: ScoreData) -> float:
    """
    Heuristic to choose a beat grid that yields more readable notation.
    Lower is better.
    """
    items = [it for m in (score.measures or []) for it in (m.items or [])]
    if not items:
        return 1e9

    n_items = float(len(items))
    n_measures = float(len(score.measures or []))
    n_short = float(sum(1 for it in items if str(it.duration) in ("16", "32")))
    n_ties = float(sum(1 for it in items if it.tie))
    non_rest = [it for it in items if not it.rest]
    avg_poly = float(np.mean([len(it.keys or []) for it in non_rest])) if non_rest else 0.0

    # Bias toward ~4-8 measures for short clips; avoid excessive fragmentation.
    return (
        n_items
        + 0.85 * n_short
        + 0.25 * n_ties
        + 0.35 * avg_poly
        + 0.6 * abs(n_measures - 6.0)
    )


def _pick_best_beat_times(
    note_events: list[NoteEvent],
    beat_times: np.ndarray | None,
    *,
    time_signature: str,
) -> np.ndarray | None:
    if beat_times is None or len(beat_times) < 2 or not note_events:
        return beat_times

    beats = np.asarray(beat_times, dtype=np.float32)
    beats = beats[np.isfinite(beats)]
    if beats.size < 2:
        return beat_times

    # Keep selection fast on long jobs.
    events = sorted(note_events, key=lambda e: float(e.start_time_s))
    if len(events) > 600:
        # Bias toward higher-confidence events but keep temporal ordering.
        top = sorted(events, key=lambda e: float(e.amplitude), reverse=True)[:600]
        events = sorted(top, key=lambda e: float(e.start_time_s))

    candidates: list[np.ndarray] = [beats]
    if beats.size >= 4:
        candidates.append(beats[::2])
        candidates.append(beats[1::2])

    best_cost = None
    best = beats
    for cand in candidates:
        if cand.size < 2:
            continue
        try:
            q = quantize_note_events_to_score(
                events,
                tempo_bpm=120.0,  # ignored when beat_times is provided
                beat_times=cand,
                time_signature=time_signature,
            )
            cost = float(_score_complexity_cost(q.score))
        except Exception:
            continue

        if best_cost is None or cost < best_cost:
            best_cost = cost
            best = cand

    return best.astype(np.float32)


def run_pipeline(job_dir: Path, input_path: Path) -> JobResult:
    work = job_dir / "work"
    out = job_dir / "out"
    work.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)

    wav_path = work / "audio_mono_44k.wav"
    ffmpeg_to_wav_mono_44k(input_path, wav_path)

    mix_path = wav_path
    transcription_path = mix_path
    beat_path = mix_path
    demucs_error = None
    stems_dir: Path | None = None
    transcription_source = "mix"
    beat_source = "mix"

    if bool(settings.ENABLE_DEMUCS):
        try:
            stems_dir = run_demucs_4stems(
                mix_path,
                work / "demucs",
                model=str(settings.DEMUCS_MODEL),
                return_stem=False,
            )
            transcription_path = select_stem_path(stems_dir, ("other", "vocals"))
            transcription_source = transcription_path.stem
            drums_path = get_stem_path(stems_dir, "drums")
            if drums_path is not None:
                beat_path = drums_path
                beat_source = drums_path.stem
        except Exception as e:
            demucs_error = str(e)
            stems_dir = None
            transcription_path = mix_path
            beat_path = mix_path
            transcription_source = "mix"
            beat_source = "mix"

    y_trans, sr_trans = load_wav(transcription_path)
    y_trans = peak_normalize(y_trans)

    harmonic_path = transcription_path
    y_harm = y_trans
    try:
        y_harm = librosa.effects.harmonic(y_trans)
        y_harm = peak_normalize(y_harm)
        harmonic_path = work / "audio_harmonic.wav"
        sf.write(str(harmonic_path), y_harm, int(sr_trans))
    except Exception as e:
        print(f"Error computing harmonic stem: {e}")
        y_harm = y_trans
        harmonic_path = transcription_path

    y_beats, sr_beats = load_wav(beat_path)
    y_beats = peak_normalize(y_beats)

    time_sig = "4/4"
    tempo_raw, beat_times_raw = estimate_beats_librosa(
        y_beats,
        sr_beats,
        use_harmonic=(beat_source != "drums"),
    )
    tempo_for_bp = float(tempo_raw) if tempo_raw and tempo_raw > 0 else 120.0

    # --- 1. Basic Pitch Transcription ---
    try:
        midi_data, note_events = transcribe_basic_pitch(
            harmonic_path,
            midi_tempo=float(tempo_for_bp),
            onset_threshold=float(settings.BASIC_PITCH_ONSET_THRESHOLD),
            frame_threshold=float(settings.BASIC_PITCH_FRAME_THRESHOLD),
            minimum_note_length_ms=float(settings.BASIC_PITCH_MIN_NOTE_MS),
            melodia_trick=True,
        )
    except Exception as e:
        print(f"Error executing basic_pitch: {e}")
        note_events = []
        midi_data = None

    # Decide whether to keep beats as-is (double-time) or halved (half-time) based on
    # readability of the quantized score. This keeps `tempo_bpm`, `beat_times`, and
    # downstream exports consistent.
    beat_times_sel = _pick_best_beat_times(note_events, beat_times_raw, time_signature=time_sig)
    tempo_bpm = _tempo_from_beat_times(beat_times_sel)
    if not tempo_bpm or tempo_bpm <= 0:
        tempo_bpm = float(tempo_for_bp)

    beat_times_norm, beat_offset = normalize_beat_times(beat_times_sel)
    note_events = _shift_note_events(note_events, beat_offset)

    # --- 2. Chord Detection ---
    chroma, _times, chords = extract_chords_template(
        y_harm,
        sr_trans,
        vocab=str(settings.CHORD_VOCAB),
        switch_penalty=float(settings.SWITCH_PENALTY),
        min_segment_sec=float(settings.MIN_SEGMENT_SEC),
        beat_times=beat_times_sel,
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

    spelled_chords = _simplify_chord_segments(
        spelled_chords,
        chroma=chroma,
        times=_times,
        min_confidence=float(_SEVENTH_MIN_CONFIDENCE),
        min_duration=float(_SEVENTH_MIN_DURATION),
        seventh_ratio=float(_SEVENTH_RATIO),
    )
    spelled_chords = _shift_chords(spelled_chords, beat_offset)

    if note_events:
        note_events = _merge_overlapping_notes(note_events, gap_s=0.03)

        amps = np.asarray([float(ev.amplitude) for ev in note_events], dtype=np.float32)
        if amps.size > 0:
            min_amp = max(0.2, float(np.percentile(amps, 35)))
        else:
            min_amp = 0.2

        sec_per_q = 60.0 / float(tempo_bpm if tempo_bpm else 120.0)
        min_dur_s = max(0.08, 0.2 * sec_per_q)

        chord_conf_threshold = None
        if spelled_chords:
            confs = np.asarray([float(c.confidence) for c in spelled_chords], dtype=np.float32)
            if confs.size > 0:
                chord_conf_threshold = max(float(_CHORD_CONFIDENCE_THRESHOLD), float(np.median(confs)) * 0.9)

        note_events = _filter_note_events(
            note_events,
            chords=spelled_chords,
            min_amp=min_amp,
            min_dur_s=min_dur_s,
            min_pitch=40,
            max_pitch=88,
            chord_tone_bias=float(_CHORD_TONE_BIAS),
            chord_confidence_threshold=chord_conf_threshold,
        )
        note_events = _limit_onset_polyphony(
            note_events,
            max_notes=6,
            onset_window_s=0.06,
        )

    # --- Debug artifacts ---
    try:
        beat_payload = {
            "tempo_bpm": float(tempo_bpm),
            "tempo_raw_bpm": float(tempo_raw or 0.0),
            "beat_times_s": (beat_times_norm.tolist() if beat_times_norm is not None else []),
            "beat_times_raw_s": (beat_times_sel.tolist() if beat_times_sel is not None else []),
            "beat_offset_s": float(beat_offset),
            "beat_source": beat_source,
            "transcription_source": transcription_source,
            "demucs_enabled": bool(settings.ENABLE_DEMUCS),
            "demucs_error": demucs_error,
        }
        (out / "beat_times.json").write_text(
            json.dumps(beat_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        save_note_events_csv(note_events, out / "note_events.csv")
        chords_payload = [c.model_dump() for c in spelled_chords]
        (out / "chords.json").write_text(
            json.dumps(chords_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as e:
        print(f"Error writing debug artifacts: {e}")

    # --- 3. Quantization to ScoreData ---
    quant_res = quantize_note_events_to_score(
        note_events,
        tempo_bpm=float(tempo_bpm),
        beat_times=beat_times_norm,
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
        chords=[Segment(c.start, c.end, c.label, c.confidence) for c in spelled_chords],
        beat_times=beat_times_norm,
        pickup_quarters=float(quant_res.pickup_quarters),
    )

    midi_path = out / "transcription.mid"
    midi_start = 0.0
    if spelled_chords:
        midi_start = min(0.0, min(float(c.start) for c in spelled_chords))
    export_chords_midi(
        midi_path,
        spelled_chords,
        tempo_bpm=float(tempo_bpm),
        time_signature=time_sig,
        root_octave=3,
        start_time_s=float(midi_start),
    )

    # PDF (LilyPond) generation - kept as fallback/supplement
    # Ideally LilyPond service should also be updated to support full score,
    # but for now we focus on MusicXML/OSMD as requested.
    pdf_error = None
    if shutil.which("lilypond") is not None:
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
