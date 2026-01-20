from __future__ import annotations

from pathlib import Path

from app.core.config import settings
from app.schemas import JobResult
from app.services.amt.basic_pitch import save_note_events_csv, transcribe_basic_pitch
from app.services.audio import ffmpeg_to_wav_mono_44k, load_wav, peak_normalize
from app.services.grid.beats import estimate_beats_librosa
from app.services.musicxml.export import export_musicxml
from app.services.theory.quantize import quantize_note_events_to_score

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


def run_pipeline(job_dir: Path, input_path: Path) -> JobResult:
    work = job_dir / "work"
    out = job_dir / "out"
    work.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)

    wav_path = work / "audio_mono_44k.wav"
    ffmpeg_to_wav_mono_44k(input_path, wav_path)

    y, sr = load_wav(wav_path)
    y = peak_normalize(y)

    tempo_bpm, _beat_times = estimate_beats_librosa(y, sr)
    if not tempo_bpm or tempo_bpm <= 0:
        tempo_bpm = 120.0

    if not settings.ENABLE_BASIC_PITCH:
        raise RuntimeError("ENABLE_BASIC_PITCH=0: Basic Pitch es obligatorio en este pipeline.")

    # --- 1. Transcription ---
    midi_data, note_events = transcribe_basic_pitch(
        wav_path,
        midi_tempo=float(tempo_bpm),
        onset_threshold=float(settings.BASIC_PITCH_ONSET_THRESHOLD),
        frame_threshold=float(settings.BASIC_PITCH_FRAME_THRESHOLD),
        minimum_note_length_ms=float(settings.BASIC_PITCH_MIN_NOTE_MS),
        melodia_trick=True,
    )

    midi_path = out / "transcription.mid"
    midi_data.write(str(midi_path))
    save_note_events_csv(note_events, out / "note_events.csv")

    time_sig = "4/4"
    quant = quantize_note_events_to_score(note_events, tempo_bpm=float(tempo_bpm), time_signature=time_sig)

    # --- 2. Chord Detection ---
    chords = detect_chords(y, sr)

    # --- 3. Export MusicXML (Guitar Lead Sheet) ---
    musicxml_path = out / "result.musicxml"
    export_musicxml(
        musicxml_path,
        quant.score,
        tempo_bpm=float(tempo_bpm),
        time_signature=time_sig,
        key_signature_fifths=(quant.key_signature.fifths if quant.key_signature else None),
        title="Audio Tabs",
        instrument="guitar",  # Changed to guitar for Tab/Chords logic
        chords=chords,        # Pass detected chords
    )

    return JobResult(
        job_id=job_dir.name,
        tempo_bpm=float(tempo_bpm),
        time_signature=time_sig,
        key_signature=quant.key_signature,
        transcription_backend="basic_pitch",
        transcription_error=None,
        score=quant.score,
    )
