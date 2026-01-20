from __future__ import annotations

from pathlib import Path

from app.core.config import settings
from app.schemas import JobResult
from app.services.amt.basic_pitch import save_note_events_csv, transcribe_basic_pitch
from app.services.audio import ffmpeg_to_wav_mono_44k, load_wav, peak_normalize
from app.services.grid.beats import estimate_beats_librosa
from app.services.musicxml.export import export_musicxml
from app.services.theory.quantize import quantize_note_events_to_score


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

    musicxml_path = out / "result.musicxml"
    export_musicxml(
        musicxml_path,
        quant.score,
        tempo_bpm=float(tempo_bpm),
        time_signature=time_sig,
        key_signature_fifths=(quant.key_signature.fifths if quant.key_signature else None),
        title="Audio Tabs",
        instrument="piano",
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

