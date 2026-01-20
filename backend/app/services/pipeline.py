from __future__ import annotations
from pathlib import Path
import numpy as np
import librosa

from app.core.config import settings
from app.schemas import ChordSegment, JobResult
from app.services.audio import ffmpeg_to_wav_mono_44k, load_wav, peak_normalize
from app.services.grid.beats import estimate_beats_librosa
from app.services.chords.template import (
    build_chord_library, chroma_features, emission_probs,
    frames_to_segments, finalize_segments
)
from app.services.chords.viterbi import viterbi_decode
from app.services.musicxml.export import export_musicxml
from app.services.engraving.lilypond import build_lilypond_score, render_lilypond_pdf, QuantCfg


def run_pipeline(job_dir: Path, input_path: Path) -> JobResult:
    work = job_dir / "work"
    out = job_dir / "out"
    work.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)

    wav_path = work / "audio_mono_44k.wav"
    ffmpeg_to_wav_mono_44k(input_path, wav_path)

    y, sr = load_wav(wav_path)
    y = peak_normalize(y)

    # (v1) Sin Demucs por defecto (feature-flag listo para v2)
    # Si ENABLE_DEMUCS=1, aquí mezclarías "other"+"bass" y usarías esa señal.
    # Demucs v4 (htdemucs/htdemucs_ft) documentado. :contentReference[oaicite:6]{index=6}

    tempo_bpm, beat_times = estimate_beats_librosa(y, sr)

    hop_length = 512
    chroma = chroma_features(y, sr, hop_length=hop_length)
    labels, T = build_chord_library(settings.CHORD_VOCAB)
    probs = emission_probs(chroma, labels, T)

    path, conf = viterbi_decode(probs, switch_penalty=float(settings.SWITCH_PENALTY))

    times = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr, hop_length=hop_length).astype(np.float32)
    raw = frames_to_segments(path, conf, times, min_len=float(settings.MIN_SEGMENT_SEC))
    segs = finalize_segments(raw, labels)

    chords = [ChordSegment(start=s.start, end=s.end, label=s.label, confidence=s.confidence) for s in segs]
    time_sig = "4/4"

    musicxml_path = out / "result.musicxml"
    export_musicxml(musicxml_path, chords=chords, tempo_bpm=float(tempo_bpm), time_signature=time_sig)

    # PDF "Real Book" (engraving profesional con LilyPond)
    try:
        ly = build_lilypond_score(
            chords=chords,
            tempo_bpm=float(tempo_bpm),
            time_signature=time_sig,
            title="Chord Extractor",
            composer="",
            cfg=QuantCfg(grid_q=0.5),  # corchea por defecto (tu requisito)
            rehearsal_every_measures=8,
        )
        render_lilypond_pdf(ly, out, basename="score")
    except Exception as e:
        # No rompemos el job si lilypond no está disponible; nos quedamos con MusicXML
        # (en docker sí estará)
        pass

    return JobResult(
        job_id=job_dir.name,
        tempo_bpm=float(tempo_bpm),
        time_signature=time_sig,
        chords=chords,
    )
