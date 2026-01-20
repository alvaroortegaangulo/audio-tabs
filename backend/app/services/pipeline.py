from __future__ import annotations

import json
from pathlib import Path

from app.core.config import settings
from app.schemas import JobResult, KeySignature, ChordSegment
from app.services.audio import ffmpeg_to_wav_mono_44k, load_wav, peak_normalize
from app.services.chords.extract import extract_chords_template
from app.services.grid.beats import estimate_beats_librosa
from app.services.engraving.lilypond import build_lilypond_score, render_lilypond_pdf
from app.services.midi.export import export_chords_midi
from app.services.musicxml.lead_sheet import export_lead_sheet_musicxml
from app.services.theory.key import estimate_key_from_chroma, spell_chord_label


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

    chroma, _times, chords = extract_chords_template(
        y,
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
    spelled: list[ChordSegment] = []
    for c in chords:
        spelled.append(
            ChordSegment(
                start=float(c.start),
                end=float(c.end),
                label=spell_chord_label(str(c.label), use_flats=use_flats),
                confidence=float(c.confidence),
            )
        )

    title = _job_title(job_dir, input_path)

    musicxml_path = out / "result.musicxml"
    export_lead_sheet_musicxml(
        musicxml_path,
        spelled,
        tempo_bpm=float(tempo_bpm),
        time_signature=time_sig,
        key_signature_fifths=(key_sig.fifths if key_sig else None),
        title=title,
        instrument="guitar",
    )

    midi_path = out / "transcription.mid"
    export_chords_midi(
        midi_path,
        spelled,
        tempo_bpm=float(tempo_bpm),
        time_signature=time_sig,
        root_octave=3,
    )

    # PDF (LilyPond) best-effort: no fallar el job si LilyPond no está disponible.
    try:
        ly = build_lilypond_score(
            spelled,
            tempo_bpm=float(tempo_bpm),
            time_signature=time_sig,
            key_tonic=(key_sig.tonic if key_sig else None),
            key_mode=(key_sig.mode if key_sig else "major"),
            title=title,
        )
        render_lilypond_pdf(ly, out, basename="score")
        pdf_error = None
    except Exception as e:
        pdf_error = f"No se pudo generar PDF (LilyPond): {e}"

    return JobResult(
        job_id=job_dir.name,
        tempo_bpm=float(tempo_bpm),
        time_signature=time_sig,
        key_signature=key_sig,
        chords=spelled,
        transcription_backend="chords_template_viterbi",
        transcription_error=pdf_error,
        score=None,
    )
