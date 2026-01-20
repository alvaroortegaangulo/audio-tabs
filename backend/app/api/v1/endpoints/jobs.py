from __future__ import annotations
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path
import uuid

from app.core.config import settings
from app.schemas import JobCreateResponse, JobInfo
from app.services.storage.local import LocalStorage
from app.workers.tasks import process_job

router = APIRouter()
storage = LocalStorage(settings.DATA_DIR)

@router.post("", response_model=JobCreateResponse)
async def create_job(file: UploadFile = File(...)):
    job_id = uuid.uuid4().hex
    job_dir = storage.job_dir(job_id)
    inp = job_dir / "input"
    inp.mkdir(parents=True, exist_ok=True)

    raw_path = inp / f"upload{Path(file.filename or '').suffix}"
    max_bytes = int(settings.MAX_UPLOAD_MB) * 1024 * 1024
    bytes_written = 0
    chunk_size = 1024 * 1024
    try:
        with raw_path.open("wb") as out:
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                bytes_written += len(chunk)
                if bytes_written > max_bytes:
                    raise HTTPException(413, f"Archivo supera {settings.MAX_UPLOAD_MB} MB")
                out.write(chunk)
    except HTTPException:
        if raw_path.exists():
            raw_path.unlink()
        raise
    finally:
        await file.close()
    storage.write_json(inp / "meta.json", {"path": str(raw_path), "filename": file.filename})

    storage.write_json(job_dir / "status.json", {"job_id": job_id, "status": "queued", "error": None})

    if settings.CELERY_ENABLED:
        process_job.delay(job_id)
        return JobCreateResponse(job_id=job_id, status="queued")
    else:
        # modo simple: ejecuta inline (no recomendado en prod)
        from app.services.pipeline import run_pipeline
        try:
            storage.write_json(job_dir / "status.json", {"job_id": job_id, "status": "running", "error": None})
            run_pipeline(job_dir, raw_path)
            storage.write_json(job_dir / "status.json", {"job_id": job_id, "status": "done", "error": None})
            return JobCreateResponse(job_id=job_id, status="done")
        except Exception as e:
            storage.write_json(job_dir / "status.json", {"job_id": job_id, "status": "error", "error": str(e)})
            return JobCreateResponse(job_id=job_id, status="error")

@router.get("/{job_id}", response_model=JobInfo)
def get_job(job_id: str):
    job_dir = storage.job_dir(job_id)
    status_path = job_dir / "status.json"
    if not status_path.exists():
        raise HTTPException(404, "Job not found")
    st = storage.read_json(status_path)
    return JobInfo(**st)

@router.get("/{job_id}/musicxml")
def get_musicxml(job_id: str):
    job_dir = storage.job_dir(job_id)
    xml_path = job_dir / "out" / "result.musicxml"
    if not xml_path.exists():
        raise HTTPException(404, "MusicXML not ready")
    return FileResponse(str(xml_path), media_type="application/xml", filename="result.musicxml")

@router.get("/{job_id}/result.json")
def get_result(job_id: str):
    job_dir = storage.job_dir(job_id)
    p = job_dir / "out" / "result.json"
    if not p.exists():
        raise HTTPException(404, "Result not ready")
    return storage.read_json(p)

@router.get("/{job_id}/score.pdf")
def get_score_pdf(job_id: str):
    job_dir = storage.job_dir(job_id)
    pdf_path = job_dir / "out" / "score.pdf"
    if not pdf_path.exists():
        raise HTTPException(404, "Score PDF not ready")
    return FileResponse(str(pdf_path), media_type="application/pdf", filename="score.pdf")

@router.get("/{job_id}/transcription.mid")
def get_transcription_midi(job_id: str):
    job_dir = storage.job_dir(job_id)
    midi_path = job_dir / "out" / "transcription.mid"
    if not midi_path.exists():
        raise HTTPException(404, "MIDI transcription not ready")
    return FileResponse(str(midi_path), media_type="audio/midi", filename="transcription.mid")

@router.get("/{job_id}/note_events.csv")
def get_note_events(job_id: str):
    job_dir = storage.job_dir(job_id)
    p = job_dir / "out" / "note_events.csv"
    if not p.exists():
        raise HTTPException(404, "Note events not ready")
    return FileResponse(str(p), media_type="text/csv", filename="note_events.csv")
