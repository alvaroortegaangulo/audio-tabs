from __future__ import annotations

# Monkey patch numpy ANTES de cualquier otro import
import app.compat.numpy_compat  # noqa: F401

from pathlib import Path
from celery.utils.log import get_task_logger

from app.workers.celery_app import celery
from app.core.config import settings
from app.services.storage.local import LocalStorage
from app.services.pipeline import run_pipeline

logger = get_task_logger(__name__)
storage = LocalStorage(settings.DATA_DIR)

@celery.task(name="app.workers.tasks.process_job")
def process_job(job_id: str) -> dict:
    job_dir = storage.job_dir(job_id)
    status_path = job_dir / "status.json"

    def set_status(st: str, error: str | None = None):
        storage.write_json(status_path, {"job_id": job_id, "status": st, "error": error})

    try:
        set_status("running")
        input_meta = storage.read_json(job_dir / "input" / "meta.json")
        input_path = Path(input_meta["path"])
        result = run_pipeline(job_dir, input_path)

        # guarda resultado json
        storage.write_json(job_dir / "out" / "result.json", result.model_dump())
        set_status("done")
        return {"ok": True}
    except Exception as e:
        logger.exception("Job failed")
        set_status("error", error=str(e))
        return {"ok": False, "error": str(e)}
