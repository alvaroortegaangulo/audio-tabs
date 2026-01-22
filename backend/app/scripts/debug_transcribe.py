from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


def _resolve_backend_root() -> Path:
    # app/scripts/debug_transcribe.py -> app -> backend
    return Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the transcription pipeline on a local WAV.")
    parser.add_argument("wav", help="Path to input WAV/MP3")
    parser.add_argument(
        "--job-dir",
        help="Output job directory (defaults to backend/data/jobs/debug_cli)",
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="Do not delete an existing job directory",
    )
    args = parser.parse_args()

    wav_path = Path(args.wav).expanduser().resolve()
    if not wav_path.exists():
        print(f"Input not found: {wav_path}")
        return 2

    backend_root = _resolve_backend_root()
    sys.path.insert(0, str(backend_root))

    from app.services.pipeline import run_pipeline  # pylint: disable=import-error

    if args.job_dir:
        job_dir = Path(args.job_dir).expanduser().resolve()
    else:
        job_dir = backend_root / "data" / "jobs" / "debug_cli"

    if job_dir.exists() and not args.keep:
        shutil.rmtree(job_dir)

    (job_dir / "input").mkdir(parents=True, exist_ok=True)
    (job_dir / "input" / "meta.json").write_text(
        json.dumps({"filename": wav_path.name, "path": str(wav_path)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    result = run_pipeline(job_dir, wav_path)
    (job_dir / "out" / "result.json").write_text(
        json.dumps(result.model_dump(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"done: {job_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
