from pydantic import BaseModel
from typing import Literal, Optional, List

JobStatus = Literal["queued", "running", "done", "error"]

class JobCreateResponse(BaseModel):
    job_id: str
    status: JobStatus

class JobInfo(BaseModel):
    job_id: str
    status: JobStatus
    error: Optional[str] = None

class ChordSegment(BaseModel):
    start: float
    end: float
    label: str
    confidence: float

class JobResult(BaseModel):
    job_id: str
    tempo_bpm: float
    time_signature: str
    chords: List[ChordSegment]
