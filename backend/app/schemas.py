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

class KeySignature(BaseModel):
    tonic: str
    mode: Literal["major", "minor"]
    fifths: int
    name: str
    vexflow: str
    use_flats: bool
    score: float

class TupletSpec(BaseModel):
    num_notes: int
    notes_occupied: int

class ScoreItem(BaseModel):
    rest: bool = False
    keys: List[str] = []
    duration: str
    dots: int = 0
    tuplet: Optional[TupletSpec] = None
    tie: Optional[Literal["start", "stop", "continue"]] = None

class ScoreMeasure(BaseModel):
    number: int
    items: List[ScoreItem]

class ScoreData(BaseModel):
    grid_q: float
    grid_kind: Literal["straight", "triplet"]
    measures: List[ScoreMeasure]

class JobResult(BaseModel):
    job_id: str
    tempo_bpm: float
    time_signature: str
    key_signature: Optional[KeySignature] = None
    chords: List[ChordSegment] = []
    transcription_backend: Optional[str] = None
    transcription_error: Optional[str] = None
    score: Optional[ScoreData] = None
