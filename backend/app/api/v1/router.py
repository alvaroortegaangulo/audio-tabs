from fastapi import APIRouter
from app.api.v1.endpoints.jobs import router as jobs_router

router = APIRouter(prefix="/v1")
router.include_router(jobs_router, prefix="/jobs", tags=["jobs"])
