from celery import Celery
from app.core.config import settings

celery = Celery(
    "chord_extractor",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=["app.workers.tasks"],
)

celery.conf.update(
    task_routes={
        "app.workers.tasks.process_job": {"queue": "gpu"},
    },
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
)
