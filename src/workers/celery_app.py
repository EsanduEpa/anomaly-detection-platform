from celery import Celery
from celery.schedules import crontab
from src.config import settings

# This is the "kitchen manager" object.
# It knows where the ticket board is (broker) and where to write finished results (backend).
celery_app = Celery(
    "anomaly_platform",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=["src.workers.tasks"],   # <-- new: "read this recipe book on startup"
)

# A few sane defaults — don't worry too much about memorizing these,
# just know: tasks and their results get sent around as JSON, and we use UTC everywhere.
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
)

# Alarm clock — runs aggregate_metrics every 5 minutes automatically
celery_app.conf.beat_schedule = {
    "aggregate-metrics-every-5-minutes": {
        "task": "tasks.aggregate_metrics",
        "schedule": crontab(minute="*/5"),  # every 5 minutes
    },
}