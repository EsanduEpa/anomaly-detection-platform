from src.workers.celery_app import celery_app

@celery_app.task(name="tasks.ping")
def ping():
    """A tiny test task — just proves a Celery worker can receive and finish a job."""
    return "pong"