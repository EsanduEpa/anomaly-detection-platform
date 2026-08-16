from fastapi import APIRouter

router = APIRouter(tags=["Health"])

@router.get("/health")
def health_check():
    """Simple health check to confirm the server is running."""
    return {"status": "ok", "message": "Anomaly Detection Platform is running"}