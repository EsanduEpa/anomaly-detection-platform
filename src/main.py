from fastapi import FastAPI

# This creates our "waiter" (the web server app)
app = FastAPI(
    title="Anomaly Detection API",
    description="API for collecting metrics and detecting anomalies",
    version="1.0.0"
)

# This is a "Route". 
# It tells the waiter what to do when someone visits the "/health" address.
@app.get("/health")
def health_check():
    return {"status": "ok", "message": "Server is running perfectly!"}