# Phase 2 — Data Processing Pipeline

**Project:** Intelligent Anomaly Detection & Incident Prediction Platform
**Status:** ✅ Complete
**Goal:** Transform the synchronous API from Phase 1 into a fully asynchronous, background-processing pipeline using Celery, Redis, and feature engineering.

---

## 1. What Phase 2 Delivers

By the end of this phase, the system does this:

```
API receives metrics payload
        ↓
.delay() → throws task to Redis instantly (non-blocking)
        ↓
API returns 202 immediately
        ↓
Celery worker picks up task from Redis
        ↓
clean_metrics()     → fixes bad/missing values
        ↓
compute_features()  → adds rolling avg, z-score, rate of change
        ↓
Saves 7 enriched rows to metric_datapoints table
        ↓
Every 5 minutes, Celery Beat fires automatically
        ↓
aggregate_metrics() → reads last 5 mins of raw data
        ↓
Saves summary rows to metric_aggregations table
```

Phase 1 wrote to the DB synchronously (slow, blocks the API). Phase 2 makes the whole pipeline asynchronous — the API never touches the database directly.

---

## 2. New Tech Introduced in Phase 2

| Technology | Role | Analogy |
|---|---|---|
| Celery | Background task worker | The kitchen cook |
| Redis (broker) | Task queue — holds tasks until worker picks them up | The ticket board |
| Redis (backend) | Result store — holds task return values | The pickup counter |
| Celery Beat | Scheduler — fires tasks on a timer | The alarm clock |
| `collections.deque` | Fixed-size sliding window for history | Whiteboard with 10 slots |
| `statistics` module | Rolling averages and standard deviation | Built-in Python math |

---

## 3. New Project Structure

```
src/
├── processing/
│   ├── __init__.py         ← makes folder importable
│   ├── cleaning.py         ← fix bad/missing metric values
│   └── features.py         ← compute rolling avg, z-score, rate of change
├── workers/
│   ├── celery_app.py       ← Celery app + Beat schedule
│   └── tasks.py            ← ping, process_metrics, aggregate_metrics
├── models/
│   └── aggregation.py      ← MetricAggregation SQLAlchemy table (NEW)
└── db/
    └── migrations/versions/
        └── xxxx_add_metric_aggregations_table.py
```

---

## 4. Step 1 — Celery + Redis Wiring

### `src/workers/celery_app.py`

```python
from celery import Celery
from celery.schedules import crontab
from src.config import settings

celery_app = Celery(
    "anomaly_platform",
    broker  = settings.REDIS_URL,    # Redis receives tasks
    backend = settings.REDIS_URL,    # Redis stores results
    include = ["src.workers.tasks"], # tells worker where tasks live
)

celery_app.conf.update(
    task_serializer   = "json",
    result_serializer = "json",
    accept_content    = ["json"],
    timezone          = "UTC",
    enable_utc        = True,
)

celery_app.conf.beat_schedule = {
    "aggregate-every-5-minutes": {
        "task"    : "tasks.aggregate_metrics",
        "schedule": crontab(minute="*/5"),
    },
}
```

Key points:
- `broker` = Redis inbox where tasks wait to be picked up
- `backend` = Redis outbox where task results are stored after completion
- `include` is critical — without it the worker starts but doesn't know any tasks exist, causing `NotRegistered` errors
- `beat_schedule` tells Celery Beat which tasks to fire automatically and when

### First task — ping/pong test

```python
@celery_app.task(name="tasks.ping")
def ping():
    return "pong"
```

Used to verify the full Celery ↔ Redis loop works before building real tasks.

### How the loop works

```
.delay()              → sends task to Redis broker
worker picks it up    → runs the function
return value          → stored in Redis backend automatically
result.get()          → fetches stored value from Redis backend
```

You write `.delay()` and `return`. Celery handles everything in between.

---

## 5. Step 2 — Data Cleaning

### `src/processing/cleaning.py`

```python
METRIC_BOUNDS = {
    "cpu_usage":           (0, 100),
    "memory_usage":        (0, 100),
    "request_latency_ms":  (0, None),
    "requests_per_sec":    (0, None),
    "error_rate":          (0, 1),
    "db_connections":      (0, None),
    "disk_usage":          (0, 100),
}

def clean_metrics(metrics: dict) -> dict:
    cleaned = {}
    for name, value in metrics.items():
        low, high = METRIC_BOUNDS.get(name, (None, None))
        if value is None:
            value = low if low is not None else 0
        if low is not None and value < low:
            value = low
        if high is not None and value > high:
            value = high
        cleaned[name] = value
    return cleaned
```

What it fixes:

| Problem | Fix |
|---|---|
| `None` (missing value) | Replace with the lower bound (or 0) |
| Value below minimum | Clip to minimum |
| Value above maximum | Clip to maximum |
| Value in range | Leave unchanged |

Example:
```
Input:   {"cpu_usage": 150, "memory_usage": None, "error_rate": -0.1, "db_connections": 12}
Output:  {"cpu_usage": 100, "memory_usage": 0,    "error_rate": 0,    "db_connections": 12}
```

---

## 6. Step 3 — Feature Engineering

### `src/processing/features.py`

Adds 3 computed features to every raw metric reading:

| Feature | What it means | How it's calculated |
|---|---|---|
| `rolling_avg` | Average of last 10 readings | `statistics.mean(history)` |
| `z_score` | How many standard deviations from normal | `(value - avg) / std` |
| `rate_of_change` | How much it jumped since last reading | `value - history[-1]` |

```python
WINDOW_SIZE = 10
_history: dict[str, deque] = {}

def compute_features(service_name: str, metric_name: str, value: float) -> dict:
    key = f"{service_name}:{metric_name}"
    if key not in _history:
        _history[key] = deque(maxlen=WINDOW_SIZE)

    history = _history[key]

    rate_of_change = value - history[-1] if len(history) > 0 else 0.0
    rolling_avg    = statistics.mean(history) if len(history) > 0 else value

    if len(history) >= 2:
        std = statistics.stdev(history)
        z_score = (value - rolling_avg) / std if std > 0 else 0.0
    else:
        z_score = 0.0

    history.append(value)

    return {
        "value"          : value,
        "rolling_avg"    : round(rolling_avg, 4),
        "z_score"        : round(z_score, 4),
        "rate_of_change" : round(rate_of_change, 4),
    }
```

Key design decision: `history.append(value)` happens **after** all calculations. We always compare the current reading against past readings — not against itself.

Z-score note: when all values are identical, `std = 0` and z-score returns `0.0` (can't divide by zero). In real noisy data this doesn't happen.

---

## 7. Step 4 — Real Celery Task

### `src/workers/tasks.py` — `process_metrics`

```python
@celery_app.task(name="tasks.process_metrics", bind=True, max_retries=3)
def process_metrics(self, payload: dict):
    service_name = payload["service_name"]
    host         = payload["host"]
    timestamp    = datetime.fromisoformat(payload["timestamp"])
    raw_metrics  = payload["metrics"]

    cleaned = clean_metrics(raw_metrics)

    db = SessionLocal()
    try:
        rows_saved = 0
        for metric_name, value in cleaned.items():
            features = compute_features(service_name, metric_name, value)
            db.add(MetricDataPoint(
                timestamp    = timestamp,
                service_name = service_name,
                host         = host,
                metric_name  = metric_name,
                value        = features["value"],
                labels       = {
                    "rolling_avg"    : features["rolling_avg"],
                    "z_score"        : features["z_score"],
                    "rate_of_change" : features["rate_of_change"],
                }
            ))
            rows_saved += 1
        db.commit()
        return {"status": "success", "rows_saved": rows_saved}

    except Exception as exc:
        db.rollback()
        raise self.retry(exc=exc, countdown=5)

    finally:
        db.close()
```

Key points:
- `bind=True` gives the task a `self` reference so it can call `self.retry()`
- `max_retries=3` — if DB is temporarily down, retry up to 3 times
- `countdown=5` — wait 5 seconds before each retry
- `datetime.fromisoformat()` — converts string timestamp back to datetime object (JSON can't carry datetime objects directly)
- The 3 computed features are stored in the `labels` JSON column alongside the raw value
- `finally: db.close()` — always releases the DB connection even on error

---

## 8. Step 5 — API Hands Off to Celery

### `src/api/routes/metrics.py` — `ingest_metrics`

```python
@router.post("", response_model=IngestResponse, status_code=202)
def ingest_metrics(payload: MetricIngest):
    payload_dict = payload.model_dump(mode="json")
    process_metrics.delay(payload_dict)
    return IngestResponse(
        status="accepted",
        message="Metrics received and queued for processing",
        rows_saved=0
    )
```

What changed from Phase 1:
- `db: Session = Depends(get_db)` removed — route no longer touches the DB
- `payload.model_dump(mode="json")` — converts Pydantic model to plain dict with JSON-safe types (datetime → string)
- `.delay()` replaces the direct DB write — task goes to Redis, route returns instantly
- `rows_saved=0` — honest response, rows haven't been saved yet when the API responds

The `202 Accepted` status code (set up in Phase 1) now means exactly what it says: "accepted for processing" — Celery will handle the actual work.

---

## 9. Step 6 — Scheduled Aggregation

### `src/models/aggregation.py` — `MetricAggregation` table

```python
class MetricAggregation(Base):
    __tablename__ = "metric_aggregations"

    id           = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    window_start = Column(DateTime(timezone=True), nullable=False, index=True)
    window_end   = Column(DateTime(timezone=True), nullable=False)
    window_size  = Column(String(20), nullable=False)   # "5min"
    service_name = Column(String(100), nullable=False, index=True)
    metric_name  = Column(String(100), nullable=False)
    avg_value    = Column(Float, nullable=False)
    min_value    = Column(Float, nullable=False)
    max_value    = Column(Float, nullable=False)
    sample_count = Column(Float, nullable=False)
    created_at   = Column(DateTime(timezone=True), server_default=func.now())
```

### `src/workers/tasks.py` — `aggregate_metrics`

```python
@celery_app.task(name="tasks.aggregate_metrics")
def aggregate_metrics():
    db = SessionLocal()
    try:
        now          = datetime.now(timezone.utc)
        window_end   = now
        window_start = now - timedelta(minutes=5)

        services = db.query(MetricDataPoint.service_name).filter(
            MetricDataPoint.timestamp >= window_start,
            MetricDataPoint.timestamp <= window_end,
        ).distinct().all()

        rows_saved = 0
        for (service_name,) in services:
            metrics = db.query(MetricDataPoint.metric_name).filter(
                MetricDataPoint.service_name == service_name,
                MetricDataPoint.timestamp    >= window_start,
                MetricDataPoint.timestamp    <= window_end,
            ).distinct().all()

            for (metric_name,) in metrics:
                stats = db.query(
                    sa_func.avg(MetricDataPoint.value),
                    sa_func.min(MetricDataPoint.value),
                    sa_func.max(MetricDataPoint.value),
                    sa_func.count(MetricDataPoint.value),
                ).filter(
                    MetricDataPoint.service_name == service_name,
                    MetricDataPoint.metric_name  == metric_name,
                    MetricDataPoint.timestamp    >= window_start,
                    MetricDataPoint.timestamp    <= window_end,
                ).one()

                avg_val, min_val, max_val, count = stats
                if count == 0:
                    continue

                db.add(MetricAggregation(
                    window_start = window_start,
                    window_end   = window_end,
                    window_size  = "5min",
                    service_name = service_name,
                    metric_name  = metric_name,
                    avg_value    = round(avg_val, 4),
                    min_value    = round(min_val, 4),
                    max_value    = round(max_val, 4),
                    sample_count = count,
                ))
                rows_saved += 1

        db.commit()
        return {"status": "success", "aggregations_saved": rows_saved}

    except Exception as exc:
        db.rollback()
        raise exc

    finally:
        db.close()
```

How it works:
- Looks back 5 minutes from now
- Finds every distinct `service_name` that sent data in that window
- For each service, finds every distinct `metric_name`
- Runs one SQL query per combination to get avg/min/max/count
- Saves one summary row per combination
- Beat fires this automatically every 5 minutes — no API call needed

### Raw data vs aggregated data

```
metric_datapoints (raw):         metric_aggregations (summary):
360 rows/service/hour            12 rows/service/hour
                                 (one per 5-min window)
```

---

## 10. Running the Full Phase 2 Stack

You need 3 terminals running simultaneously:

```
Terminal 1 — API server:
uvicorn src.main:app --reload --port 8080

Terminal 2 — Celery worker:
celery -A src.workers.celery_app.celery_app worker --loglevel=info

Terminal 3 — Celery Beat:
celery -A src.workers.celery_app.celery_app beat --loglevel=info
```

Always restart the worker after adding new tasks to `tasks.py` — it only reads the file once at startup.

---

## 11. What Was Verified

- ✅ Celery ↔ Redis ping/pong round trip working
- ✅ `clean_metrics()` correctly clips out-of-range and fills missing values
- ✅ `compute_features()` correctly returns rolling_avg, z_score, rate_of_change
- ✅ `process_metrics` Celery task cleans, enriches, and saves 7 rows per payload
- ✅ `POST /api/v1/metrics` returns `202` immediately without touching DB directly
- ✅ Celery Beat fires `aggregate_metrics` automatically every 5 minutes
- ✅ `metric_aggregations` table receives summary rows without any manual trigger

---

## 12. Key Design Decisions

- **Asynchronous ingest** — API returns instantly, Celery handles DB writes in background. Enables horizontal scaling: run more workers to handle more throughput without touching the API.
- **`bind=True` + `max_retries=3`** — tasks survive temporary DB outages by retrying with a delay instead of losing data.
- **Features stored in `labels` JSON column** — no schema change needed to store computed features alongside raw values. The existing column handles it.
- **`mode="json"` on model_dump** — ensures datetime objects become ISO strings before travelling through Redis. Redis only speaks JSON.
- **Beat as a separate process** — scheduler and worker are decoupled. You can run multiple workers for throughput without running multiple schedulers.
- **`distinct()` in aggregation** — prevents duplicate processing. A service with 60 raw rows still appears once in the service list.

---

## 13. What's Next — Phase 3 Preview

Phase 3 (ML Models) will:
- Train an **Isolation Forest** model on the feature-engineered data — detects anomalies by finding points that are "isolated" from the normal cluster
- Train an **LSTM Autoencoder** — learns normal time-series patterns and flags readings it can't reconstruct well
- Add a **z-score statistical baseline** — a fast, lightweight anomaly check that runs without ML
- Combine all three into an **ensemble** — a reading is flagged as anomalous only when multiple models agree
- Wire the anomaly detection into the Celery pipeline so every incoming metric is scored automatically
