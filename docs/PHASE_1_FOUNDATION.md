# Phase 1 — Foundation

**Project:** Intelligent Anomaly Detection & Incident Prediction Platform
**Status:** ✅ Complete
**Goal:** Build the skeleton and backbone of the entire system — project structure, databases, configuration, a working ingestion API, and a realistic data simulator.

---

## 1. What Phase 1 Delivers

By the end of this phase, the system can do one full, real loop:

```
Simulator generates fake server metrics
        ↓ HTTP POST
FastAPI validates the request (Pydantic)
        ↓
SQLAlchemy writes rows
        ↓
PostgreSQL + TimescaleDB stores them permanently
```

No ML, no background workers, and no dashboard yet — that's Phases 2–7. Phase 1 proves the foundation is solid: a working API, a real database with version-controlled schema, and realistic-looking test data flowing through it end to end.

---

## 2. Tech Stack Used in This Phase

| Layer | Technology | Why |
|---|---|---|
| Web framework | FastAPI | Async-ready, auto-generates interactive docs at `/docs`, built-in request validation |
| Data validation | Pydantic v2 | Type-checked request/response schemas, fails loudly on bad input |
| Config management | pydantic-settings | Reads `.env` into a typed, validated `Settings` object |
| ORM | SQLAlchemy 2.0 | Maps Python classes to Postgres tables |
| Migrations | Alembic | Version-controls the database schema like Git version-controls code |
| Database | PostgreSQL + TimescaleDB | Postgres reliability + TimescaleDB's time-series optimizations |
| Cache/queue (provisioned, not yet used) | Redis | Will back Celery in Phase 2 |
| Containerization | Docker Compose | One command (`docker compose up`) starts Postgres + Redis identically on any machine |

---

## 3. Project Structure

```
anomaly-detection-platform/
├── .env                  # Real secrets (gitignored)
├── .env.example           # Template for other developers
├── .gitignore
├── docker-compose.yml      # Postgres + Redis containers
├── requirements.txt
├── alembic.ini
├── src/
│   ├── config.py           # Settings loaded from .env
│   ├── main.py              # FastAPI app entrypoint
│   ├── api/
│   │   └── routes/
│   │       ├── health.py
│   │       └── metrics.py
│   ├── db/
│   │   ├── session.py       # Engine, SessionLocal, Base, get_db()
│   │   └── migrations/      # Alembic env + versioned migration scripts
│   ├── models/
│   │   ├── metric.py         # MetricDataPoint SQLAlchemy table
│   │   └── alert.py          # Alert SQLAlchemy table
│   └── schemas/
│       ├── metric.py         # Pydantic input/output schemas
│       └── alert.py
├── simulator/
│   ├── generator.py         # Realistic fake metric generation
│   └── runner.py             # Sends metrics every 5 seconds
└── tests/
```

---

## 4. Docker — Running Postgres & Redis

`docker-compose.yml` defines two containers:

```yaml
services:
  db:
    image: timescale/timescaledb:latest-pg14
    container_name: anomaly_db
    environment:
      - POSTGRES_USER=admin
      - POSTGRES_PASSWORD=secretpassword
      - POSTGRES_DB=anomaly_db
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data
    restart: always

  redis:
    image: redis:7
    container_name: anomaly_redis
    ports:
      - "6379:6379"
    restart: always

volumes:
  pgdata:
```

Key points:
- The `timescale/timescaledb` image is Postgres with the TimescaleDB extension pre-installed — no separate install step needed.
- The `pgdata` named volume means data survives container restarts (`docker compose down` doesn't wipe the database; only `docker compose down -v` would).
- `restart: always` means both containers come back up automatically if they crash or the machine reboots.
- Redis is running and healthy but **not used yet** — it's provisioned ahead of Phase 2, where Celery will use it as a task queue.

---

## 5. Configuration — `src/config.py`

```python
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    DATABASE_URL: str
    REDIS_URL: str
    APP_NAME: str = "Anomaly Detection Platform"
    DEBUG: bool = True

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore"
    )

settings = Settings()
```

- `DATABASE_URL` and `REDIS_URL` have no defaults → they are **required**. If `.env` is missing them, the app fails immediately at import time with a clear validation error, instead of failing mysteriously later.
- `extra="ignore"` lets `.env` contain Docker-only variables (`POSTGRES_USER`, `POSTGRES_PASSWORD`, etc.) without Pydantic rejecting the whole file.
- `settings` is created once at import time and reused everywhere (`from src.config import settings`) — a singleton, so `.env` is only ever parsed once.

---

## 6. Database Layer

### `src/db/session.py` — connection plumbing

```python
engine = create_engine(settings.DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

- `engine` — the actual connection pool to Postgres, created once.
- `SessionLocal` — a factory; each API request calls it to get its own isolated session/transaction.
- `Base` — parent class every ORM model inherits from.
- `get_db()` — a FastAPI dependency. It hands a session to the route via `yield`, then guarantees the session is closed afterward (even on error) via `finally`.

### `src/models/metric.py` — `MetricDataPoint` table

```python
class MetricDataPoint(Base):
    __tablename__ = "metric_datapoints"
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    service_name = Column(String(100), nullable=False, index=True)
    host = Column(String(100), nullable=False)
    metric_name = Column(String(100), nullable=False)
    value = Column(Float, nullable=False)
    labels = Column(JSON, nullable=True)
    ingested_at = Column(DateTime(timezone=True), server_default=func.now())
```

- **Narrow/long table design**: one row per metric per reading (not one row per timestamp with 7 columns). This is the standard shape for time-series data — it lets TimescaleDB partition and index efficiently by `metric_name`, and new metric types can be added later with zero schema changes.
- `BigInteger` primary key (not `Integer`) — at 3 services × 7 metrics every 5 seconds, a regular 32-bit integer would overflow within a few years in production.

### `src/models/alert.py` — `Alert` table

```python
class Alert(Base):
    __tablename__ = "alerts"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    severity = Column(String(20), nullable=False)       # INFO / WARNING / CRITICAL
    status = Column(String(20), default="ACTIVE")         # ACTIVE / ACKNOWLEDGED / RESOLVED
    service_name = Column(String(100), nullable=False)
    host = Column(String(100), nullable=False)
    metric_name = Column(String(100), nullable=False)
    anomaly_score = Column(Float, nullable=False)
    explanation_text = Column(Text, nullable=True)
    contributing_features = Column(JSON, nullable=True)
```

- UUID primary key instead of an auto-incrementing integer — prevents anyone from enumerating alerts by guessing sequential IDs (`/alerts/1`, `/alerts/2`, ...). This table isn't populated yet; it's ready for Phase 5 (Alert & Incident Management).

### Alembic — schema version control

`src/db/migrations/env.py` points Alembic at `settings.DATABASE_URL` and at `Base.metadata` (which includes both `MetricDataPoint` and `Alert`). Running `alembic revision --autogenerate` compared the models against the live database and generated:

`88789e7a81c0_create_metrics_and_alerts_tables.py` — creates both tables plus indexes on `metric_datapoints.service_name` and `metric_datapoints.timestamp`. Every migration has an `upgrade()` and a `downgrade()`, so schema changes are reversible and reproducible across every developer's machine and every environment (dev, staging, prod).

Verified in Postgres: tables `metric_datapoints`, `alerts`, and Alembic's own bookkeeping table `alembic_version` all exist.

---

## 7. API Layer

### `src/main.py` — app entrypoint

```python
app = FastAPI(
    title="Anomaly Detection API",
    description="API for collecting metrics and detecting anomalies",
    version="1.0.0"
)
app.include_router(health.router)
app.include_router(metrics.router)
```

Routers keep endpoints organized by feature area instead of piling everything into one file — `health.py` and `metrics.py` are self-contained modules registered onto the main app.

### Pydantic schemas — the validation boundary

`src/schemas/metric.py` splits **input** shapes from **output** shapes:

```python
class MetricsPayload(BaseModel):
    cpu_usage: float = Field(..., ge=0, le=100)
    memory_usage: float = Field(..., ge=0, le=100)
    request_latency_ms: float = Field(..., ge=0)
    requests_per_sec: float = Field(..., ge=0)
    error_rate: float = Field(..., ge=0, le=1)
    db_connections: int = Field(..., ge=0)
    disk_usage: float = Field(..., ge=0, le=100)

class MetricIngest(BaseModel):
    service_name: str
    host: str
    timestamp: datetime
    metrics: MetricsPayload

class MetricResponse(BaseModel):
    id: int
    service_name: str
    host: str
    metric_name: str
    value: float
    timestamp: datetime
    model_config = {"from_attributes": True}
```

`Field(..., ge=0, le=100)` enforces range constraints automatically — invalid data (e.g. `cpu_usage: 150`) is rejected with `422 Unprocessable Entity` before any route code runs. `MetricResponse` deliberately exposes fewer fields than the database model has (no `ingested_at`, no `labels`) — the API's public contract is intentionally decoupled from the internal storage shape.

### `POST /api/v1/metrics` — ingestion

```python
@router.post("", response_model=IngestResponse, status_code=status.HTTP_202_ACCEPTED)
def ingest_metrics(payload: MetricIngest, db: Session = Depends(get_db)):
    metrics_dict = payload.metrics.model_dump()
    rows_saved = 0
    for metric_name, value in metrics_dict.items():
        db.add(MetricDataPoint(
            timestamp=payload.timestamp,
            service_name=payload.service_name,
            host=payload.host,
            metric_name=metric_name,
            value=value
        ))
        rows_saved += 1
    db.commit()
    return IngestResponse(status="accepted", message="Metrics received and saved successfully", rows_saved=rows_saved)
```

One incoming payload (1 timestamp, 1 service, 7 metric values) is unpacked into 7 separate `MetricDataPoint` rows. Status code `202 Accepted` (not `200 OK`) is a deliberate forward-looking choice — in Phase 2 this endpoint will hand off to a Celery background task instead of writing synchronously, and `202` is the honest signal for "accepted for processing" even before that change lands.

### `GET /api/v1/metrics` — querying

Supports optional filters (`service_name`, `metric_name`, `from_time`, `to_time`, `limit`) that are chained onto a single SQLAlchemy query lazily, so only one SQL statement executes no matter how many filters are applied.

### `GET /health`

Simple liveness check returning `{"status": "ok", ...}` — used to confirm the server is up before wiring in monitoring/orchestration later.

Interactive docs auto-generate at `/docs` (Swagger UI) from the FastAPI app + Pydantic schemas — no extra work required.

---

## 8. Data Simulator

### `simulator/generator.py` — realistic fake metrics

```python
def get_time_of_day_factor():
    hour = datetime.now().hour
    factor = 0.5 + 0.5 * math.sin(math.pi * (hour - 2) / 12)
    return max(0.3, min(1.0, factor))
```

A sine wave that troughs around 2am and peaks around 2pm, modeling real diurnal traffic patterns.

```python
def add_noise(value, noise_level=0.05):
    noise = random.gauss(0, noise_level * value)
    return value + noise
```

Gaussian noise layered on top so values look organically noisy rather than perfectly flat — important groundwork for Phase 3, where ML models need to learn what "noisy but normal" looks like.

```python
def generate_cpu_spike_metrics():
    base = generate_normal_metrics()
    base["cpu_usage"] = add_noise(92)
    base["request_latency_ms"] = add_noise(800)
    base["error_rate"] = add_noise(0.15, 0.2)
    return base
```

Models a **cascading failure**: CPU pressure pushed up together with correlated latency and error-rate spikes, rather than one metric moving in isolation — the kind of multi-metric pattern the anomaly detection models will need to recognize.

`get_metrics_payload()` injects this anomaly scenario randomly 10% of the time when `inject_anomaly=True`.

### `simulator/runner.py` — the send loop

Simulates 3 services (`payment-service`, `user-service`, `api-gateway`) each on their own host, POSTing a fresh metrics payload for every service to the API every 5 seconds, in an infinite loop.

---

## 9. What Was Verified

- ✅ `docker compose up` starts Postgres (TimescaleDB) and Redis, both pass health checks.
- ✅ Alembic migration applied successfully; `metric_datapoints`, `alerts`, `alembic_version` tables exist in Postgres.
- ✅ `GET /health` returns `200 OK`.
- ✅ `POST /api/v1/metrics` accepts a payload and saves 7 rows (one per metric) per request.
- ✅ `GET /api/v1/metrics` returns historical data, with filters working correctly.
- ✅ Running `simulator/runner.py` sends realistic metrics for 3 services every 5 seconds and they land correctly in PostgreSQL end to end.
- ✅ Interactive docs available at `/docs`.

---

## 10. Key Design Decisions (Why, Not Just What)

- **Fail-fast config** — required settings have no defaults so misconfiguration crashes the app at startup, not mid-request.
- **Narrow/long metric table** — optimized for time-series querying and painless addition of new metric types later.
- **UUID keys for alerts** — prevents ID enumeration; not needed for metrics since those are never looked up by ID directly.
- **Input/output schema separation** — decouples the public API contract from both the database schema and from validation rules that only apply one direction.
- **Dependency-injected DB sessions (`get_db`)** — guarantees connections are always released, even on error, and keeps sessions request-scoped rather than shared/global.
- **`202 Accepted` on ingest** — anticipates the move to asynchronous processing in Phase 2 without needing an API contract change later.
- **Version-controlled schema (Alembic)** — every schema change is reproducible and reversible across environments.

---

## 11. What's Next — Phase 2 Preview

Phase 2 (Data Processing Pipeline) will:
- Wire up Celery, using the already-provisioned Redis as the task queue.
- Move `POST /api/v1/metrics` from a synchronous DB write to pushing a task onto the queue (making the `202 Accepted` status fully accurate).
- Add feature engineering: rolling averages, z-scores, and rate-of-change calculations on incoming metrics.
- Add data cleaning for missing values, outliers, and invalid readings.
- Add time-window aggregations (5-minute and 1-hour buckets).

This turns the current fully-synchronous API into an asynchronous, horizontally-scalable ingestion pipeline capable of handling much higher throughput without blocking on the database.
