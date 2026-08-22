# 🔍 Intelligent Anomaly Detection & Incident Prediction Platform

**Catch server problems before they become outages — not after.**

A production-style backend platform that ingests live server metrics, cleans and enriches them in real time, and runs a layered detection pipeline (statistical + machine learning) to catch anomalies that simple threshold alerts miss.

```
Metrics come in → get cleaned → get enriched → get scored for anomalies → you get alerted
                                                        ↑
                                          before things break, not after
```

---

## 🚀 Why This Exists

Most monitoring systems only ask one question: *"Is this one number too high right now?"*

That catches obvious spikes. It misses everything else:

- A memory leak that creeps up 2% every hour for ten hours
- Unusual server load at 3am when nothing should be running
- A slow-motion cascading failure across CPU, latency, and error rate together

This platform is built to catch all three — not just the loud, obvious kind.

---

## 🧠 What It Actually Does

```
┌─────────────┐     ┌───────────┐     ┌──────────────┐     ┌────────────────┐
│  Simulator / │ ──▶ │  FastAPI   │ ──▶ │    Redis      │ ──▶ │  Celery Worker  │
│  Real servers│     │  ingest    │     │  task queue   │     │  (background)   │
└─────────────┘     └───────────┘     └──────────────┘     └────────────────┘
                                                                     │
                                                                     ▼
                                                        ┌────────────────────────┐
                                                        │  Clean → Engineer       │
                                                        │  features → Score       │
                                                        │  for anomalies          │
                                                        └────────────────────────┘
                                                                     │
                                                                     ▼
                                                        ┌────────────────────────┐
                                                        │  PostgreSQL +           │
                                                        │  TimescaleDB            │
                                                        └────────────────────────┘
```

1. **Ingest** — accepts live metrics (CPU, memory, latency, error rate, and more) through a REST API
2. **Clean** — fixes missing values and clips impossible readings automatically
3. **Engineer features** — computes rolling averages, z-scores, and rate-of-change per metric, per service
4. **Detect** — scores every reading through multiple anomaly detection methods running together
5. **Aggregate** — rolls raw data into 5-minute summaries automatically, on a schedule, with zero manual triggers
6. **Alert** — surfaces what matters, with an explanation of *why* it was flagged

---

## 🏗️ Tech Stack

| Layer | Technology | Why it's here |
|---|---|---|
| API | FastAPI + Pydantic v2 | Async-ready, auto-validates every request, self-documenting |
| Background processing | Celery + Redis | Non-blocking ingestion — the API never waits on heavy work |
| Database | PostgreSQL + TimescaleDB | Built for exactly this: high-volume time-series data |
| ORM & migrations | SQLAlchemy + Alembic | Version-controlled, reversible schema changes |
| Containers | Docker Compose | One command spins up the entire stack, identically, anywhere |
| ML (in progress) | Isolation Forest, LSTM Autoencoder, XGBoost + SHAP | Layered detection — not one model doing all the work |

---

## 📊 Anomaly Types It's Designed to Catch

```
┌──────────────────────┬────────────────────────────┬──────────────────┐
│ Type                 │ What it looks like          │ Why it's tricky   │
├──────────────────────┼────────────────────────────┼──────────────────┤
│ Point anomaly        │ One sudden spike            │ Easy — obvious    │
│ Contextual anomaly    │ Normal value, wrong time    │ Needs context     │
│ Collective anomaly    │ Slow drift, no single       │ Invisible to      │
│                       │ reading looks wrong          │ simple thresholds │
└──────────────────────┴────────────────────────────┴──────────────────┘
```

---

## 📁 Project Structure

```
anomaly-detection-platform/
├── src/
│   ├── api/routes/        → REST endpoints (health, metrics)
│   ├── db/                → SQLAlchemy engine, session, migrations
│   ├── models/             → Database table definitions
│   ├── schemas/            → Request/response validation
│   ├── processing/         → Data cleaning + feature engineering
│   └── workers/            → Celery app, background tasks, scheduling
├── simulator/               → Realistic fake metric generator for testing
├── docs/                    → Full phase-by-phase build documentation
├── docker-compose.yml
└── requirements.txt
```

---

## ⚡ Quick Start

```bash
# 1. Clone and enter the project
git clone <your-repo-url>
cd anomaly-detection-platform

# 2. Start Postgres + Redis
docker compose up -d

# 3. Install dependencies
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 4. Apply database migrations
alembic upgrade head

# 5. Start the API
uvicorn src.main:app --reload --port 8080

# 6. Start the Celery worker (separate terminal)
celery -A src.workers.celery_app.celery_app worker --loglevel=info

# 7. Start the scheduler (separate terminal)
celery -A src.workers.celery_app.celery_app beat --loglevel=info

# 8. Feed it live data (separate terminal)
python simulator/runner.py
```

Then visit **http://127.0.0.1:8080/docs** for interactive API docs.

---

## 🗺️ Build Roadmap

```
✅  Phase 1 — Foundation (API, database, simulator)
✅  Phase 2 — Async Data Pipeline (Celery, Redis, feature engineering)
✅  Phase 3 — ML Anomaly Detection Models
✅  Phase 4 — Incident Prediction (XGBoost + SHAP explainability)
⬜  Phase 5 — Alert & Incident Management
⬜  Phase 6 — Real-Time Dashboard (React)
⬜  Phase 7 — Monitoring & Observability (Prometheus + Grafana)
⬜  Phase 8 — Production Hardening (auth, tests, CI/CD)
```

Full write-up of every phase — including design decisions and why they were made — is in [`/docs`](./docs).

---

## 💡 Design Decisions Worth Knowing

- **Asynchronous by default** — the API never blocks on database writes; every ingest request is handed to a background worker and returns instantly
- **Narrow/long table schema** — new metric types can be added with zero schema changes
- **Layered detection, not one model** — a fast statistical check runs alongside heavier ML models, so nothing waits on the slowest method
- **Self-scheduling aggregation** — summary data is generated automatically on a timer, with no manual trigger required

---

## 📬 Status

Actively being built, phase by phase, with full documentation at every step. Follow the `/docs` folder for the complete build log.

---

<p align="center">Built to catch what threshold alerts miss.</p>
