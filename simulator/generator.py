"""
METRIC GENERATOR
================

Produces one payload per call, per service, in exactly the shape
`MetricIngest` (src/schemas/metric.py) expects.

The big change from the Phase 1 version: this generator is now STATEFUL.

    Phase 1 version:  every call was independent — an anomaly was one bad row.
    This version:     each service has its own state, and anomalies are
                      EPISODES that unfold across many consecutive readings.

Why that matters for Phase 3:

    * features.py keeps a rolling window of 10 readings. A one-tick spike
      barely moves the rolling average; a 60-tick memory leak moves it a lot.
      Only episodes let you see the difference.
    * The LSTM Autoencoder learns SEQUENCES. If every anomaly is a single
      isolated row, there is no sequence to learn and the LSTM is pointless.
    * Real incidents last minutes, not one reading. Training on realistic
      duration is what makes the model work on real data.

Each service runs a small state machine:

    NORMAL ──(random chance)──> EPISODE ──(episode ends)──> COOLDOWN ──> NORMAL
      ▲                                                                    │
      └────────────────────────────────────────────────────────────────────┘
"""

import math
import random
import time as _time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

from simulator.scenarios import (
    SCENARIOS,
    SCENARIOS_BY_NAME,
    Scenario,
    severity_for,
)
from simulator.shapes import add_noise, clamp

# ---------------------------------------------------------------------------
# Tuning knobs
# ---------------------------------------------------------------------------

# Chance per tick of STARTING a new episode while a service is healthy.
# 0.012 yields roughly 20% anomalous readings overall — deliberately high,
# because ML training needs plenty of positive examples.
# Real production is nearer 0.1%. Lower this to ~0.002 (--anomaly-rate 0.002)
# when you want a realistic evaluation set instead of a training set.
ANOMALY_START_PROBABILITY = 0.012

# Forced healthy period after an episode ends, so the rolling window in
# features.py can recover before the next anomaly begins.
COOLDOWN_TICKS = (18, 45)

# Intensity range. Low = mild/subtle, high = severe/obvious.
# Mild anomalies are included on purpose: a detector that only catches
# severe ones is not much of a detector.
INTENSITY_RANGE = (0.45, 1.0)

WINDOW_WARMUP_TICKS = 12  # normal-only readings before any anomaly may fire

# Ceiling on how far a service's baseline may permanently drift from repeated
# deployment_regression episodes. Without this, shifts compound forever and the
# service ends up pinned at the clamp limits — unrealistic, and it would teach
# the models that "broken" is normal.
MAX_BASELINE_SHIFT = 3.0


# ---------------------------------------------------------------------------
# Per-service state
# ---------------------------------------------------------------------------

@dataclass
class ServiceState:
    """Everything the generator must remember about one service between ticks."""
    service_name: str
    host: str

    # slowly-drifting disk usage, persists across ticks like a real disk
    disk_base: float = field(default_factory=lambda: random.uniform(42.0, 49.0))

    # active episode
    scenario: Optional[Scenario] = None
    episode_id: Optional[str] = None
    ticks_elapsed: int = 0
    ticks_total: int = 0
    intensity: float = 1.0

    cooldown: int = 0
    warmup: int = WINDOW_WARMUP_TICKS

    frozen: Optional[dict] = None          # used by service_flatline
    baseline_shift: dict = field(default_factory=dict)   # permanent multipliers
    pending_shift: Optional[dict] = None   # staged by deployment_regression

    total_ticks: int = 0
    anomalous_ticks: int = 0
    episode_counts: dict = field(default_factory=dict)


_STATES: dict[str, ServiceState] = {}


def get_state(service_name: str, host: str) -> ServiceState:
    """Fetch (or create) the persistent state for one service."""
    if service_name not in _STATES:
        _STATES[service_name] = ServiceState(service_name=service_name, host=host)
    return _STATES[service_name]


def reset_states() -> None:
    """Wipe all state. Useful in tests."""
    _STATES.clear()


def all_states() -> dict[str, ServiceState]:
    return _STATES


# ---------------------------------------------------------------------------
# The healthy baseline
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# The simulated clock
# ---------------------------------------------------------------------------
# Server behaviour follows a 24-hour rhythm, so training data needs to cover a
# full day and night. Waiting 24 real hours to collect it is wasteful — this is
# a simulator, not a real server.
#
# A simulated clock lets time run faster than real time:
#
#     scale = 1    -> real time (1 real second = 1 simulated second)
#     scale = 80   -> 18 real minutes = 24 simulated hours
#
# CRITICAL: the payload timestamp uses this same clock. If the timestamp said
# 14:05 while the metrics reflected 3am behaviour, the data would contradict
# itself and no model could learn the daily pattern from it.

_CLOCK = {
    "scale":       1.0,
    "real_origin": None,   # monotonic seconds when the clock was configured
    "sim_origin":  None,   # simulated datetime at that moment
}


def configure_clock(scale: float = 1.0, start_hour: Optional[float] = None) -> None:
    """
    Set up the simulated clock.

    scale      : how many simulated seconds pass per real second (>= 1.0)
    start_hour : UTC hour to begin the simulated day at (0-24).
                 Defaults to the current real hour.
    """
    now = datetime.now(timezone.utc)
    sim_origin = now
    if start_hour is not None:
        hour = int(start_hour) % 24
        minute = int((start_hour - int(start_hour)) * 60)
        sim_origin = now.replace(hour=hour, minute=minute, second=0, microsecond=0)

    _CLOCK["scale"]       = max(1.0, float(scale))
    _CLOCK["real_origin"] = _time.monotonic()
    _CLOCK["sim_origin"]  = sim_origin


def sim_now() -> datetime:
    """Current simulated time (UTC). Identical to real time when scale is 1."""
    if _CLOCK["real_origin"] is None or _CLOCK["scale"] == 1.0:
        return datetime.now(timezone.utc)
    elapsed_real = _time.monotonic() - _CLOCK["real_origin"]
    return _CLOCK["sim_origin"] + timedelta(seconds=elapsed_real * _CLOCK["scale"])


def get_time_of_day_factor(now: Optional[datetime] = None) -> float:
    """
    Business-hours rhythm: ~0.3 at night, ~1.0 mid-afternoon.

    Uses a smooth 24-hour cosine so the curve is continuous across midnight
    (the Phase 1 sine version jumped discontinuously, which taught the LSTM a
    fake nightly cliff).

        1.0 |        ,-''-.
            |     ,-'      '-.
        0.3 |_,-'              '-._
            0h    6h   14h   20h  24h

    Derived from the SAME clock as the payload timestamp (UTC), so hour-of-day
    can be recovered from the database later as an ML feature.
    """
    now = now or sim_now()
    hour = now.hour + now.minute / 60.0
    # peak at 14:00 UTC, trough at 02:00 UTC
    wave = 0.5 + 0.5 * math.cos(2 * math.pi * (hour - 14.0) / 24.0)
    return clamp(0.30 + 0.70 * wave, 0.30, 1.0)


def generate_normal_metrics(state: ServiceState, factor: float) -> dict:
    """
    One set of healthy, realistic metrics.

    Values follow the time-of-day factor and carry Gaussian noise, so "normal"
    is a noisy band rather than a flat line. That band is precisely what the
    models learn — and anything outside it is what they flag.
    """
    # disk creeps upward permanently, like a real filesystem
    state.disk_base = min(state.disk_base + 0.0009, 92.0)

    metrics = {
        "cpu_usage":          add_noise(35 * factor + 10),          # ~20-45%
        "memory_usage":       add_noise(50 + 10 * factor),          # ~53-60%
        "request_latency_ms": add_noise(50 + 30 * factor),          # ~59-80ms
        "requests_per_sec":   add_noise(100 * factor + 20),         # ~50-120/s
        "error_rate":         add_noise(0.002, noise_level=0.5),    # ~0.2%
        "db_connections":     add_noise(20 * factor + 5),           # ~11-25
        "disk_usage":         add_noise(state.disk_base, 0.002),
    }

    # A past deployment_regression permanently raised this service's baseline.
    for key, multiplier in state.baseline_shift.items():
        metrics[key] *= multiplier

    return metrics


def _finalize(metrics: dict) -> dict:
    """
    Clamp everything into the ranges `MetricsPayload` validates against, so the
    API never rejects a payload with 422. Also rounds for readable JSON.
    """
    metrics["cpu_usage"]          = clamp(metrics["cpu_usage"], 0.5, 100.0)
    metrics["memory_usage"]       = clamp(metrics["memory_usage"], 0.5, 100.0)
    metrics["disk_usage"]         = clamp(metrics["disk_usage"], 0.5, 100.0)
    metrics["error_rate"]         = clamp(metrics["error_rate"], 0.0, 1.0)
    metrics["request_latency_ms"] = clamp(metrics["request_latency_ms"], 0.5, 60_000.0)
    metrics["requests_per_sec"]   = clamp(metrics["requests_per_sec"], 0.0, 100_000.0)
    metrics["db_connections"]     = int(clamp(metrics["db_connections"], 0, 500))

    return {
        key: (value if key == "db_connections" else round(float(value), 4))
        for key, value in metrics.items()
    }


# ---------------------------------------------------------------------------
# Episode lifecycle
# ---------------------------------------------------------------------------

def _eligible(ctx: dict) -> list[Scenario]:
    """Scenarios allowed to start right now (off_hours_surge needs night)."""
    return [s for s in SCENARIOS if s.precondition is None or s.precondition(ctx)]


def _start_episode(state: ServiceState, ctx: dict, forced: Optional[str] = None) -> None:
    if forced:
        scenario = SCENARIOS_BY_NAME[forced]
    else:
        pool = _eligible(ctx)
        scenario = random.choices(pool, weights=[s.weight for s in pool], k=1)[0]

    state.scenario      = scenario
    state.episode_id    = uuid.uuid4().hex[:8]
    state.ticks_total   = random.randint(scenario.min_ticks, scenario.max_ticks)
    state.ticks_elapsed = 0
    state.intensity     = random.uniform(*INTENSITY_RANGE)
    state.frozen        = None
    state.pending_shift = None
    state.episode_counts[scenario.name] = state.episode_counts.get(scenario.name, 0) + 1


def _end_episode(state: ServiceState) -> None:
    scenario = state.scenario

    # deployment_regression leaves a permanent new normal behind,
    # capped so repeated regressions cannot drift a service into nonsense
    if scenario is not None and scenario.permanent and state.pending_shift:
        for key, multiplier in state.pending_shift.items():
            current = state.baseline_shift.get(key, 1.0)
            state.baseline_shift[key] = min(current * multiplier, MAX_BASELINE_SHIFT)

    # someone cleaned up the disk after a disk incident
    if scenario is not None and scenario.name in ("disk_filling", "disk_full"):
        state.disk_base = random.uniform(43.0, 49.0)

    state.scenario      = None
    state.episode_id    = None
    state.ticks_elapsed = 0
    state.ticks_total   = 0
    state.frozen        = None
    state.pending_shift = None
    state.cooldown      = random.randint(*COOLDOWN_TICKS)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_tick(
    service_name: str,
    host: str,
    inject_anomaly: bool = True,
    start_probability: float = ANOMALY_START_PROBABILITY,
    force_scenario: Optional[str] = None,
) -> tuple[dict, dict]:
    """
    Advance one service by one tick.

    Returns (payload, label):
        payload = exactly what the API expects
        label   = GROUND TRUTH about this reading

    The label is the reason this simulator is worth building carefully. Because
    you know which readings were genuinely anomalous, you can measure your
    models properly in Phase 3 — precision, recall, and per-shape accuracy
    ("we catch 98% of spikes but only 61% of contextual anomalies").
    Without labels you can only squint at charts and hope.
    """
    state = get_state(service_name, host)
    state.total_ticks += 1

    now = sim_now()
    factor = get_time_of_day_factor(now)
    ctx = {"factor": factor, "state": state}

    metrics = generate_normal_metrics(state, factor)

    # --- state machine -----------------------------------------------------
    if state.warmup > 0:
        state.warmup -= 1

    elif state.scenario is None and state.cooldown > 0:
        state.cooldown -= 1

    elif state.scenario is None and (force_scenario or inject_anomaly):
        if force_scenario or random.random() < start_probability:
            eligible = _eligible(ctx)
            if force_scenario or eligible:
                _start_episode(state, ctx, forced=force_scenario)

    # --- apply the active episode -----------------------------------------
    label = {
        "is_anomaly":   False,
        "anomaly_type": "normal",
        "category":     "normal",
        "severity":     "none",
        "intensity":    0.0,
        "progress":     0.0,
        "episode_id":   None,
    }

    if state.scenario is not None:
        scenario = state.scenario
        progress = state.ticks_elapsed / max(1, state.ticks_total)

        scenario.apply(metrics, progress, state.intensity, ctx)

        label = {
            "is_anomaly":   True,
            "anomaly_type": scenario.name,
            "category":     scenario.category,
            "severity":     severity_for(state.intensity),
            "intensity":    round(state.intensity, 3),
            "progress":     round(progress, 3),
            "episode_id":   state.episode_id,
        }
        state.anomalous_ticks += 1
        state.ticks_elapsed += 1

        if state.ticks_elapsed >= state.ticks_total:
            _end_episode(state)

    label["hour_utc"] = round(now.hour + now.minute / 60.0, 2)
    label["tod_factor"] = round(factor, 3)

    payload = {
        "service_name": service_name,
        "host":         host,
        "timestamp":    now.isoformat(),
        "metrics":      _finalize(metrics),
    }
    return payload, label


def get_metrics_payload(
    service_name: str,
    host: str,
    inject_anomaly: bool = False,
) -> dict:
    """
    Backwards-compatible wrapper matching the Phase 1 signature, so anything
    that already imports this keeps working. Returns just the payload.
    """
    payload, _ = generate_tick(service_name, host, inject_anomaly=inject_anomaly)
    return payload
