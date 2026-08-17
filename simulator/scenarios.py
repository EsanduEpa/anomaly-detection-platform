"""
THE ANOMALY CATALOG
===================

Every anomaly a monitoring platform must catch falls into one of a small number
of STATISTICAL SHAPES. Different shapes need different detectors — that is the
whole reason Phase 3 uses an ensemble of three models instead of one.

    category           what it looks like               which model catches it best
    ---------------------------------------------------------------------------------
    spike              one or a few extreme readings    z-score  (instant, obvious)
    plateau            jumps up and STAYS up            z-score + Isolation Forest
    trend              creeps up slowly over minutes    LSTM     (z-score adapts & misses)
    cliff              normal, normal, then collapse    LSTM + Isolation Forest
    dip                drops toward zero                Isolation Forest
    flatline           variance disappears entirely     LSTM     (z-score sees nothing)
    variance           mean is fine, wobble explodes    LSTM
    level_shift        permanent new baseline           LSTM (then becomes concept drift)
    correlation_break  metrics stop agreeing            Isolation Forest (multi-metric)
    contextual         value is normal, TIMING is not   LSTM only
    transient          brief burst that self-heals      z-score

Each scenario below is an EPISODE: it lasts many consecutive readings and
unfolds over time. That matters enormously — a real memory leak is not one bad
number, it is 200 readings each slightly worse than the last. Phase 2's
features.py keeps a rolling window of 10, and the LSTM in Phase 3 reads
sequences, so anomalies must have a time dimension to be learnable.

`apply(m, p, i, ctx)` mutates the metrics dict in place, where:
    m   = the metrics dict (already filled with NORMAL values)
    p   = progress through the episode, 0.0 -> 1.0
    i   = intensity, ~0.45 (mild) -> 1.0 (severe)
    ctx = {"factor": time-of-day factor, "state": ServiceState}
"""

from dataclasses import dataclass
from typing import Callable, Optional

from simulator.shapes import (
    add_noise,
    cliff,
    decay,
    ease_in,
    jitter,
    lerp,
    plateau,
    pulse,
    ramp,
)


@dataclass
class Scenario:
    """One anomaly type and how it behaves over an episode."""
    name: str
    category: str
    min_ticks: int
    max_ticks: int
    weight: float                       # relative chance of being chosen
    description: str
    apply: Callable
    permanent: bool = False             # does it leave a lasting baseline change?
    precondition: Optional[Callable] = None   # only fire when this returns True


# ===========================================================================
# 1. SPIKES — sudden extreme values, the classic "something is on fire"
# ===========================================================================

def _cpu_spike(m, p, i, ctx):
    """CPU pegs high and drags latency + errors with it (cascading failure)."""
    m["cpu_usage"]          = add_noise(lerp(74, 99, i), 0.03)
    m["request_latency_ms"] = add_noise(lerp(300, 1200, i), 0.15)
    m["error_rate"]         = add_noise(lerp(0.03, 0.28, i), 0.25)
    m["db_connections"]     = add_noise(lerp(30, 70, i), 0.12)


def _error_burst(m, p, i, ctx):
    """
    Errors explode but EVERY OTHER METRIC LOOKS PERFECT.
    Classic bad deploy. Tests whether the model watches all 7 metrics or
    only the loud ones (CPU/latency).
    """
    m["error_rate"] = add_noise(lerp(0.12, 0.60, i), 0.2)


# ===========================================================================
# 2. PLATEAUS — jumps up and refuses to come back down
# ===========================================================================

def _cpu_exhaustion(m, p, i, ctx):
    """CPU saturates and stays pinned. Throughput actually FALLS — the server
    is too busy to accept new work."""
    m["cpu_usage"]          = add_noise(plateau(p, m["cpu_usage"], lerp(92, 99.5, i)), 0.02)
    m["request_latency_ms"] = add_noise(plateau(p, m["request_latency_ms"], lerp(700, 2600, i)), 0.18)
    m["requests_per_sec"]   = add_noise(m["requests_per_sec"] * lerp(0.7, 0.3, i), 0.1)
    m["error_rate"]         = add_noise(lerp(0.04, 0.22, i), 0.3)


def _db_pool_exhaustion(m, p, i, ctx):
    """
    Connection pool is fully consumed; every request queues.
    COUNTER-INTUITIVE SIGNATURE: latency is terrible but CPU is LOW, because
    the server is idle-waiting on the database, not computing.
    """
    m["db_connections"]     = add_noise(plateau(p, m["db_connections"], lerp(80, 100, i)), 0.04)
    m["request_latency_ms"] = add_noise(plateau(p, m["request_latency_ms"], lerp(1400, 4200, i)), 0.2)
    m["cpu_usage"]          = add_noise(lerp(24, 11, i), 0.12)
    m["error_rate"]         = add_noise(lerp(0.05, 0.35, i), 0.25)


def _disk_full(m, p, i, ctx):
    """Disk is out of space. Writes start failing — errors rise, everything
    else looks deceptively healthy."""
    m["disk_usage"]         = add_noise(lerp(98.0, 99.9, i), 0.002)
    m["error_rate"]         = add_noise(ramp(p, 0.01, lerp(0.10, 0.38, i)), 0.2)
    m["request_latency_ms"] = add_noise(m["request_latency_ms"] * lerp(1.4, 3.0, i), 0.12)


# ===========================================================================
# 3. TRENDS — slow creeping degradation, the hardest kind to notice
# ===========================================================================

def _memory_leak(m, p, i, ctx):
    """
    Memory climbs steadily and never gets released. CPU stays completely
    normal, which is exactly why humans miss it for hours.
    A rolling z-score also misses it: the baseline creeps up WITH the leak.
    """
    m["memory_usage"]   = add_noise(ramp(p, m["memory_usage"], lerp(82, 96, i)), 0.012)
    m["db_connections"] = add_noise(m["db_connections"] * lerp(1.1, 1.5, i), 0.1)
    if p > 0.6:
        m["request_latency_ms"] = add_noise(m["request_latency_ms"] * lerp(1.2, 2.2, i), 0.12)


def _disk_filling(m, p, i, ctx):
    """Runaway log file. Disk grows for a long time before it becomes critical."""
    m["disk_usage"] = add_noise(ramp(p, ctx["state"].disk_base, lerp(88, 97.5, i)), 0.004)


def _latency_creep(m, p, i, ctx):
    """
    Gradual performance rot — a slow query getting slower as a table grows.
    Accelerates (ease_in) rather than rising linearly.
    """
    m["request_latency_ms"] = add_noise(ease_in(p, m["request_latency_ms"], lerp(420, 950, i)), 0.1)
    m["cpu_usage"]          = add_noise(ease_in(p, m["cpu_usage"], m["cpu_usage"] * lerp(1.3, 1.9, i)), 0.06)
    if p > 0.7:
        m["error_rate"] = add_noise(lerp(0.01, 0.06, i), 0.3)


# ===========================================================================
# 4. CLIFFS — looks fine right up until it doesn't
# ===========================================================================

def _memory_exhaustion(m, p, i, ctx):
    """
    Leak reaches the ceiling and the process starts being OOM-killed.
    Memory ramps for most of the episode, then errors fall off a cliff.
    """
    m["memory_usage"]       = add_noise(ramp(p, m["memory_usage"], lerp(95, 99.6, i)), 0.008)
    m["error_rate"]         = add_noise(cliff(p, 0.004, lerp(0.25, 0.75, i), edge=0.55), 0.15)
    m["request_latency_ms"] = add_noise(cliff(p, m["request_latency_ms"], lerp(1500, 5000, i), edge=0.55), 0.2)
    m["requests_per_sec"]   = add_noise(cliff(p, m["requests_per_sec"], m["requests_per_sec"] * 0.2, edge=0.55), 0.15)


# ===========================================================================
# 5. DIPS — values fall toward zero. Dangerous because "low" looks healthy.
# ===========================================================================

def _traffic_drop(m, p, i, ctx):
    """
    Upstream load balancer stopped routing to us, or the service died.
    Every metric goes QUIET. A naive threshold alert ("CPU > 90") never fires,
    yet the service is completely down.
    """
    fade = lerp(0.25, 0.02, i)
    m["requests_per_sec"]   = add_noise(m["requests_per_sec"] * fade, 0.3)
    m["cpu_usage"]          = add_noise(lerp(9, 2, i), 0.2)
    m["request_latency_ms"] = add_noise(lerp(12, 2, i), 0.3)
    m["db_connections"]     = add_noise(lerp(3, 0, i), 0.4)
    m["error_rate"]         = add_noise(0.001, 0.5)


def _network_packet_loss(m, p, i, ctx):
    """Flaky network: throughput falls, latency becomes erratic, errors rise."""
    m["request_latency_ms"] = jitter(lerp(700, 2200, i), lerp(120, 500, i))
    m["error_rate"]         = add_noise(lerp(0.07, 0.32, i), 0.3)
    m["requests_per_sec"]   = add_noise(m["requests_per_sec"] * lerp(0.6, 0.25, i), 0.2)
    m["cpu_usage"]          = add_noise(m["cpu_usage"] * 0.7, 0.15)


# ===========================================================================
# 6. FLATLINE — the variance disappears. A frozen agent looks "perfectly stable".
# ===========================================================================

def _service_flatline(m, p, i, ctx):
    """
    The monitoring agent hung and keeps reporting its last reading forever.
    Values are plausible; the giveaway is that noise vanished completely.
    z-score is blind here (std -> 0). Only a sequence model notices.
    """
    state = ctx["state"]
    if state.frozen is None:
        state.frozen = dict(m)
    m.update(state.frozen)


# ===========================================================================
# 7. VARIANCE — mean stays normal, but stability is gone
# ===========================================================================

def _noisy_neighbor(m, p, i, ctx):
    """
    Another tenant on the same host is stealing CPU in bursts.
    AVERAGE CPU looks fine, so any average-based check passes — but the
    swing is violent. Detecting this requires looking at variability.
    """
    m["cpu_usage"]          = jitter(m["cpu_usage"] + lerp(8, 20, i), lerp(14, 30, i))
    m["request_latency_ms"] = jitter(m["request_latency_ms"] * lerp(1.4, 2.6, i), lerp(60, 220, i))
    m["error_rate"]         = add_noise(lerp(0.004, 0.03, i), 0.6)


def _intermittent_failure(m, p, i, ctx):
    """
    Flapping: one bad instance behind a load balancer. Errors and latency
    oscillate in and out several times instead of staying bad. Tends to
    generate alert storms, which is why Phase 5 needs deduplication.
    """
    m["error_rate"]         = add_noise(pulse(p, 0.002, lerp(0.12, 0.45, i), cycles=4), 0.15)
    m["request_latency_ms"] = add_noise(pulse(p, m["request_latency_ms"], lerp(600, 1800, i), cycles=4), 0.12)
    m["cpu_usage"]          = add_noise(pulse(p, m["cpu_usage"], lerp(60, 85, i), cycles=4), 0.08)


# ===========================================================================
# 8. LEVEL SHIFT — a new permanent normal (this becomes Phase 7 drift)
# ===========================================================================

def _deployment_regression(m, p, i, ctx):
    """
    A release made things permanently worse. Latency and CPU STEP UP to a new
    baseline and never return.

    This is the most interesting scenario in the whole catalog: it starts as
    an anomaly, and if nobody fixes it, it becomes the model's new "normal".
    That is CONCEPT DRIFT — exactly what Phase 7's drift detection is for.
    """
    lat_mult = lerp(2.2, 4.5, i)
    cpu_mult = lerp(1.35, 1.9, i)
    m["request_latency_ms"] = add_noise(plateau(p, m["request_latency_ms"], m["request_latency_ms"] * lat_mult, knee=0.15), 0.08)
    m["cpu_usage"]          = add_noise(plateau(p, m["cpu_usage"], m["cpu_usage"] * cpu_mult, knee=0.15), 0.06)
    m["error_rate"]         = add_noise(lerp(0.008, 0.035, i), 0.3)
    # Remember the shift so it persists after the episode ends.
    ctx["state"].pending_shift = {
        "request_latency_ms": lat_mult,
        "cpu_usage": cpu_mult,
    }


# ===========================================================================
# 9. CORRELATION BREAK — metrics that normally move together stop agreeing
# ===========================================================================

def _slow_query_storm(m, p, i, ctx):
    """
    Database is grinding. Latency is awful, connections are elevated, but CPU
    on the app server DROPS because it is only waiting.
    Normally latency and CPU rise together — here they move opposite ways.
    No single-metric detector can see that. Isolation Forest can.
    """
    m["request_latency_ms"] = add_noise(lerp(1100, 3200, i), 0.18)
    m["db_connections"]     = add_noise(lerp(42, 78, i), 0.1)
    m["cpu_usage"]          = add_noise(lerp(26, 13, i), 0.12)
    m["requests_per_sec"]   = add_noise(m["requests_per_sec"] * lerp(0.85, 0.55, i), 0.12)
    m["error_rate"]         = add_noise(lerp(0.01, 0.09, i), 0.3)


def _cache_failure(m, p, i, ctx):
    """
    Redis/cache went away, so every request now hits Postgres.
    Traffic is UNCHANGED but DB connections and latency explode — the
    relationship between requests and DB load has broken.
    """
    m["db_connections"]     = add_noise(lerp(52, 92, i), 0.08)
    m["request_latency_ms"] = add_noise(lerp(340, 900, i), 0.15)
    m["cpu_usage"]          = add_noise(lerp(56, 82, i), 0.08)
    m["error_rate"]         = add_noise(lerp(0.008, 0.05, i), 0.3)


def _zombie_process(m, p, i, ctx):
    """
    A runaway background job burns CPU while traffic is normal or low.
    High CPU with NO traffic to justify it — the ratio is the anomaly, not
    either value on its own.
    """
    m["cpu_usage"]        = add_noise(lerp(74, 96, i), 0.04)
    m["memory_usage"]     = add_noise(m["memory_usage"] * lerp(1.1, 1.35, i), 0.03)
    m["requests_per_sec"] = add_noise(m["requests_per_sec"] * lerp(0.9, 0.5, i), 0.12)
    # latency deliberately left NORMAL — users are not even affected yet


# ===========================================================================
# 10. SURGES — real load, not a bug. The platform must still flag it.
# ===========================================================================

def _traffic_surge(m, p, i, ctx):
    """Flash crowd / marketing campaign. Legitimate traffic, but far outside
    the learned envelope."""
    m["requests_per_sec"]   = add_noise(lerp(320, 780, i), 0.12)
    m["cpu_usage"]          = add_noise(lerp(62, 88, i), 0.06)
    m["request_latency_ms"] = add_noise(lerp(160, 420, i), 0.15)
    m["db_connections"]     = add_noise(lerp(38, 72, i), 0.1)
    m["error_rate"]         = add_noise(lerp(0.004, 0.04, i), 0.3)


def _ddos_attack(m, p, i, ctx):
    """Everything maxes out at once — the loudest, easiest anomaly to detect.
    Included as the 'obvious' end of the difficulty range."""
    m["requests_per_sec"]   = add_noise(lerp(900, 2600, i), 0.15)
    m["error_rate"]         = add_noise(lerp(0.30, 0.80, i), 0.12)
    m["request_latency_ms"] = add_noise(lerp(1800, 5500, i), 0.18)
    m["cpu_usage"]          = add_noise(lerp(90, 99.8, i), 0.02)
    m["db_connections"]     = add_noise(lerp(85, 100, i), 0.05)
    m["memory_usage"]       = add_noise(lerp(72, 93, i), 0.04)


def _retry_storm(m, p, i, ctx):
    """
    A small failure triggers client retries, which amplify the load, which
    causes more failures. Self-reinforcing feedback loop.
    """
    amp = lerp(2.5, 6.0, i)
    m["requests_per_sec"]   = add_noise(m["requests_per_sec"] * amp, 0.15)
    m["error_rate"]         = add_noise(ease_in(p, 0.02, lerp(0.20, 0.55, i)), 0.2)
    m["request_latency_ms"] = add_noise(ease_in(p, m["request_latency_ms"], lerp(800, 2400, i)), 0.18)
    m["db_connections"]     = add_noise(lerp(48, 90, i), 0.1)
    m["cpu_usage"]          = add_noise(lerp(70, 94, i), 0.06)


# ===========================================================================
# 11. CONTEXTUAL — every value is individually normal. Only the TIMING is wrong.
# ===========================================================================

def _off_hours_surge(m, p, i, ctx):
    """
    Daytime traffic levels at 3am. Data exfiltration, a misfiring cron job,
    or a bot farm.

    THE HARDEST CASE IN THE CATALOG. Take any single reading in isolation and
    it looks perfectly healthy — 110 req/s and 40% CPU are great numbers.
    It is only anomalous relative to the time of day.

    z-score and Isolation Forest will MISS this almost entirely.
    A time-aware sequence model is the only thing that catches it.
    That contrast is exactly why the project uses an ensemble.
    """
    m["requests_per_sec"]   = add_noise(lerp(95, 135, i), 0.08)
    m["cpu_usage"]          = add_noise(lerp(38, 52, i), 0.08)
    m["db_connections"]     = add_noise(lerp(20, 28, i), 0.1)
    m["request_latency_ms"] = add_noise(lerp(70, 90, i), 0.1)


def _is_night(ctx) -> bool:
    """off_hours_surge only makes sense when the server SHOULD be quiet."""
    return ctx["factor"] < 0.55


# ===========================================================================
# 12. TRANSIENT — brief and self-healing. Should NOT page a human at 3am.
# ===========================================================================

def _cold_start(m, p, i, ctx):
    """
    Service just restarted: caches empty, JIT not warm, connection pool
    rebuilding. Bad for 30 seconds, then fine on its own.
    Memory starts LOW and climbs (opposite of a leak) — a useful contrast
    case so the model does not learn "rising memory = always bad".
    """
    m["request_latency_ms"] = add_noise(decay(p, lerp(900, 2600, i), 80), 0.15)
    m["cpu_usage"]          = add_noise(decay(p, lerp(72, 94, i), 30), 0.08)
    m["memory_usage"]       = add_noise(ramp(p, lerp(28, 18, i), 54), 0.03)
    m["error_rate"]         = add_noise(decay(p, lerp(0.08, 0.30, i), 0.002), 0.25)
    m["db_connections"]     = add_noise(ramp(p, 2, 18), 0.15)


# ===========================================================================
# THE REGISTRY
# ===========================================================================
# Weights control how often each scenario appears in the training data.
# Subtle, hard-to-detect anomalies get HIGHER weights on purpose — those are
# the ones the models need the most practice on. Obvious ones (ddos) get less.

SCENARIOS: list[Scenario] = [
    # -- spikes -------------------------------------------------------------
    Scenario("cpu_spike", "spike", 3, 9, 1.0,
             "CPU pegs high, dragging latency and errors up with it", _cpu_spike),
    Scenario("error_burst", "spike", 5, 15, 1.0,
             "Errors explode while every other metric looks perfect", _error_burst),

    # -- plateaus -----------------------------------------------------------
    Scenario("cpu_exhaustion", "plateau", 14, 40, 0.9,
             "CPU saturates and stays pinned; throughput falls", _cpu_exhaustion),
    Scenario("db_pool_exhaustion", "plateau", 10, 28, 1.1,
             "Connection pool maxed; latency terrible but CPU LOW", _db_pool_exhaustion),
    Scenario("disk_full", "plateau", 10, 30, 0.9,
             "Disk out of space; writes fail, other metrics look fine", _disk_full),

    # -- trends (hard) ------------------------------------------------------
    Scenario("memory_leak", "trend", 40, 95, 1.5,
             "Memory creeps up for minutes while CPU stays normal", _memory_leak),
    Scenario("disk_filling", "trend", 55, 120, 1.2,
             "Runaway logs slowly consume the disk", _disk_filling),
    Scenario("latency_creep", "trend", 30, 70, 1.4,
             "Performance rots gradually and accelerates", _latency_creep),

    # -- cliffs -------------------------------------------------------------
    Scenario("memory_exhaustion", "cliff", 20, 45, 1.0,
             "Leak hits the ceiling, then OOM kills start", _memory_exhaustion),

    # -- dips ---------------------------------------------------------------
    Scenario("traffic_drop", "dip", 10, 32, 1.3,
             "Traffic collapses to near zero; low values look deceptively healthy", _traffic_drop),
    Scenario("network_packet_loss", "dip", 10, 26, 1.1,
             "Flaky network: throughput down, latency erratic, errors up", _network_packet_loss),

    # -- flatline -----------------------------------------------------------
    Scenario("service_flatline", "flatline", 12, 32, 1.2,
             "Monitoring agent hung; values frozen, variance gone", _service_flatline),

    # -- variance -----------------------------------------------------------
    Scenario("noisy_neighbor", "variance", 20, 50, 1.3,
             "Average is fine but CPU swings violently", _noisy_neighbor),
    Scenario("intermittent_failure", "variance", 18, 45, 1.2,
             "Flapping instance: errors come and go in waves", _intermittent_failure),

    # -- level shift --------------------------------------------------------
    Scenario("deployment_regression", "level_shift", 25, 70, 0.8,
             "Bad release permanently raises latency and CPU baseline",
             _deployment_regression, permanent=True),

    # -- correlation break (multi-metric) -----------------------------------
    Scenario("slow_query_storm", "correlation_break", 8, 22, 1.3,
             "Latency up but CPU DOWN — server is waiting, not working", _slow_query_storm),
    Scenario("cache_failure", "correlation_break", 15, 38, 1.3,
             "Cache gone: same traffic, but DB load and latency explode", _cache_failure),
    Scenario("zombie_process", "correlation_break", 18, 45, 1.2,
             "High CPU with no traffic to justify it", _zombie_process),

    # -- surges -------------------------------------------------------------
    Scenario("traffic_surge", "surge", 15, 42, 1.0,
             "Flash crowd: legitimate traffic far outside the normal envelope", _traffic_surge),
    Scenario("ddos_attack", "surge", 10, 30, 0.6,
             "Everything maxes out at once — the easiest anomaly to detect", _ddos_attack),
    Scenario("retry_storm", "surge", 10, 26, 1.0,
             "Failures trigger retries which cause more failures", _retry_storm),

    # -- contextual (hardest) ----------------------------------------------
    Scenario("off_hours_surge", "contextual", 15, 34, 1.4,
             "Daytime traffic levels in the middle of the night — values alone look normal",
             _off_hours_surge, precondition=_is_night),

    # -- transient ----------------------------------------------------------
    Scenario("cold_start", "transient", 5, 14, 0.9,
             "Fresh restart: slow and error-prone briefly, then self-heals", _cold_start),
]


SCENARIOS_BY_NAME = {s.name: s for s in SCENARIOS}


def severity_for(intensity: float) -> str:
    """Turn a 0.45-1.0 intensity into a human label (used as ground truth)."""
    if intensity < 0.62:
        return "mild"
    if intensity < 0.85:
        return "moderate"
    return "severe"


def categories() -> dict[str, list[str]]:
    """Group scenario names by category — handy for reporting per-shape recall."""
    out: dict[str, list[str]] = {}
    for s in SCENARIOS:
        out.setdefault(s.category, []).append(s.name)
    return out
