"""
Pure math helpers used to build metric values.

These functions know nothing about servers or anomalies — they only describe
SHAPES over time. Keeping them separate means scenarios.py stays readable:
it describes "what happens", these describe "how a number moves".

progress (p) is always a float from 0.0 (episode just started)
to 1.0 (episode about to end).
"""

import math
import random


# ---------------------------------------------------------------------------
# Basic building blocks
# ---------------------------------------------------------------------------

def clamp(value: float, low: float, high: float) -> float:
    """Force a value to stay inside [low, high]."""
    return max(low, min(high, value))


def lerp(start: float, end: float, t: float) -> float:
    """
    Linear interpolation: blend between start and end.
    t=0.0 -> start,  t=0.5 -> halfway,  t=1.0 -> end
    """
    return start + (end - start) * t


def add_noise(value: float, noise_level: float = 0.05) -> float:
    """
    Add small random fluctuation so data looks organic, not perfectly flat.
    noise_level=0.05 means roughly +/-5% Gaussian variation.
    """
    return value + random.gauss(0, noise_level * abs(value))


def jitter(value: float, spread: float) -> float:
    """
    Add ABSOLUTE random wobble (not a percentage).
    Used for high-variance anomalies like noisy_neighbor.
    """
    return value + random.gauss(0, spread)


# ---------------------------------------------------------------------------
# Time shapes — how a metric evolves across an episode
# ---------------------------------------------------------------------------

def ramp(p: float, start: float, end: float) -> float:
    """
    Straight line from start to end.

        end   |            /
              |         /
        start |______/
              0            1   (progress)

    Used for: memory leak, disk filling — steady linear growth.
    """
    return lerp(start, end, clamp(p, 0.0, 1.0))


def ease_in(p: float, start: float, end: float) -> float:
    """
    Slow at first, then accelerates (quadratic).

        end   |          /
              |        /
        start |______/
              0          1

    Used for: latency creep — degradation that gets worse faster over time.
    """
    p = clamp(p, 0.0, 1.0)
    return lerp(start, end, p * p)


def plateau(p: float, start: float, end: float, knee: float = 0.35) -> float:
    """
    Rise quickly, then hold flat at the top.

        end   |    ______________
              |   /
        start |__/
              0   knee          1

    Used for: sustained saturation — it maxes out and stays there.
    """
    p = clamp(p, 0.0, 1.0)
    if p >= knee:
        return end
    return lerp(start, end, p / knee)


def decay(p: float, start: float, end: float) -> float:
    """
    Start high, settle back down (exponential-ish).

        start |\\
              | \\___
        end   |     \\_________
              0                1

    Used for: cold start — a restarted service is slow at first, then warms up.
    """
    p = clamp(p, 0.0, 1.0)
    return lerp(start, end, 1.0 - (1.0 - p) ** 2)


def cliff(p: float, start: float, end: float, edge: float = 0.6) -> float:
    """
    Stay near start, then fall off a cliff near the end.

        end   |                /
              |               /
        start |______________/
              0            edge  1

    Used for: memory exhaustion — fine, fine, fine, then everything breaks.
    """
    p = clamp(p, 0.0, 1.0)
    if p <= edge:
        return start
    return lerp(start, end, (p - edge) / (1.0 - edge))


def pulse(p: float, low: float, high: float, cycles: float = 3.0) -> float:
    """
    Oscillate between low and high several times during the episode.

    Used for: flapping / intermittent failures that come and go.
    """
    wave = 0.5 + 0.5 * math.sin(2 * math.pi * cycles * clamp(p, 0.0, 1.0))
    return lerp(low, high, wave)
