"""Time is int64 nanoseconds everywhere in Python (ADR-0002)."""

from __future__ import annotations

import numpy as np

NS_PER_S = 1_000_000_000


def s_to_ns(t_s) -> np.ndarray | int:
    arr = np.rint(np.asarray(t_s, dtype=np.float64) * NS_PER_S).astype(np.int64)
    return int(arr) if arr.ndim == 0 else arr


def ns_to_s(t_ns) -> np.ndarray | float:
    arr = np.asarray(t_ns, dtype=np.int64).astype(np.float64) / NS_PER_S
    return float(arr) if arr.ndim == 0 else arr


def rate_to_period_ns(hz: float) -> int:
    return int(round(NS_PER_S / hz))


def time_grid_ns(t0_ns: int, t1_ns: int, hz: float) -> np.ndarray:
    """Inclusive-of-start, exclusive-of-end grid with an exact integer period."""
    dt = rate_to_period_ns(hz)
    return np.arange(t0_ns, t1_ns, dt, dtype=np.int64)
