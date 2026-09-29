"""Timestamp association (nearest neighbour within a tolerance)."""

from __future__ import annotations

import numpy as np


def associate(t_a_ns: np.ndarray, t_b_ns: np.ndarray, max_dt_s: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
    """Return index arrays (ia, ib) of matched pairs; each b used at most once, a sorted."""
    t_a = np.asarray(t_a_ns, dtype=np.int64)
    t_b = np.asarray(t_b_ns, dtype=np.int64)
    j = np.searchsorted(t_b, t_a)
    j0 = np.clip(j - 1, 0, len(t_b) - 1)
    j1 = np.clip(j, 0, len(t_b) - 1)
    d0 = np.abs(t_a - t_b[j0])
    d1 = np.abs(t_a - t_b[j1])
    jb = np.where(d1 < d0, j1, j0)
    d = np.minimum(d0, d1)
    ok = d <= int(max_dt_s * 1e9)
    ia = np.nonzero(ok)[0]
    ib = jb[ok]
    # enforce uniqueness of b (keep the closest)
    _, first = np.unique(ib, return_index=True)
    keep = np.sort(first)
    return ia[keep], ib[keep]
