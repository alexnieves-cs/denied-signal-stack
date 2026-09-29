"""Time synchronization: offset/jitter injection, offset estimation, interpolation."""

from __future__ import annotations

import numpy as np

from dss.core.rotations import quat_normalize
from dss.core.time import NS_PER_S


def inject(t_ns: np.ndarray, offset_s: float = 0.0, jitter_s: float = 0.0, g: np.random.Generator | None = None) -> np.ndarray:
    t = t_ns.astype(np.int64) + int(round(offset_s * NS_PER_S))
    if jitter_s > 0:
        if g is None:
            raise ValueError("jitter needs a generator")
        t = t + np.rint(g.normal(0, jitter_s * NS_PER_S, len(t))).astype(np.int64)
    return t


def estimate_offset(t_ref_ns: np.ndarray, x_ref: np.ndarray, t_ns: np.ndarray, x: np.ndarray,
                    max_offset_s: float = 0.1, grid_hz: float = 2000.0) -> float:
    """Estimate d such that x(t) ~= x_ref(t - d) (i.e. the second clock reads late by d).

    Signals are resampled on a common fine grid; the cross-correlation peak is refined with a
    parabola. Returns seconds.
    """
    tr = t_ref_ns / NS_PER_S
    tx = t_ns / NS_PER_S
    t0 = max(tr[0], tx[0]) + max_offset_s
    t1 = min(tr[-1], tx[-1]) - max_offset_s
    g = np.arange(t0, t1, 1.0 / grid_hz)
    a = np.interp(g, tr, x_ref)
    a = a - a.mean()
    lags = np.arange(-int(max_offset_s * grid_hz), int(max_offset_s * grid_hz) + 1)
    # evaluate x on shifted grids: corr(k) = sum a(g) * x(g + k/grid)
    corr = np.empty(len(lags))
    for i, k in enumerate(lags):
        b = np.interp(g + k / grid_hz, tx, x)
        corr[i] = np.dot(a, b - b.mean())
    i = int(np.argmax(corr))
    # continuous refinement: least squares over the offset around the coarse peak
    from scipy.optimize import minimize_scalar

    def ssd(d):
        b = np.interp(g + d, tx, x)
        return float(np.sum((a - (b - b.mean())) ** 2))

    c = lags[i] / grid_hz
    r = minimize_scalar(ssd, bounds=(c - 2.0 / grid_hz, c + 2.0 / grid_hz), method="bounded",
                        options={"xatol": 1e-7})
    return float(r.x)


def interp_linear(t_query_ns: np.ndarray, t_ns: np.ndarray, x: np.ndarray) -> np.ndarray:
    tq = np.asarray(t_query_ns, dtype=np.float64)
    ts = np.asarray(t_ns, dtype=np.float64)
    x = np.asarray(x)
    if x.ndim == 1:
        return np.interp(tq, ts, x)
    return np.stack([np.interp(tq, ts, x[:, j]) for j in range(x.shape[1])], axis=1)


def slerp(q0: np.ndarray, q1: np.ndarray, u: np.ndarray) -> np.ndarray:
    q0 = np.atleast_2d(q0)
    q1 = np.atleast_2d(q1).copy()
    u = np.atleast_1d(u)[:, None]
    d = np.sum(q0 * q1, axis=1, keepdims=True)
    q1[d[:, 0] < 0] *= -1
    d = np.abs(d)
    th = np.arccos(np.clip(d, -1, 1))
    small = th[:, 0] < 1e-8
    s = np.sin(th)
    s[small] = 1.0
    w0 = np.where(small[:, None], 1 - u, np.sin((1 - u) * th) / s)
    w1 = np.where(small[:, None], u, np.sin(u * th) / s)
    return quat_normalize(w0 * q0 + w1 * q1)


def interp_pose(t_query_ns, t_ns, p, q):
    tq = np.asarray(t_query_ns, dtype=np.int64)
    i = np.clip(np.searchsorted(t_ns, tq) - 1, 0, len(t_ns) - 2)
    u = (tq - t_ns[i]) / (t_ns[i + 1] - t_ns[i])
    u = np.clip(u, 0, 1)
    pp = p[i] + u[:, None] * (p[i + 1] - p[i])
    qq = slerp(q[i], q[i + 1], u)
    return pp, qq
