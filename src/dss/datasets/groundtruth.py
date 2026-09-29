"""Ground-truth loading for ASL-format sequences."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from dss.core.rotations import quat_normalize
from dss.sensors.samples import Stream


def load_groundtruth(mav: Path) -> Stream:
    mav = Path(mav)
    for sub in ("state_groundtruth_estimate0", "mocap0"):
        f = mav / sub / "data.csv"
        if f.exists():
            t = np.loadtxt(f, delimiter=",", comments="#", dtype=np.int64, usecols=0, ndmin=1)
            d = np.loadtxt(f, delimiter=",", comments="#", dtype=np.float64, ndmin=2)[:, 1:]
            fields = {"p": d[:, 0:3], "q": quat_normalize(d[:, 3:7])}
            if d.shape[1] >= 10:
                fields["v"] = d[:, 7:10]
            else:
                fields["v"] = velocity_from_positions(t, d[:, 0:3])
            return Stream("gt", "gt", t, fields)
    raise FileNotFoundError(f"no ground truth under {mav}")


def velocity_from_positions(t_ns: np.ndarray, p: np.ndarray, win_s: float = 0.05) -> np.ndarray:
    t = t_ns * 1e-9
    v = np.gradient(p, t, axis=0)
    # light smoothing (mocap jitter)
    k = max(1, int(win_s / np.median(np.diff(t))))
    if k > 1:
        ker = np.ones(k) / k
        v = np.stack([np.convolve(v[:, i], ker, mode="same") for i in range(3)], 1)
    return v
