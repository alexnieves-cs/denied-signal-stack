"""Downward lidar rangefinder: range along body -z to the surface (DSM, so canopy returns count)."""

from __future__ import annotations

import numpy as np

from dss.core.rng import rng
from dss.core.time import s_to_ns


def range_sigma(r, cfg: dict):
    return np.sqrt(cfg["sigma0_m"] ** 2 + (cfg["sigma_rel"] * np.asarray(r)) ** 2)


def true_range(p: np.ndarray, r_wb: np.ndarray, surface_z, max_range: float = 200.0) -> np.ndarray:
    """Ray-march along -z_body to a height field ``surface_z(x, y)``."""
    d = -r_wb[:, :, 2]  # world direction of body -z
    n = len(p)
    out = np.full(n, np.inf)
    cosz = -d[:, 2]
    ok = cosz > 0.2
    # initial guess: flat surface under the vehicle, then 3 fixed-point refinements
    h = p[:, 2] - surface_z(p[:, 0], p[:, 1])
    r = np.where(ok, h / np.maximum(cosz, 1e-6), np.inf)
    for _ in range(4):
        q = p + r[:, None] * d
        h = p[:, 2] - surface_z(q[:, 0], q[:, 1])
        r = np.where(ok, h / np.maximum(cosz, 1e-6), np.inf)
    out[ok] = r[ok]
    out[out > max_range] = np.inf
    return out


def simulate_rangefinder(t: np.ndarray, p: np.ndarray, r_wb: np.ndarray, cfg: dict, seed: int, surface_z,
                         dropout: tuple[float, float] | None = None, bias_m: float = 0.0) -> dict:
    g = rng(seed, "lidar", cfg.get("name", "lidar0"))
    r_true = true_range(p, r_wb, surface_z)
    valid = (r_true >= cfg["min_range_m"]) & (r_true <= cfg["max_range_m"])
    if dropout is not None:
        valid &= ~((t >= dropout[0]) & (t < dropout[1]))
    sig = range_sigma(np.where(np.isfinite(r_true), r_true, 0.0), cfg)
    r = r_true + bias_m + g.standard_normal(len(t)) * sig
    r = np.where(valid, r, np.nan)
    return {"t_ns": s_to_ns(t), "range_m": r, "valid": valid, "sigma_m": sig, "range_true": r_true}
