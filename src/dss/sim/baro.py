"""Barometer: ISA troposphere + Gauss-Markov bias + first-order lag + white noise."""

from __future__ import annotations

import numpy as np

from dss.core.rng import rng
from dss.core.time import s_to_ns

G0 = 9.80665
R_AIR = 287.05287
LAPSE = 0.0065  # K/m


def isa_pressure(h_m, p0=101325.0, t0=288.15):
    return p0 * (1.0 - LAPSE * np.asarray(h_m) / t0) ** (G0 / (R_AIR * LAPSE))


def isa_altitude(p_pa, p0=101325.0, t0=288.15):
    return t0 / LAPSE * (1.0 - (np.asarray(p_pa) / p0) ** (R_AIR * LAPSE / G0))


def gauss_markov(g: np.random.Generator, n: int, dt: float, sigma: float, tau: float, x0=None) -> np.ndarray:
    """Stationary first-order Gauss-Markov process sampled at dt (exact discretization)."""
    phi = np.exp(-dt / tau)
    q = sigma * np.sqrt(1 - phi**2)
    x = np.empty(n)
    x[0] = g.normal(0, sigma) if x0 is None else x0
    w = g.standard_normal(n)
    for k in range(1, n):
        x[k] = phi * x[k - 1] + q * w[k]
    return x


def simulate_baro(t: np.ndarray, h_true: np.ndarray, cfg: dict, seed: int, elevation_offset_m: float = 0.0) -> dict:
    """h_true is the altitude above the ISA datum (m)."""
    g = rng(seed, "baro", cfg.get("name", "baro0"))
    n = len(t)
    dt = float(np.median(np.diff(t))) if n > 1 else 1.0
    bias = gauss_markov(g, n, dt, cfg["bias_sigma_m"], cfg["bias_tau_s"])
    h = np.asarray(h_true) + elevation_offset_m
    lag = cfg.get("lag_tau_s", 0.0)
    if lag > 0:
        a = np.exp(-dt / lag)
        hl = np.empty(n)
        hl[0] = h[0]
        for k in range(1, n):
            hl[k] = a * hl[k - 1] + (1 - a) * h[k]
        h = hl
    p = isa_pressure(h + bias, cfg["p0_pa"], cfg["t0_k"]) + g.normal(0, cfg["noise_sigma_pa"], n)
    alt = isa_altitude(p, cfg["p0_pa"], cfg["t0_k"])
    return {"t_ns": s_to_ns(t), "pressure_pa": p, "alt_m": alt - elevation_offset_m, "bias_m": bias}
