"""Magnetometer: WMM2025 field (pygeomag) + residual hard/soft iron + white noise."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from dss.core.rng import rng
from dss.core.time import s_to_ns


@lru_cache(maxsize=8)
def _geomag():
    import pygeomag

    cof = Path(pygeomag.__file__).parent / "wmm" / "WMM_2025.COF"
    return pygeomag.GeoMag(coefficients_file=str(cof))


def wmm_field(lat_deg: float, lon_deg: float, alt_km: float, decimal_year: float):
    """Return the WMM result object (x north, y east, z down, nT; d declination, i inclination, deg)."""
    return _geomag().calculate(glat=lat_deg, glon=lon_deg, alt=alt_km, time=decimal_year)


def field_enu(cfg: dict) -> np.ndarray:
    r = wmm_field(cfg["lat_deg"], cfg["lon_deg"], cfg["alt_km"], cfg["decimal_year"])
    return np.array([r.y, r.x, -r.z])  # nT, ENU


def declination_rad(cfg: dict) -> float:
    return float(np.deg2rad(wmm_field(cfg["lat_deg"], cfg["lon_deg"], cfg["alt_km"], cfg["decimal_year"]).d))


def simulate_mag(t: np.ndarray, r_wb: np.ndarray, cfg: dict, seed: int, anomaly_nt: np.ndarray | None = None) -> dict:
    g = rng(seed, "mag", cfg.get("name", "mag0"))
    b_w = field_enu(cfg)
    if anomaly_nt is not None:
        b_w = b_w + anomaly_nt  # (N,3) or (3,)
    b_w = np.broadcast_to(b_w, (len(t), 3))
    b_b = np.einsum("nji,nj->ni", r_wb, b_w)
    si = np.asarray(cfg["residual_soft_iron"], dtype=float)
    hi = np.asarray(cfg["residual_hard_iron_nt"], dtype=float)
    m = b_b @ si.T + hi + g.normal(0, cfg["noise_sigma_nt"], (len(t), 3))
    return {"t_ns": s_to_ns(t), "field_nt": m}
