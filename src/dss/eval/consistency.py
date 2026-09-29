"""Covariance consistency: NEES/ANEES with chi-square bands (docs/metrics.md)."""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2


def nees(err: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """err (T,d), cov (T,d,d) -> (T,)"""
    return np.einsum("ti,tij,tj->t", err, np.linalg.inv(cov), err)


def anees_band(dof: int, n_runs: int, conf: float = 0.95) -> tuple[float, float]:
    a = (1 - conf) / 2
    return chi2.ppf(a, dof * n_runs) / n_runs, chi2.ppf(1 - a, dof * n_runs) / n_runs


def anees_upper_one_sided(dof: int, n_runs: int, conf: float = 0.95) -> float:
    return chi2.ppf(conf, dof * n_runs) / n_runs


def anees_test(nees_runs: np.ndarray, dof: int, conf: float = 0.95, frac_required: float = 0.9,
               one_sided: bool = False) -> dict:
    """nees_runs (N, T). Two-sided: ANEES(t) inside band in >= 90% of epochs and time-average inside.

    One-sided (raw VIO): <= upper bound in >= 90% of epochs and time-average >= 1.0.
    """
    n = nees_runs.shape[0]
    an = nees_runs.mean(0)
    avg = float(an.mean())
    if one_sided:
        hi = anees_upper_one_sided(dof, n, conf)
        frac = float(np.mean(an <= hi))
        ok = frac >= frac_required and avg >= 1.0 and avg <= hi
        return {"anees_t": an, "mean": avg, "hi": hi, "lo": None, "frac_inside": frac, "pass": bool(ok)}
    lo, hi = anees_band(dof, n, conf)
    frac = float(np.mean((an >= lo) & (an <= hi)))
    ok = frac >= frac_required and lo <= avg <= hi
    return {"anees_t": an, "mean": avg, "lo": lo, "hi": hi, "frac_inside": frac, "pass": bool(ok)}


def inside_ellipse(err: np.ndarray, cov: np.ndarray, p: float = 0.99) -> np.ndarray:
    d = err.shape[1]
    return nees(err, cov) <= chi2.ppf(p, d)
