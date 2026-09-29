"""Metrics frozen in docs/metrics.md: ATE, RPE, drift %, TTR, relocalization, spoof timing."""

from __future__ import annotations

import numpy as np

from dss.core.rotations import so3_log
from dss.eval import align as al


def path_length(p: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)))


def ate(p_est: np.ndarray, p_gt: np.ndarray, method: str = "posyaw") -> dict:
    R, t, s = al.align(p_est, p_gt, method)
    pa = al.apply(R, t, s, p_est)
    e = np.linalg.norm(pa - p_gt, axis=1)
    L = path_length(p_gt)
    rmse = float(np.sqrt(np.mean(e**2)))
    return {
        "rmse_m": rmse,
        "mean_m": float(e.mean()),
        "median_m": float(np.median(e)),
        "max_m": float(e.max()),
        "path_m": L,
        "drift_pct": 100.0 * rmse / L if L > 0 else float("nan"),
        "errors": e,
        "aligned": pa,
    }


def ate_se3_rot(p_est, R_est, p_gt, R_gt, method="se3"):
    """Translation + rotation ATE; rotation error in degrees."""
    R, t, s = al.align(p_est, p_gt, method)
    pa = al.apply(R, t, s, p_est)
    te = np.linalg.norm(pa - p_gt, axis=1)
    re = np.array([np.linalg.norm(so3_log((R @ Re).T @ Rg)) for Re, Rg in zip(R_est, R_gt)])
    return {"rmse_m": float(np.sqrt(np.mean(te**2))), "rot_rmse_deg": float(np.rad2deg(np.sqrt(np.mean(re**2))))}


def rpe(p_est: np.ndarray, p_gt: np.ndarray, lengths=(8, 16, 24, 32, 40)) -> dict:
    """Relative translation error over sub-trajectories of given path lengths (rpg style).

    Segments longer than the path are reported as empty (count 0), never an error.
    """
    d = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p_gt, axis=0), axis=1))])
    out = {}
    for L in lengths:
        errs = []
        step = max(1, len(p_gt) // 200)
        for i in range(0, len(p_gt), step):
            j = np.searchsorted(d, d[i] + L)
            if j >= len(p_gt):
                break
            dg = p_gt[j] - p_gt[i]
            de = p_est[j] - p_est[i]
            # align the segment start orientation via posyaw of the pair (translation-only RPE)
            errs.append(np.linalg.norm(de - dg))
        errs = np.asarray(errs)
        out[L] = {
            "count": int(len(errs)),
            "mean_m": float(errs.mean()) if len(errs) else float("nan"),
            "median_m": float(np.median(errs)) if len(errs) else float("nan"),
            "pct": float(100 * errs.mean() / L) if len(errs) else float("nan"),
        }
    return out


def endpoint_drift(p_est, p_gt, n_first: int, n_last: int) -> dict:
    """Align on the first n_first samples, measure on the last n_last (TUM-VI corridor style)."""
    R, t, s = al.align_se3(p_est[:n_first], p_gt[:n_first])
    pa = al.apply(R, t, s, p_est)
    e = np.linalg.norm(pa[-n_last:] - p_gt[-n_last:], axis=1).mean()
    L = path_length(p_gt)
    return {"endpoint_err_m": float(e), "path_m": L, "drift_pct": float(100 * e / L)}


def ttr(t_s: np.ndarray, err_m: np.ndarray, inside99: np.ndarray, t_loss: float, tol_m: float = 5.0,
        window_s: float = 10.0) -> dict:
    """Time to recover after GPS loss (docs/metrics.md). Inputs sampled at 1 Hz (or finer).

    t* is the first time from which every sample in [t*, t*+window] has err<=tol and truth inside
    the 99% ellipse. TTR = t* - t_loss; 0 if it already holds at t_loss; censored if never.
    """
    ok = (err_m <= tol_m) & inside99
    idx = np.nonzero(t_s >= t_loss)[0]
    for k in idx:
        w = (t_s >= t_s[k]) & (t_s <= t_s[k] + window_s)
        if t_s[-1] < t_s[k] + window_s:
            break
        if ok[w].all():
            return {"ttr_s": float(max(0.0, t_s[k] - t_loss)), "censored": False, "t_star": float(t_s[k])}
    return {"ttr_s": float("inf"), "censored": True, "t_star": None}


def relocalization(errors_m: np.ndarray, accepted: np.ndarray, success_m: float = 5.0, false_m: float = 20.0) -> dict:
    n = len(errors_m)
    acc = np.asarray(accepted, bool)
    e = np.asarray(errors_m)
    succ = acc & (e <= success_m)
    fa = acc & (e > false_m)
    return {
        "attempts": int(n),
        "accepted": int(acc.sum()),
        "success_rate": float(succ.sum() / n) if n else float("nan"),
        "precision": float(succ.sum() / acc.sum()) if acc.sum() else float("nan"),
        "false_accept_rate": float(fa.sum() / acc.sum()) if acc.sum() else 0.0,
        "median_m": float(np.median(e[acc])) if acc.any() else float("nan"),
        "p95_m": float(np.percentile(e[acc], 95)) if acc.any() else float("nan"),
    }


def spoof_timing(t_s, offset_true_m, alarm, sigma_innov_m, mdb_factor: float = 5.94) -> dict:
    """t_onset: first epoch the spoofed position departs from truth; t_detectable: offset > MDB."""
    t_s = np.asarray(t_s)
    off = np.asarray(offset_true_m)
    on = np.nonzero(off > 1e-6)[0]
    if len(on) == 0:
        return {"t_onset": None}
    t_onset = float(t_s[on[0]])
    det = np.nonzero(off > mdb_factor * sigma_innov_m)[0]
    t_det = float(t_s[det[0]]) if len(det) else None
    al_idx = np.nonzero(np.asarray(alarm) & (t_s >= t_onset))[0]
    t_flag = float(t_s[al_idx[0]]) if len(al_idx) else None
    return {
        "t_onset": t_onset,
        "t_detectable": t_det,
        "t_flag": t_flag,
        "latency_from_onset_s": (t_flag - t_onset) if t_flag is not None else None,
        "latency_from_detectable_s": (t_flag - t_det) if (t_flag is not None and t_det is not None) else None,
    }
