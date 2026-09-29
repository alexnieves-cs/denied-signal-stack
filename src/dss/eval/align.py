"""Trajectory alignment (ported in spirit from MIT rpg_trajectory_evaluation).

``align_se3`` is Umeyama without scale; ``align_posyaw`` estimates translation + yaw only
(the 4-DoF unobservable subspace of VIO).
"""

from __future__ import annotations

import numpy as np

from dss.core.rotations import rot_z


def align_se3(p_est: np.ndarray, p_gt: np.ndarray, with_scale: bool = False):
    """Return (R, t, s) minimizing |p_gt - (s R p_est + t)|^2."""
    mu_e = p_est.mean(0)
    mu_g = p_gt.mean(0)
    e = p_est - mu_e
    g = p_gt - mu_g
    C = g.T @ e / len(p_est)
    U, D, Vt = np.linalg.svd(C)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = float(np.trace(np.diag(D) @ S) / (np.sum(e**2) / len(p_est))) if with_scale else 1.0
    t = mu_g - s * R @ mu_e
    return R, t, s


def align_posyaw(p_est: np.ndarray, p_gt: np.ndarray):
    """Yaw + translation closed form (rpg: alignTrajectory 'posyaw')."""
    mu_e = p_est.mean(0)
    mu_g = p_gt.mean(0)
    e = p_est - mu_e
    g = p_gt - mu_g
    C = g.T @ e  # sum g e^T
    # maximize trace(Rz^T C) over yaw: A = C00+C11, B = C10-C01
    A = C[0, 0] + C[1, 1]
    B = C[1, 0] - C[0, 1]
    yaw = np.arctan2(B, A)
    R = rot_z(yaw)
    t = mu_g - R @ mu_e
    return R, t, 1.0


def apply(R, t, s, p: np.ndarray) -> np.ndarray:
    return s * p @ R.T + t


def align(p_est, p_gt, method: str = "posyaw"):
    if method == "posyaw":
        return align_posyaw(p_est, p_gt)
    if method == "se3":
        return align_se3(p_est, p_gt)
    if method == "sim3":
        return align_se3(p_est, p_gt, with_scale=True)
    if method == "none":
        return np.eye(3), np.zeros(3), 1.0
    raise ValueError(method)
