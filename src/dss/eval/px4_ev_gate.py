"""EKF2-style external-vision gate emulator.

A position/velocity Kalman filter driven by the sim IMU (truth specific force + BMI160-class noise, as EKF2
would integrate it) fuses VISION_POSITION_ESTIMATE positions with the VPE's own variance and applies EKF2's
innovation gate (EKF2_EVP_GATE = 5 SD). Reports rejections and the longest rejection streak.
"""

from __future__ import annotations

import numpy as np


def replay(t: np.ndarray, p_vpe: np.ndarray, var_vpe: np.ndarray, acc_w_true: np.ndarray, seed: int = 0,
           gate_sd: float = 5.0, acc_noise: float = 0.35) -> dict:
    g = np.random.default_rng(seed)
    x = np.r_[p_vpe[0], np.zeros(3)]
    P = np.diag([var_vpe[0]] * 3 + [1.0] * 3)
    rej = np.zeros(len(t), bool)
    for k in range(1, len(t)):
        dt = t[k] - t[k - 1]
        a = acc_w_true[k] + g.normal(0, acc_noise, 3)
        F = np.eye(6)
        F[:3, 3:] = np.eye(3) * dt
        x = F @ x + np.r_[0.5 * a * dt * dt, a * dt]
        Q = np.zeros((6, 6))
        Q[3:, 3:] = np.eye(3) * acc_noise**2 * dt
        Q[:3, :3] = np.eye(3) * acc_noise**2 * dt**3 / 3
        P = F @ P @ F.T + Q
        r = p_vpe[k] - x[:3]
        S = P[:3, :3] + np.eye(3) * var_vpe[k]
        tr = r**2 / np.diag(S)
        if np.any(tr > gate_sd**2):
            rej[k] = True
            continue
        K = P[:, :3] @ np.linalg.inv(S)
        x = x + K @ r
        P = (np.eye(6) - K[:, :3] @ np.eye(3, 6)) @ P
    streak = 0
    best = 0
    for v in rej:
        streak = streak + 1 if v else 0
        best = max(best, streak)
    dt = float(np.median(np.diff(t)))
    return {"rejections": int(rej.sum()), "max_streak_s": best * dt, "samples": int(len(t))}
