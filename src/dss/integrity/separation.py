"""Published output: REF (+) clamp(ALL - REF), rate-limited (bled) so no absolute correction appears as a jump.

p_pub(k) = p_pub(k-1) + v_target dt + clip(target - predicted, 0.5 m/s * dt); yaw likewise at <= 2 deg/s.
The published covariance carries the un-bled offset: P_pub = P_target + e e^T.
HPL(t) = 5.68 sqrt(lambda_max(P_pub,h)) (1e-7 integrity risk, 2-D Gaussian).
"""

from __future__ import annotations

import numpy as np

from dss.core.rotations import rot_z, wrap_angle, yaw_of

K_SEP = 5.0
HPL_K = 5.68


def separation_stat(p_all, P_all, p_ref, P_ref, floor_var: float = 1.27**2) -> float:
    """|ALL - REF| / sqrt(lambda_max(P_REF - P_ALL)), floored by the GNSS coloured-error variance
    (the separation can never be expected to be smaller than GPS's own correlated error)."""
    d = p_all[:2] - p_ref[:2]
    D = P_ref[:2, :2] - P_all[:2, :2]
    lam = max(float(np.max(np.linalg.eigvalsh(0.5 * (D + D.T)))), floor_var)
    return float(np.linalg.norm(d) / np.sqrt(lam))


def clamp_target(p_all, P_all, p_ref, P_ref):
    d = p_all - p_ref
    D = P_ref[:3, :3] - P_all[:3, :3]
    lam = max(float(np.max(np.linalg.eigvalsh(0.5 * (D + D.T)))), 1e-6)
    lim = K_SEP * np.sqrt(lam)
    n = np.linalg.norm(d)
    return p_ref + (d if n <= lim else d * lim / n)


class Publisher:
    def __init__(self, p0, R0, P0, v_max: float = 0.5, yaw_rate_max: float = np.deg2rad(2.0)):
        self.p = np.array(p0, float)
        self.R = np.array(R0, float)
        self.P = np.array(P0, float)
        self.v_max = v_max
        self.w_max = yaw_rate_max
        self.reset_counter = 0
        self.last_jump = 0.0
        self.bleeding = False

    def step(self, dt: float, target_p, target_v, target_R, target_P, yaw_rate: float = 0.0):
        pred = self.p + target_v * dt
        e = target_p - pred
        n = np.linalg.norm(e)
        lim = self.v_max * dt
        step = e if n <= lim else e * lim / n
        self.p = pred + step
        self.bleeding = n > lim
        # attitude: take roll/pitch from target, bleed yaw
        yaw_pred = yaw_of(self.R) + yaw_rate * dt
        dyaw = float(wrap_angle(yaw_of(target_R) - yaw_pred))
        yaw_new = yaw_pred + float(np.clip(dyaw, -self.w_max * dt, self.w_max * dt))
        lag = float(wrap_angle(yaw_of(target_R) - yaw_new))
        self.R = rot_z(-lag) @ target_R  # target tilt, published yaw bled toward the target
        resid = target_p - self.p
        self.P = np.array(target_P[:3, :3], float) + np.outer(resid, resid)
        return self.p, self.R, self.P

    def hpl(self) -> float:
        return HPL_K * float(np.sqrt(max(np.max(np.linalg.eigvalsh(self.P[:2, :2])), 0.0)))
