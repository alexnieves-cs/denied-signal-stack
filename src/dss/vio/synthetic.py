"""Synthetic VIO: ground truth + a drift model with an honest covariance (fusion never waits on C++).

Error model per run (restarted by ``reanchor``/``init_gt``):
  e_p(D) = e_p0 + b * D + W(D) + n        position error vs distance travelled D
    b ~ N(0, (drift_frac/sqrt(3))^2 I)     constant per anchor -> |e| ~ drift_frac * D
    W  Brownian in distance, sigma_rw^2 per metre
    n  white, sigma_noise
  e_th(t) = e_th0 + Brownian in time, att_rw^2 per second (local body-frame theta)
Output: p = p_true + e_p, q = q_true (x) exp(e_th), v = v_true + b*|v|.
cov6 [p, theta] = diag(P0_p + (s_b^2 D^2 + s_rw^2 D + s_n^2) I, P0_th + att_rw^2 (t - t0) I), exactly the
distribution the errors are drawn from, so NEES is chi2(6) by construction.
"""

from __future__ import annotations

import numpy as np

from dss.core.rng import rng
from dss.core.rotations import quat_mul, rot_to_quat, so3_exp
from dss.vio.protocol import ST_NOT_INITIALIZED, ST_OK, VioState


class SyntheticVio:
    def __init__(self, seed: int, drift_frac: float = 0.005, sigma_rw: float = 0.02, att_rw: float = 2e-4,
                 sigma_noise: float = 0.05, name: str = "synthetic"):
        self.g = rng(seed, "vio", name)
        self.drift_frac = drift_frac
        self.s_b = drift_frac / np.sqrt(3.0)
        self.s_rw = sigma_rw
        self.att_rw = att_rw
        self.s_n = sigma_noise
        self.anchored = False
        self._pending: tuple | None = None
        self.n_tracked = 150
        self.vision_ok = True

    # VioBackend no-ops (truth-driven)
    def init(self) -> None:
        pass

    def imu(self, t_s, w, a) -> None:
        pass

    def close(self) -> None:
        pass

    def init_gt(self, t_s, q_wxyz, p, v, bg=(0, 0, 0), ba=(0, 0, 0)) -> None:
        self._pending = (np.asarray(p, float), np.asarray(q_wxyz, float), np.zeros((6, 6)))

    def reanchor(self, t_s, q_wxyz, p, v, bg, ba, P15: np.ndarray) -> None:
        P15 = np.asarray(P15)
        P0 = np.zeros((6, 6))
        P0[:3, :3] = P15[0:3, 0:3]
        P0[3:, 3:] = P15[6:9, 6:9]
        P0[:3, 3:] = P15[0:3, 6:9]
        P0[3:, :3] = P15[6:9, 0:3]
        self._pending = (np.asarray(p, float), np.asarray(q_wxyz, float), P0)

    def _anchor(self, t_s, p_true, q_true, pending) -> None:
        p_a, q_a, P0 = pending
        self.e_p0 = p_a - p_true
        from dss.core.rotations import quat_conj, quat_to_rot, so3_log

        self.e_th0 = so3_log(quat_to_rot(quat_mul(quat_conj(q_true), q_a)))
        self.P0 = P0
        self.b = self.g.normal(0, self.s_b, 3)
        self.W = np.zeros(3)
        self.th = np.zeros(3)
        self.D = 0.0
        self.t0 = self.t_prev = t_s
        self.p_prev = np.asarray(p_true, float)
        self.anchored = True

    def step(self, t_s: float, p_true, q_true, v_true) -> VioState:
        p_true = np.asarray(p_true, float)
        q_true = np.asarray(q_true, float)
        v_true = np.asarray(v_true, float)
        if self._pending is not None:
            self._anchor(t_s, p_true, q_true, self._pending)
            self._pending = None
        if not self.anchored:
            return VioState(ST_NOT_INITIALIZED, float("nan"), np.zeros(3), np.array([1.0, 0, 0, 0]), np.zeros(3), np.zeros((6, 6)))
        dD = float(np.linalg.norm(p_true - self.p_prev))
        dt = max(t_s - self.t_prev, 0.0)
        self.D += dD
        self.W += self.g.normal(0, self.s_rw * np.sqrt(dD), 3) if dD > 0 else 0.0
        self.th += self.g.normal(0, self.att_rw * np.sqrt(dt), 3) if dt > 0 else 0.0
        n = self.g.normal(0, self.s_n, 3)
        self.p_prev, self.t_prev = p_true, t_s
        e_p = self.e_p0 + self.b * self.D + self.W + n
        e_th = self.e_th0 + self.th
        q = quat_mul(q_true, rot_to_quat(so3_exp(e_th)))
        var_p = self.s_b**2 * self.D**2 + self.s_rw**2 * self.D + self.s_n**2
        var_th = self.att_rw**2 * (t_s - self.t0)
        cov = self.P0.copy()
        cov[:3, :3] += var_p * np.eye(3)
        cov[3:, 3:] += var_th * np.eye(3) + 1e-12 * np.eye(3)
        return VioState(ST_OK, float(t_s), p_true + e_p, q if q[0] >= 0 else -q, v_true + self.b * np.linalg.norm(v_true),
                        cov, n_tracked=self.n_tracked, n_msckf=40, n_slam=25)
