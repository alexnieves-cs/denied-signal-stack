"""Fusion backend (ES-EKF, PLAN 2.6 fallback promoted to primary; see LEDGER Phase 2).

Two filters share one implementation:
  REF — IMU + VIO velocity + map fixes + baro + lidar + mag. Never sees GPS.
  ALL — REF's inputs + GPS, which arrives only through the integrity gate.
Outage mode (vision FAILED): VIO updates stop; the filter coasts on IMU + baro + lidar + mag.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2

from dss.fusion.eskf import ESKF, IP, ImuNoise

CHI2_2_99 = chi2.ppf(0.99, 2)  # 9.21
CHI2_3_999 = chi2.ppf(0.999, 3)
CHI2_2_999 = chi2.ppf(0.999, 2)


class NavFilter:
    def __init__(self, name: str, x0: dict, P0: np.ndarray, noise: ImuNoise, cfg: dict):
        self.name = name
        self.f = ESKF(x0["p"], x0["v"], x0["R"], x0["bg"], x0["ba"], P0, noise, n_aug=3,
                      aug_sigma=float(cfg.get("vio_bias_sigma", 0.10)), aug_tau=float(cfg.get("vio_bias_tau", 60.0)))
        self.f.q_vel_extra = float(cfg.get("q_vel_extra", 0.10))
        self.cfg = cfg
        self.t_ns = int(x0["t_ns"])
        self.last_fix_t = None
        self.stats = {"vio": 0, "map_acc": 0, "map_rej": 0, "baro": 0, "lidar": 0, "mag": 0, "gps": 0, "gps_rej": 0}
        self.consecutive_fix = 0

    # --- state access
    @property
    def p(self):
        return self.f.p

    @property
    def v(self):
        return self.f.v

    @property
    def R(self):
        return self.f.R

    @property
    def P(self):
        return self.f.P

    def pos_cov(self):
        return self.f.P[IP, IP]

    def copy_from(self, other: NavFilter):
        self.f = other.f.copy()
        self.t_ns = other.t_ns

    def propagate(self, gyro, acc, t_ns: int):
        dt = (t_ns - self.t_ns) * 1e-9
        self.f.propagate(gyro, acc, dt)
        self.t_ns = t_ns

    # --- aiding
    def vio_velocity(self, v_w: np.ndarray, sigma: float) -> bool:
        # horizontal only: vertical is held by baro + lidar, and mono VIO's vertical velocity degrades on descent
        ok, _ = self.f.update_velocity(v_w, np.eye(3) * sigma**2, gate_chi2=CHI2_2_999, with_bias=True, dims=(0, 1))
        self.stats["vio"] += int(ok)
        if not ok:
            self.stats["vio_rej"] = self.stats.get("vio_rej", 0) + 1
        return ok

    def map_fix(self, xy: np.ndarray, cov: np.ndarray, require_consecutive: int = 1, quality: float = 0.0
                ) -> tuple[bool, float]:
        """Absolute (x, y) fix. chi2_2 gate vs this filter's prediction; after an outage the fix must also
        agree with the previous candidate (pairwise-consistent, relative motion removed). Consensus
        relocalization: if >= 4 recent strong rejected fixes agree with each other, the filter is judged
        wrong (not the fixes): its position covariance is inflated by the consensus offset and the fix applied."""
        H = np.zeros((2, self.f.nx))
        H[0, 0] = H[1, 1] = 1.0
        r = xy - self.f.p[:2]
        S = H @ self.f.P @ H.T + cov
        nis = float(r @ np.linalg.solve(S, r))
        if not hasattr(self, "_rej"):
            self._rej, self._prev_cand = [], None
        if nis > CHI2_2_99:
            self.stats["map_rej"] += 1
            self.consecutive_fix = 0
            if quality >= 2.0:
                self._rej.append(r.copy())
                self._rej = self._rej[-4:]
                if len(self._rej) == 4:
                    R4 = np.array(self._rej)
                    if np.max(np.linalg.norm(R4 - R4.mean(0), axis=1)) < 3.0:
                        off = R4.mean(0)
                        self.f.P[0:2, 0:2] += np.outer(off, off) + np.eye(2) * 4.0
                        self.f.update(r, H, cov)
                        self.stats["map_reloc"] = self.stats.get("map_reloc", 0) + 1
                        self._rej = []
                        self.last_fix_t = self.t_ns
                        return True, nis
            return False, nis
        self._rej = []
        cand = (xy - self.f.p[:2]).copy()
        consistent = self._prev_cand is None or np.linalg.norm(cand - self._prev_cand) < 3.0
        self._prev_cand = cand
        self.consecutive_fix = self.consecutive_fix + 1 if consistent else 1
        if self.consecutive_fix < require_consecutive:
            return False, nis
        self.f.update(r, H, cov)
        self.stats["map_acc"] += 1
        self.last_fix_t = self.t_ns
        return True, nis

    def flow_velocity(self, v_body_xy: np.ndarray, sigma: float):
        """Optical flow (<= 20 m AGL): horizontal body-frame velocity."""
        from dss.core.rotations import skew

        H = np.zeros((2, self.f.nx))
        Rt = self.f.R.T
        H[:, 3:6] = Rt[:2, :]
        H[:, 6:9] = skew(Rt @ self.f.v)[:2, :]
        r = v_body_xy - (Rt @ self.f.v)[:2]
        ok, _ = self.f.update(r, H, np.eye(2) * sigma**2, gate_chi2=25.0)
        self.stats.setdefault("flow", 0)
        self.stats["flow"] += int(ok)

    def baro(self, alt: float, var: float):
        ok, _ = self.f.update_altitude(alt, var, gate_chi2=25.0)
        self.stats["baro"] += int(ok)

    def lidar(self, rng_m: float, var: float, ground_z: float):
        # range along body -z: z = ground + r * cos(tilt)
        c = self.f.R[2, 2]
        ok, _ = self.f.update_altitude(ground_z + rng_m * c, var * c * c + self.cfg.get("dem_var", 1.0), gate_chi2=25.0)
        self.stats["lidar"] += int(ok)

    def mag_yaw(self, yaw: float, var: float):
        ok, _ = self.f.update_yaw(yaw, var, gate_chi2=25.0)
        self.stats["mag"] += int(ok)

    def gps(self, pos: np.ndarray, cov: np.ndarray, gate: float = 13.82, corr_inflation: float = 1.0) -> tuple[bool, float]:
        """GPS position; ``corr_inflation`` de-weights samples of a time-correlated (Gauss-Markov) error."""
        H = np.zeros((2, self.f.nx))
        H[0, 0] = H[1, 1] = 1.0
        r = pos[:2] - self.f.p[:2]
        S = self.f.P[:2, :2] + cov[:2, :2]
        nis = float(r @ np.linalg.solve(S, r))
        if nis > gate:
            self.stats["gps_rej"] += 1
            return False, nis
        ok, _ = self.f.update(r, H, cov[:2, :2] * corr_inflation)
        self.stats["gps" if ok else "gps_rej"] += 1
        return ok, nis
