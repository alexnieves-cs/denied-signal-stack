"""Error-state EKF (Sola 2017) for an IMU-driven INS, ENU world, FLU body.

Nominal state: p, v, R (world<-body), bg, ba. Error state (15): [dp, dv, dtheta(local), dbg, dba].
Used as the dead-reckoning baseline (propagation only), the baro/mag variant, and the Phase 2
fusion fallback behind the same backend interface.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dss.core.rotations import skew, so3_exp

G_W = np.array([0.0, 0.0, -9.81])
IP, IV, IT, IBG, IBA = slice(0, 3), slice(3, 6), slice(6, 9), slice(9, 12), slice(12, 15)


@dataclass
class ImuNoise:
    sigma_g: float
    sigma_a: float
    sigma_bg: float
    sigma_ba: float

    @staticmethod
    def from_config(c: dict) -> ImuNoise:
        return ImuNoise(
            c["gyroscope_noise_density"], c["accelerometer_noise_density"],
            c["gyroscope_random_walk"], c["accelerometer_random_walk"],
        )


class ESKF:
    def __init__(self, p, v, r, bg, ba, P0: np.ndarray, noise: ImuNoise, g_w: np.ndarray = G_W):
        self.p = np.array(p, dtype=float)
        self.v = np.array(v, dtype=float)
        self.R = np.array(r, dtype=float)
        self.bg = np.array(bg, dtype=float)
        self.ba = np.array(ba, dtype=float)
        self.P = np.array(P0, dtype=float)
        self.n = noise
        self.g = np.asarray(g_w, dtype=float)
        self.last_imu: tuple[np.ndarray, np.ndarray] | None = None

    def copy(self) -> ESKF:
        e = ESKF(self.p, self.v, self.R, self.bg, self.ba, self.P, self.n, self.g)
        e.last_imu = self.last_imu
        return e

    # --- propagation -------------------------------------------------------------------
    def propagate(self, gyro: np.ndarray, acc: np.ndarray, dt: float) -> None:
        if dt <= 0:
            return
        w = gyro - self.bg
        a = acc - self.ba
        r = self.R
        # rotate the specific force with the mid-interval attitude (second-order accurate)
        acc_w = r @ so3_exp(0.5 * w * dt) @ a + self.g
        self.p = self.p + self.v * dt + 0.5 * acc_w * dt * dt
        self.v = self.v + acc_w * dt
        dr = so3_exp(w * dt)
        self.R = r @ dr
        # error-state transition (first order in dt, exact rotation block)
        F = np.eye(15)
        F[IP, IV] = np.eye(3) * dt
        F[IV, IT] = -r @ skew(a) * dt
        F[IV, IBA] = -r * dt
        F[IT, IT] = dr.T
        F[IT, IBG] = -np.eye(3) * dt
        Q = np.zeros((15, 15))
        n = self.n
        Q[IV, IV] = np.eye(3) * n.sigma_a**2 * dt
        Q[IT, IT] = np.eye(3) * n.sigma_g**2 * dt
        Q[IBG, IBG] = np.eye(3) * n.sigma_bg**2 * dt
        Q[IBA, IBA] = np.eye(3) * n.sigma_ba**2 * dt
        # velocity noise also enters position at second order
        Q[IP, IP] = np.eye(3) * n.sigma_a**2 * dt**3 / 3
        Q[IP, IV] = Q[IV, IP] = np.eye(3) * n.sigma_a**2 * dt**2 / 2
        self.P = F @ self.P @ F.T + Q
        self.P = 0.5 * (self.P + self.P.T)
        self.last_imu = (gyro, acc)

    # --- updates -----------------------------------------------------------------------
    def update(self, r: np.ndarray, H: np.ndarray, Rm: np.ndarray, gate_chi2: float | None = None) -> tuple[bool, float]:
        """Linear(ized) update with residual r = z - h(x). Returns (accepted, NIS)."""
        S = H @ self.P @ H.T + Rm
        S = 0.5 * (S + S.T)
        Si = np.linalg.inv(S)
        nis = float(r @ Si @ r)
        if gate_chi2 is not None and nis > gate_chi2:
            return False, nis
        K = self.P @ H.T @ Si
        dx = K @ r
        ikh = np.eye(15) - K @ H
        self.P = ikh @ self.P @ ikh.T + K @ Rm @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        self.inject(dx)
        return True, nis

    def inject(self, dx: np.ndarray) -> None:
        self.p = self.p + dx[IP]
        self.v = self.v + dx[IV]
        self.R = self.R @ so3_exp(dx[IT])
        self.bg = self.bg + dx[IBG]
        self.ba = self.ba + dx[IBA]
        # reset Jacobian G = I - 0.5 skew(dtheta) (first order); negligible for small dtheta
        G = np.eye(15)
        G[IT, IT] = np.eye(3) - 0.5 * skew(dx[IT])
        self.P = G @ self.P @ G.T

    def update_position(self, z: np.ndarray, cov: np.ndarray, gate_chi2=None, dims=(0, 1, 2)):
        dims = list(dims)
        H = np.zeros((len(dims), 15))
        for i, d in enumerate(dims):
            H[i, d] = 1.0
        return self.update(np.asarray(z)[dims] - self.p[dims] if len(z) == 3 else np.asarray(z) - self.p[dims],
                           H, np.asarray(cov), gate_chi2)

    def update_velocity(self, z: np.ndarray, cov: np.ndarray, gate_chi2=None):
        H = np.zeros((3, 15))
        H[:, IV] = np.eye(3)
        return self.update(z - self.v, H, cov, gate_chi2)

    def update_altitude(self, z: float, var: float, gate_chi2=None):
        H = np.zeros((1, 15))
        H[0, 2] = 1.0
        return self.update(np.array([z - self.p[2]]), H, np.array([[var]]), gate_chi2)

    def update_yaw(self, yaw_meas: float, var: float, gate_chi2=None):
        """Yaw about world z. With a local attitude error, d(yaw)/d(dtheta) = e_z^T R (third row)."""
        yaw = np.arctan2(self.R[1, 0], self.R[0, 0])
        r = (yaw_meas - yaw + np.pi) % (2 * np.pi) - np.pi
        H = np.zeros((1, 15))
        H[0, IT] = self.R[2, :]
        return self.update(np.array([r]), H, np.array([[var]]), gate_chi2)

    def pos_cov(self) -> np.ndarray:
        return self.P[IP, IP]
