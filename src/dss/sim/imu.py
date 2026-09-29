"""IMU model with the Kalibr/OpenVINS discretization.

Continuous densities (sigma_g [rad/s/sqrt(Hz)], sigma_bg [rad/s^2/sqrt(Hz)], ...) become
  white noise  n_d  = sigma / sqrt(dt)
  bias walk    b_k+1 = b_k + sigma_b * sqrt(dt) * w
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dss.core.rng import rng
from dss.core.time import s_to_ns
from dss.sim.trajectory import Trajectory


@dataclass
class ImuParams:
    rate_hz: float
    gyroscope_noise_density: float
    gyroscope_random_walk: float
    accelerometer_noise_density: float
    accelerometer_random_walk: float
    initial_bias_sigma_gyro: float = 0.0
    initial_bias_sigma_accel: float = 0.0
    saturation_gyro: float = np.inf
    saturation_accel: float = np.inf

    @staticmethod
    def from_config(c: dict) -> ImuParams:
        ib = c.get("initial_bias_sigma", {})
        sat = c.get("saturation", {})
        return ImuParams(
            rate_hz=float(c["rate_hz"]),
            gyroscope_noise_density=float(c["gyroscope_noise_density"]),
            gyroscope_random_walk=float(c["gyroscope_random_walk"]),
            accelerometer_noise_density=float(c["accelerometer_noise_density"]),
            accelerometer_random_walk=float(c["accelerometer_random_walk"]),
            initial_bias_sigma_gyro=float(ib.get("gyro", 0.0)),
            initial_bias_sigma_accel=float(ib.get("accel", 0.0)),
            saturation_gyro=float(sat.get("gyro", np.inf)),
            saturation_accel=float(sat.get("accel", np.inf)),
        )


def simulate_imu(traj: Trajectory, p: ImuParams, seed: int, t0: float | None = None, t1: float | None = None,
                 noise: bool = True, name: str = "imu0") -> dict:
    t0 = traj.t0 if t0 is None else t0
    t1 = traj.t1 if t1 is None else t1
    dt = 1.0 / p.rate_hz
    n = int(np.floor((t1 - t0) / dt + 1e-9)) + 1
    t = t0 + dt * np.arange(n)
    s = traj.sample(t)
    w_true, a_true = s.omega_b, s.f_b
    g = rng(seed, "imu", name)
    if noise:
        bg0 = g.normal(0, p.initial_bias_sigma_gyro, 3)
        ba0 = g.normal(0, p.initial_bias_sigma_accel, 3)
        wg = g.standard_normal((n, 3))
        wa = g.standard_normal((n, 3))
        wbg = g.standard_normal((n, 3))
        wba = g.standard_normal((n, 3))
        bg = bg0 + np.vstack([np.zeros(3), np.cumsum(p.gyroscope_random_walk * np.sqrt(dt) * wbg[:-1], axis=0)])
        ba = ba0 + np.vstack([np.zeros(3), np.cumsum(p.accelerometer_random_walk * np.sqrt(dt) * wba[:-1], axis=0)])
        gyro = w_true + bg + p.gyroscope_noise_density / np.sqrt(dt) * wg
        acc = a_true + ba + p.accelerometer_noise_density / np.sqrt(dt) * wa
    else:
        bg = np.zeros((n, 3))
        ba = np.zeros((n, 3))
        gyro, acc = w_true.copy(), a_true.copy()
    sat = (np.abs(gyro) > p.saturation_gyro).any(1) | (np.abs(acc) > p.saturation_accel).any(1)
    gyro = np.clip(gyro, -p.saturation_gyro, p.saturation_gyro)
    acc = np.clip(acc, -p.saturation_accel, p.saturation_accel)
    return {
        "t_ns": s_to_ns(t),
        "gyro": gyro,
        "acc": acc,
        "bias_gyro": bg,
        "bias_acc": ba,
        "saturated": sat,
    }
