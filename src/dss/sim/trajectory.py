"""Differentially-flat multirotor trajectories (kinematic; no physics engine, PLAN F6).

Flat outputs are position p(t) (ENU) and yaw psi(t). They are represented as quintic
interpolating B-splines (C4), so jerk and snap are continuous: body rates and angular
acceleration have no jumps at joints. Attitude follows from the thrust direction
z_b = (a + g e_z)/|a + g e_z| and yaw; body rates are vee(R^T dR/dt).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import make_interp_spline

from dss.core.rotations import rot_to_quat

GRAVITY = 9.81
E_Z = np.array([0.0, 0.0, 1.0])


def smoothstep7(x):
    """Septic smoothstep: C3 at both ends (zero 1st-3rd derivatives)."""
    x = np.clip(x, 0.0, 1.0)
    return x**4 * (35 - 84 * x + 70 * x**2 - 20 * x**3)


def _ramp(t, t0, t1):
    return smoothstep7((np.asarray(t) - t0) / (t1 - t0))


@dataclass
class TrajSample:
    t: np.ndarray
    p: np.ndarray
    v: np.ndarray
    a: np.ndarray
    R: np.ndarray  # (N,3,3) R_world_body (ENU <- FLU)
    omega_b: np.ndarray  # body rates, rad/s, body frame
    f_b: np.ndarray  # specific force, body frame
    yaw: np.ndarray

    @property
    def q(self) -> np.ndarray:
        return np.array([rot_to_quat(r) for r in self.R])


class Trajectory:
    def __init__(self, t_knots: np.ndarray, p_knots: np.ndarray, yaw_knots: np.ndarray, g: float = GRAVITY):
        self.t0 = float(t_knots[0])
        self.t1 = float(t_knots[-1])
        self.g = g
        self._p = make_interp_spline(t_knots, p_knots, k=5)
        self._yaw = make_interp_spline(t_knots, np.unwrap(yaw_knots), k=5)
        self._v = self._p.derivative(1)
        self._a = self._p.derivative(2)

    @property
    def duration(self) -> float:
        return self.t1 - self.t0

    def _rot(self, t: np.ndarray) -> np.ndarray:
        a = self._a(t)
        f = a + self.g * E_Z
        zb = f / np.linalg.norm(f, axis=1, keepdims=True)
        psi = self._yaw(t)
        xc = np.stack([np.cos(psi), np.sin(psi), np.zeros_like(psi)], axis=1)
        yb = np.cross(zb, xc)
        yb /= np.linalg.norm(yb, axis=1, keepdims=True)
        xb = np.cross(yb, zb)
        return np.stack([xb, yb, zb], axis=2)

    def rotation(self, t) -> np.ndarray:
        return self._rot(np.atleast_1d(np.asarray(t, dtype=float)))

    def body_rates(self, t: np.ndarray, h: float = 1e-3) -> np.ndarray:
        t = np.atleast_1d(np.asarray(t, dtype=float))
        r = self._rot(t)
        # 4th-order central difference of R(t); truncation O(h^4) ~ 1e-13 for these trajectories
        rd = (-self._rot(t + 2 * h) + 8 * self._rot(t + h) - 8 * self._rot(t - h) + self._rot(t - 2 * h)) / (12 * h)
        w = np.einsum("nji,njk->nik", r, rd)  # R^T Rdot
        a = 0.5 * (w - np.transpose(w, (0, 2, 1)))
        return np.stack([a[:, 2, 1], a[:, 0, 2], a[:, 1, 0]], axis=1)

    def sample(self, t) -> TrajSample:
        t = np.atleast_1d(np.asarray(t, dtype=float))
        p = self._p(t)
        v = self._v(t)
        a = self._a(t)
        r = self._rot(t)
        f_w = a + self.g * E_Z
        f_b = np.einsum("nji,nj->ni", r, f_w)
        return TrajSample(t=t, p=p, v=v, a=a, R=r, omega_b=self.body_rates(t), f_b=f_b, yaw=self._yaw(t))


def static(duration: float = 60.0, p0=(0.0, 0.0, 0.0), yaw0: float = 0.0) -> Trajectory:
    t = np.linspace(0.0, duration, int(duration * 10) + 1)
    p = np.tile(np.asarray(p0, dtype=float), (len(t), 1))
    return Trajectory(t, p, np.full(len(t), yaw0))


def route(
    length_m: float = 2000.0,
    agl_m: float = 100.0,
    speed: float = 9.0,
    t_takeoff: float = 5.0,
    t_climb_end: float = 25.0,
    t_ramp: float = 10.0,
    t_accel: float = 18.0,
    descent_s: float = 25.0,
    hold_after: float = 5.0,
    heading0: float = 0.0,
    turn_amp_rad: float = 0.6,
    turn_wavelength_m: float = 1000.0,
    ground_z: float = 0.0,
    knot_hz: float = 20.0,
) -> Trajectory:
    """S-turn route: static init, climb, cruise at ``speed`` for ``length_m``, descend, land."""
    cruise_dist = length_m - speed * t_ramp  # two septic ramps each cover v*t_ramp/2
    t_decel = t_accel + t_ramp + cruise_dist / speed
    t_stop = t_decel + t_ramp
    t_desc0 = t_stop - 2.0
    t_land = t_desc0 + descent_s
    t_end = t_land + hold_after
    tt = np.arange(0.0, t_end + 1e-9, 1.0 / knot_hz)
    # arc length s(t) = integral of v(t); integrate finely
    fine = np.arange(0.0, t_end + 1e-9, 1e-3)
    vf = speed * (_ramp(fine, t_accel, t_accel + t_ramp) - _ramp(fine, t_decel, t_stop))
    sf = np.concatenate([[0.0], np.cumsum(0.5 * (vf[1:] + vf[:-1]) * 1e-3)])
    s = np.interp(tt, fine, sf)
    # arc-length parametrized planar curve: heading psi(s)
    ds = 0.05
    sg = np.arange(0.0, length_m + 50.0, ds)
    psi_g = heading0 + turn_amp_rad * np.sin(2 * np.pi * sg / turn_wavelength_m)
    xg = np.concatenate([[0.0], np.cumsum(0.5 * (np.cos(psi_g[1:]) + np.cos(psi_g[:-1])) * ds)])
    yg = np.concatenate([[0.0], np.cumsum(0.5 * (np.sin(psi_g[1:]) + np.sin(psi_g[:-1])) * ds)])
    x = np.interp(s, sg, xg)
    y = np.interp(s, sg, yg)
    yaw = np.interp(s, sg, psi_g)
    z = ground_z + agl_m * (_ramp(tt, t_takeoff, t_climb_end) - _ramp(tt, t_desc0, t_land))
    traj = Trajectory(tt, np.stack([x, y, z], axis=1), yaw)
    traj.events = {  # type: ignore[attr-defined]
        "takeoff": t_takeoff,
        "climb_end": t_climb_end,
        "cruise_start": t_accel + t_ramp,
        "cruise_end": t_decel,
        "descent_start": t_desc0,
        "touchdown": t_land,
        "end": t_end,
    }
    return traj


def from_config(cfg: dict) -> Trajectory:
    kind = cfg.get("kind", "route")
    params = {k: v for k, v in cfg.items() if k != "kind"}
    if kind == "static":
        return static(**params)
    if kind == "route":
        return route(**params)
    raise ValueError(kind)
