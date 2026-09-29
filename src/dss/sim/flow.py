"""Downward optical-flow sensor (MAVLink OPTICAL_FLOW_RAD semantics).

Over each integration interval it reports integrated flow angles (rad) about body x/y and the
integrated gyro. Translational flow = v_body_xy * dt / h; the rotational part equals the
integrated gyro and is compensated by the consumer. Valid only at <= max_agl_m.
"""

from __future__ import annotations

import numpy as np

from dss.core.rng import rng
from dss.core.time import s_to_ns


def simulate_flow(traj, t: np.ndarray, cfg: dict, seed: int, surface_z, scale_error: float = 0.0) -> dict:
    g = rng(seed, "flow", cfg.get("name", "flow0"))
    dt = 1.0 / cfg["rate_hz"]
    # integrate over [t-dt, t] with 10 sub-steps
    sub = np.linspace(-dt, 0.0, 11)
    fx = np.zeros(len(t))
    fy = np.zeros(len(t))
    gx = np.zeros(len(t))
    gy = np.zeros(len(t))
    gz = np.zeros(len(t))
    for i in range(len(sub) - 1):
        tm = t + 0.5 * (sub[i] + sub[i + 1])
        s = traj.sample(tm)
        vb = np.einsum("nji,nj->ni", s.R, s.v)
        h = s.p[:, 2] - surface_z(s.p[:, 0], s.p[:, 1])
        h_perp = h * s.R[:, 2, 2]  # distance along optical axis (body -z) to a flat patch
        h_perp = np.maximum(h_perp, 0.05)
        step = dt / (len(sub) - 1)
        # scene appears to move opposite to the vehicle; RH convention: flow_x about +x = +vy/h
        fx += (vb[:, 1] / h_perp + s.omega_b[:, 0]) * step
        fy += (-vb[:, 0] / h_perp + s.omega_b[:, 1]) * step
        gx += s.omega_b[:, 0] * step
        gy += s.omega_b[:, 1] * step
        gz += s.omega_b[:, 2] * step
    fx = (fx - gx) * (1 + scale_error) + gx + g.normal(0, cfg["noise_rad"], len(t))
    fy = (fy - gy) * (1 + scale_error) + gy + g.normal(0, cfg["noise_rad"], len(t))
    s = traj.sample(t)
    agl = s.p[:, 2] - surface_z(s.p[:, 0], s.p[:, 1])
    valid = (agl <= cfg["max_agl_m"]) & (agl > 0.3)
    return {
        "t_ns": s_to_ns(t),
        "integration_s": np.full(len(t), dt),
        "flow_xy": np.stack([fx, fy], 1),
        "gyro_xyz": np.stack([gx, gy, gz], 1),
        "valid": valid,
        "agl_true": agl,
    }


def flow_to_velocity(flow_xy, gyro_xyz, dt, h_perp):
    """Invert the model: body-frame horizontal velocity from flow + distance."""
    vy = (flow_xy[:, 0] - gyro_xyz[:, 0]) / dt * h_perp
    vx = -(flow_xy[:, 1] - gyro_xyz[:, 1]) / dt * h_perp
    return np.stack([vx, vy], 1)
