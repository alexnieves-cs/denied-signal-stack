"""Dead-reckoning baseline: ES-EKF propagation only, plus a baro-altitude / mag-yaw variant.

Inputs are an IMU stream (+ optional baro/mag) and an initial state; outputs a trajectory with
covariance. On datasets the initial state comes from ground truth (GT init).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dss.core.rotations import quat_to_rot, rot_to_quat, so3_exp
from dss.fusion.eskf import ESKF, ImuNoise
from dss.sensors.samples import Stream


@dataclass
class DrResult:
    t_ns: np.ndarray
    p: np.ndarray
    v: np.ndarray
    q: np.ndarray
    P_pos: np.ndarray
    P_full_diag: np.ndarray


def default_P0(sig_p=0.0, sig_v=0.0, sig_att=0.0, sig_bg=0.0, sig_ba=0.0) -> np.ndarray:
    return np.diag(np.r_[[sig_p**2] * 3, [sig_v**2] * 3, [sig_att**2] * 3, [sig_bg**2] * 3, [sig_ba**2] * 3]) + 1e-18 * np.eye(15)


def perturb_initial(p, v, R, bg, ba, P0, g: np.random.Generator):
    dx = g.multivariate_normal(np.zeros(15), P0)
    return p + dx[0:3], v + dx[3:6], R @ so3_exp(dx[6:9]), bg + dx[9:12], ba + dx[12:15]


def run(imu: Stream, x0: dict, P0: np.ndarray, noise: ImuNoise, t_end_ns: int | None = None, out_hz: float = 20.0,
        baro: Stream | None = None, baro_var: float = 1.0, mag_yaw: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
        mag_yaw_var: float = np.deg2rad(2.0) ** 2) -> DrResult:
    """x0: dict(t_ns, p, v, R, bg, ba). mag_yaw: (t_ns, field_body (N,3), field_world (3,))."""
    mag_yaw_fn = lambda m, i, R: mag_yaw_fn_impl(m[1][i], R, m[2])
    f = ESKF(x0["p"], x0["v"], x0["R"], x0["bg"], x0["ba"], P0, noise)
    t = imu.t_ns
    gyro, acc = imu["gyro"], imu["acc"]
    k0 = int(np.searchsorted(t, x0["t_ns"]))
    k1 = len(t) if t_end_ns is None else int(np.searchsorted(t, t_end_ns, side="right"))
    out_period = int(1e9 / out_hz)
    next_out = int(x0["t_ns"])
    rec_t, rec_p, rec_v, rec_q, rec_P, rec_d = [], [], [], [], [], []
    bi = 0 if baro is None else int(np.searchsorted(baro.t_ns, x0["t_ns"]))
    mi = 0 if mag_yaw is None else int(np.searchsorted(mag_yaw[0], x0["t_ns"]))
    t_prev = int(x0["t_ns"])
    for k in range(k0, k1):
        tk = int(t[k])
        # mean of the bracketing samples (trapezoid), as OpenVINS's discrete propagation
        if k > k0:
            f.propagate(0.5 * (gyro[k - 1] + gyro[k]), 0.5 * (acc[k - 1] + acc[k]), (tk - t_prev) * 1e-9)
        t_prev = tk
        while baro is not None and bi < len(baro) and baro.t_ns[bi] <= tk:
            f.update_altitude(float(baro["alt_m"][bi]), baro_var)
            bi += 1
        while mag_yaw is not None and mi < len(mag_yaw[0]) and mag_yaw[0][mi] <= tk:
            f.update_yaw(mag_yaw_fn(mag_yaw, mi, f.R), mag_yaw_var)
            mi += 1
        if tk >= next_out:
            rec_t.append(tk)
            rec_p.append(f.p.copy())
            rec_v.append(f.v.copy())
            rec_q.append(rot_to_quat(f.R))
            rec_P.append(f.pos_cov().copy())
            rec_d.append(np.diag(f.P).copy())
            next_out += out_period
    return DrResult(np.array(rec_t, dtype=np.int64), np.array(rec_p), np.array(rec_v), np.array(rec_q),
                    np.array(rec_P), np.array(rec_d))


def state_from_gt(gt: Stream, t_ns: int) -> dict:
    i = int(np.searchsorted(gt.t_ns, t_ns))
    i = min(i, len(gt) - 1)
    return {
        "t_ns": int(gt.t_ns[i]),
        "p": gt["p"][i].copy(),
        "v": gt["v"][i].copy(),
        "R": quat_to_rot(gt["q"][i]),
        "bg": np.zeros(3),
        "ba": np.zeros(3),
    }


def mag_yaw_fn_impl(field_b: np.ndarray, R_est: np.ndarray, field_w: np.ndarray) -> float:
    """Tilt-compensated yaw (ENU, rad): remove the estimate's yaw, level the body field, and compare
    its horizontal direction with the known world (WMM) field direction."""
    yaw_est = np.arctan2(R_est[1, 0], R_est[0, 0])
    c, s = np.cos(-yaw_est), np.sin(-yaw_est)
    rz_inv = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    m_level = rz_inv @ R_est @ field_b  # body field in the yaw-free level frame
    alpha = np.arctan2(m_level[1], m_level[0])
    beta = np.arctan2(field_w[1], field_w[0])
    return float((beta - alpha + np.pi) % (2 * np.pi) - np.pi)
