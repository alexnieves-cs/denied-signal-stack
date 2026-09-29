"""Rotation utilities. Quaternions are Hamilton, scalar-first ``[w, x, y, z]`` (ADR-0002).

``quat_to_rot(q)`` returns R such that ``v_parent = R @ v_child`` for a quaternion that
describes the child frame's orientation in the parent frame. OpenVINS's JPL convention is
converted only at the protocol boundary (:func:`jpl_to_hamilton`).
"""

from __future__ import annotations

import numpy as np

EPS = 1e-12


def skew(v: np.ndarray) -> np.ndarray:
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def vee(m: np.ndarray) -> np.ndarray:
    return np.array([m[2, 1], m[0, 2], m[1, 0]])


def so3_exp(phi: np.ndarray) -> np.ndarray:
    phi = np.asarray(phi, dtype=float)
    th = np.linalg.norm(phi)
    k = skew(phi)
    if th < 1e-8:
        return np.eye(3) + k + 0.5 * k @ k
    return np.eye(3) + np.sin(th) / th * k + (1 - np.cos(th)) / th**2 * k @ k


def so3_log(r: np.ndarray) -> np.ndarray:
    c = np.clip((np.trace(r) - 1.0) / 2.0, -1.0, 1.0)
    th = np.arccos(c)
    if th < 1e-8:
        return vee(r - r.T) / 2.0
    if np.pi - th < 1e-6:
        # near pi: use the symmetric part
        a = (r + np.eye(3)) / 2.0
        i = int(np.argmax(np.diag(a)))
        axis = a[:, i] / np.sqrt(max(a[i, i], EPS))
        return axis * th
    return th / (2 * np.sin(th)) * vee(r - r.T)


def quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    return q


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    )


def quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    w, x, y, z = quat_normalize(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def rot_to_quat(r: np.ndarray) -> np.ndarray:
    """Shepperd's method; returns w >= 0."""
    t = np.trace(r)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        q = np.array([0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s])
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2
        q = np.array([(r[2, 1] - r[1, 2]) / s, 0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s])
    elif r[1, 1] > r[2, 2]:
        s = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2
        q = np.array([(r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s])
    else:
        s = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2
        q = np.array([(r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s])
    q = quat_normalize(q)
    return q if q[0] >= 0 else -q


def rot_to_euler_zyx(r: np.ndarray) -> np.ndarray:
    """Return (roll, pitch, yaw) for R = Rz(yaw) Ry(pitch) Rx(roll)."""
    pitch = -np.arcsin(np.clip(r[2, 0], -1, 1))
    roll = np.arctan2(r[2, 1], r[2, 2])
    yaw = np.arctan2(r[1, 0], r[0, 0])
    return np.array([roll, pitch, yaw])


def euler_zyx_to_rot(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return rz @ ry @ rx


def yaw_of(r: np.ndarray) -> float:
    return float(np.arctan2(r[1, 0], r[0, 0]))


def rot_z(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def wrap_angle(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


# --- JPL boundary (OpenVINS) -------------------------------------------------------------
# OpenVINS stores q_GtoI in JPL [x, y, z, w]; its rotation matrix R_GtoI maps global to IMU.
# The Hamilton quaternion of the IMU in the global frame has R_ItoG = R_GtoI^T, which for JPL
# means the same four numbers: q_hamilton(ItoG) = [w, x, y, z] of the JPL q_GtoI.


def jpl_to_hamilton(q_jpl_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = q_jpl_xyzw
    q = np.array([w, x, y, z])
    return q if q[0] >= 0 else -q


def hamilton_to_jpl(q_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = q_wxyz
    return np.array([x, y, z, w])
