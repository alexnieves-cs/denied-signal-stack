"""Frames (ADR-0002): ENU world at the AOI origin, FLU body, RDF camera.

NED/FRD appears only at the MAVLink boundary.
"""

from __future__ import annotations

import numpy as np

# ENU <-> NED: x_ned = y_enu, y_ned = x_enu, z_ned = -z_enu (an involution)
R_ENU_NED = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
# FLU <-> FRD: flip y and z
R_FLU_FRD = np.diag([1.0, -1.0, -1.0])


def enu_to_ned(v: np.ndarray) -> np.ndarray:
    return np.asarray(v) @ R_ENU_NED.T


def ned_to_enu(v: np.ndarray) -> np.ndarray:
    return np.asarray(v) @ R_ENU_NED.T


def rot_flu_enu_to_frd_ned(r_enu_flu: np.ndarray) -> np.ndarray:
    """Body attitude R_world_body from (ENU, FLU) to (NED, FRD)."""
    return R_ENU_NED @ r_enu_flu @ R_FLU_FRD


def cov_enu_to_ned(p: np.ndarray) -> np.ndarray:
    return R_ENU_NED @ p @ R_ENU_NED.T


class SE3:
    """Rigid transform T_a_b: maps points in frame b into frame a."""

    __slots__ = ("r", "t")

    def __init__(self, r: np.ndarray | None = None, t: np.ndarray | None = None):
        self.r = np.eye(3) if r is None else np.asarray(r, dtype=float)
        self.t = np.zeros(3) if t is None else np.asarray(t, dtype=float)

    @staticmethod
    def from_matrix(m: np.ndarray) -> SE3:
        return SE3(m[:3, :3], m[:3, 3])

    def matrix(self) -> np.ndarray:
        m = np.eye(4)
        m[:3, :3] = self.r
        m[:3, 3] = self.t
        return m

    def __matmul__(self, other: SE3) -> SE3:
        return SE3(self.r @ other.r, self.r @ other.t + self.t)

    def inv(self) -> SE3:
        return SE3(self.r.T, -self.r.T @ self.t)

    def apply(self, p: np.ndarray) -> np.ndarray:
        return np.asarray(p) @ self.r.T + self.t

    def __repr__(self) -> str:  # pragma: no cover
        return f"SE3(t={self.t})"
