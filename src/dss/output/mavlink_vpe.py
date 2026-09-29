"""MAVLink 2 VISION_POSITION_ESTIMATE encoder (NED, 30 Hz, explicit 21-float covariance)."""

from __future__ import annotations

import os

os.environ.setdefault("MAVLINK20", "1")

import numpy as np  # noqa: E402
from pymavlink.dialects.v20 import common as mav  # noqa: E402

from dss.core.frames import R_ENU_NED, R_FLU_FRD  # noqa: E402
from dss.core.rotations import rot_to_euler_zyx  # noqa: E402


class _Buf:
    def __init__(self):
        self.data = []

    def write(self, b):
        self.data.append(bytes(b))


class VpeEncoder:
    def __init__(self, sysid: int = 1, compid: int = 197):  # MAV_COMP_ID_VISUAL_INERTIAL_ODOMETRY
        self.buf = _Buf()
        self.m = mav.MAVLink(self.buf, srcSystem=sysid, srcComponent=compid)
        self.m.WIRE_PROTOCOL_VERSION = "2.0"

    def encode(self, t_us: int, p_enu, R_enu_flu, P_pos_enu, att_var_rad2: np.ndarray, reset_counter: int) -> bytes:
        p = R_ENU_NED @ np.asarray(p_enu)
        R = R_ENU_NED @ R_enu_flu @ R_FLU_FRD
        roll, pitch, yaw = rot_to_euler_zyx(R)
        Pn = R_ENU_NED @ P_pos_enu @ R_ENU_NED.T
        # conservative diagonal: each axis variance >= the largest eigen-direction projected
        C = np.zeros((6, 6))
        C[:3, :3] = np.diag(np.diag(Pn)) * 1.0 + np.eye(3) * 1e-6
        lam = float(np.max(np.linalg.eigvalsh(Pn[:2, :2])))
        C[0, 0] = C[1, 1] = max(C[0, 0], C[1, 1], lam)
        C[3:, 3:] = np.diag(att_var_rad2)
        cov = [float(C[i, j]) for i in range(6) for j in range(i, 6)]
        msg = self.m.vision_position_estimate_encode(int(t_us), float(p[0]), float(p[1]), float(p[2]), float(roll), float(pitch),
                                                     float(yaw), cov, int(reset_counter) & 0xFF)
        return msg.pack(self.m)
