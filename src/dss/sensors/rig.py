"""Sensor rig: named sensors with extrinsics relative to the body (IMU) frame."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from dss.core.config import load
from dss.core.frames import SE3
from dss.core.rotations import euler_zyx_to_rot


@dataclass
class Rig:
    name: str
    extrinsics: dict[str, SE3] = field(default_factory=dict)  # T_body_sensor
    cameras: dict[str, dict] = field(default_factory=dict)

    def T(self, a: str, b: str) -> SE3:
        """T_a_b between any two rig members ('body' is the identity)."""
        ta = SE3() if a == "body" else self.extrinsics[a]
        tb = SE3() if b == "body" else self.extrinsics[b]
        return ta.inv() @ tb

    @staticmethod
    def from_config(path_or_cfg) -> Rig:
        c = load(path_or_cfg) if not isinstance(path_or_cfg, dict) else path_or_cfg
        ex = {}
        for name, e in c.get("extrinsics", {}).items():
            if "T" in e:
                ex[name] = SE3.from_matrix(np.asarray(e["T"], dtype=float))
            else:
                rpy = np.deg2rad(e.get("rpy_deg", [0, 0, 0]))
                ex[name] = SE3(euler_zyx_to_rot(*rpy), np.asarray(e.get("xyz", [0, 0, 0]), dtype=float))
        return Rig(c.get("name", "rig"), ex, c.get("cameras", {}))


# Nadir camera on an FLU body: camera RDF with image top toward travel (+x body).
# cam x (right) = -y_body, cam y (down in image) = -x_body, cam z (optical axis) = -z_body
R_BODY_NADIR_CAM = np.array([[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
