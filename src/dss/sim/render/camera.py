"""Pinhole camera rendering through MuJoCo. Returns grayscale uint8 (+ optional depth)."""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "cgl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from dss.core.rotations import rot_to_quat  # noqa: E402
from dss.sim.render.scene import R_RDF_MJ  # noqa: E402


class PinholeRenderer:
    def __init__(self, model: mujoco.MjModel, width: int, height: int, fy: float):
        self.m = model
        self.d = mujoco.MjData(model)
        self.w, self.h, self.f = width, height, fy
        self.r = mujoco.Renderer(model, height, width)
        self.cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "cam0")
        # MuJoCo's rendered focal length follows fovy exactly for the vertical axis; square pixels
        self.K = np.array([[fy, 0, width / 2.0], [0, fy, height / 2.0], [0, 0, 1.0]])

    def render(self, p_w_cam: np.ndarray, R_w_cam_rdf: np.ndarray, depth: bool = False):
        R_mj = R_w_cam_rdf @ R_RDF_MJ
        self.d.mocap_pos[0] = p_w_cam
        self.d.mocap_quat[0] = rot_to_quat(R_mj)
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_camlight(self.m, self.d)
        self.r.update_scene(self.d, camera=self.cam_id)
        rgb = self.r.render()
        out = {"rgb": rgb}
        if depth:
            self.r.enable_depth_rendering()
            self.r.update_scene(self.d, camera=self.cam_id)
            out["depth"] = self.r.render().copy()
            self.r.disable_depth_rendering()
        return out

    def close(self):
        self.r.close()


def to_gray(rgb: np.ndarray) -> np.ndarray:
    return (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.float32)
