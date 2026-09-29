"""Calibration parsing: Kalibr camchain/imu YAML and EuRoC ``sensor.yaml``; extrinsic chains.

Convention: ``T_a_b`` maps points in b into a. Kalibr's ``T_cam_imu`` is T_c_i; OpenVINS
stores ``T_imu_cam`` = T_i_c = inv(T_c_i). EuRoC's ``T_BS`` is T_body_sensor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from dss.core.frames import SE3


@dataclass
class CameraCalib:
    name: str
    model: str
    intrinsics: np.ndarray  # fx fy cx cy
    distortion_model: str
    distortion: np.ndarray
    resolution: tuple[int, int]
    T_imu_cam: SE3
    timeshift_cam_imu: float = 0.0
    rate_hz: float | None = None

    def K(self) -> np.ndarray:
        fx, fy, cx, cy = self.intrinsics
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])


@dataclass
class ImuCalib:
    rate_hz: float
    gyroscope_noise_density: float
    gyroscope_random_walk: float
    accelerometer_noise_density: float
    accelerometer_random_walk: float
    T_body_imu: SE3 = field(default_factory=SE3)


def _load_yaml(path: Path) -> dict:
    txt = Path(path).read_text()
    # OpenCV-style %YAML:1.0 headers are not valid for PyYAML
    lines = [ln for ln in txt.splitlines() if not ln.startswith("%YAML")]
    txt = "\n".join(lines).replace("!!opencv-matrix", "")
    return yaml.safe_load(txt)


def _mat(v) -> np.ndarray:
    if isinstance(v, dict) and "data" in v:
        return np.asarray(v["data"], dtype=float).reshape(v["rows"], v["cols"])
    return np.asarray(v, dtype=float)


def load_kalibr_camchain(path: str | Path) -> list[CameraCalib]:
    d = _load_yaml(Path(path))
    cams = []
    for name in sorted(k for k in d if k.startswith("cam")):
        c = d[name]
        if "T_cam_imu" in c:
            t_ci = SE3.from_matrix(_mat(c["T_cam_imu"]))
            t_ic = t_ci.inv()
        else:
            t_ic = SE3.from_matrix(_mat(c["T_imu_cam"]))
        cams.append(
            CameraCalib(
                name=name,
                model=c.get("camera_model", "pinhole"),
                intrinsics=np.asarray(c["intrinsics"], dtype=float),
                distortion_model=c.get("distortion_model", "radtan"),
                distortion=np.asarray(c.get("distortion_coeffs", [0, 0, 0, 0]), dtype=float),
                resolution=tuple(c.get("resolution", [0, 0])),
                T_imu_cam=t_ic,
                timeshift_cam_imu=float(c.get("timeshift_cam_imu", 0.0)),
            )
        )
    return cams


def load_openvins_imucam(path: str | Path) -> list[CameraCalib]:
    """OpenVINS ``kalibr_imucam_chain.yaml`` (stores T_imu_cam)."""
    return load_kalibr_camchain(path)


def load_euroc_sensor_yaml(path: str | Path) -> dict:
    d = _load_yaml(Path(path))
    out = {"T_BS": SE3.from_matrix(_mat(d["T_BS"])), "rate_hz": float(d.get("rate_hz", 0))}
    for k in ("intrinsics", "distortion_coefficients", "resolution"):
        if k in d:
            out[k] = np.asarray(d[k], dtype=float)
    for k in ("gyroscope_noise_density", "gyroscope_random_walk", "accelerometer_noise_density", "accelerometer_random_walk"):
        if k in d:
            out[k] = float(d[k])
    return out


def euroc_T_imu_cam(imu_sensor_yaml: str | Path, cam_sensor_yaml: str | Path) -> SE3:
    """T_imu_cam = inv(T_B_imu) @ T_B_cam."""
    t_bi = load_euroc_sensor_yaml(imu_sensor_yaml)["T_BS"]
    t_bc = load_euroc_sensor_yaml(cam_sensor_yaml)["T_BS"]
    return t_bi.inv() @ t_bc


def compose(*ts: SE3) -> SE3:
    out = SE3()
    for t in ts:
        out = out @ t
    return out
