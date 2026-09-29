"""EuRoC ASL-format reader (EuRoC and TUM-VI 'exported/euroc'). Images are listed, not loaded."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from dss.sensors.samples import Stream


def _csv(path: Path) -> np.ndarray:
    return np.loadtxt(path, delimiter=",", comments="#", dtype=np.float64, ndmin=2)


def _csv_t(path: Path) -> tuple[np.ndarray, np.ndarray]:
    t = np.loadtxt(path, delimiter=",", comments="#", dtype=np.int64, usecols=0, ndmin=1)
    return t, _csv(path)[:, 1:]


class AslSequence:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.mav = self.root / "mav0" if (self.root / "mav0").exists() else self.root

    def imu(self) -> Stream:
        t, d = _csv_t(self.mav / "imu0" / "data.csv")
        return Stream("imu0", "imu", t, {"gyro": d[:, 0:3], "acc": d[:, 3:6]})

    def groundtruth(self) -> Stream:
        """EuRoC: state_groundtruth_estimate0 (p, q wxyz, v, bg, ba); TUM-VI: mocap0 (p, q wxyz)."""
        from dss.datasets.groundtruth import load_groundtruth

        return load_groundtruth(self.mav)

    def camera(self, cam: str = "cam0") -> tuple[np.ndarray, list[Path]]:
        p = self.mav / cam / "data.csv"
        t = np.loadtxt(p, delimiter=",", comments="#", dtype=np.int64, usecols=0, ndmin=1)
        names = np.loadtxt(p, delimiter=",", comments="#", dtype=str, usecols=1, ndmin=1)
        return t, [self.mav / cam / "data" / n.strip() for n in names]

    def sensor_yaml(self, name: str) -> Path:
        return self.mav / name / "sensor.yaml"
