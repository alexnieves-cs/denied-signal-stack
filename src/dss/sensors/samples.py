"""Sensor sample containers. A stream is a columnar table sharing one int64 ns time column."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from dss.core.manifest import sha256_array

# Merge priority for equal timestamps (lower first): IMU must precede the camera frame it brackets.
PRIORITY = {"imu": 0, "baro": 1, "mag": 2, "lidar": 3, "flow": 4, "gnss": 5, "cam": 6, "gt": 9}


@dataclass
class Stream:
    name: str
    kind: str
    t_ns: np.ndarray
    fields: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self):
        self.t_ns = np.asarray(self.t_ns, dtype=np.int64)
        for k, v in self.fields.items():
            if len(v) != len(self.t_ns):
                raise ValueError(f"{self.name}.{k}: length {len(v)} != {len(self.t_ns)}")

    def __len__(self) -> int:
        return len(self.t_ns)

    def __getitem__(self, k: str) -> np.ndarray:
        return self.fields[k]

    def slice(self, t0_ns: int, t1_ns: int) -> Stream:
        m = (self.t_ns >= t0_ns) & (self.t_ns < t1_ns)
        return Stream(self.name, self.kind, self.t_ns[m], {k: v[m] for k, v in self.fields.items()})

    def sha256(self) -> str:
        parts = [sha256_array(self.t_ns)] + [k + sha256_array(np.asarray(v)) for k, v in sorted(self.fields.items())]
        return sha256_array(np.frombuffer("".join(parts).encode(), dtype=np.uint8))

    @staticmethod
    def from_dict(name: str, kind: str, d: dict) -> Stream:
        return Stream(name, kind, d["t_ns"], {k: np.asarray(v) for k, v in d.items() if k != "t_ns"})
