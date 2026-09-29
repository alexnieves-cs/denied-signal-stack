"""VioBackend interface shared by the ovserver client and the synthetic generator."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from dss.vio.protocol import VioState


class VioBackend(Protocol):
    def init(self) -> None: ...

    def imu(self, t_s: float, w: np.ndarray, a: np.ndarray) -> None: ...

    def cam(self, t_s: float, gray: np.ndarray, cam_id: int = 0) -> VioState: ...

    def init_gt(self, t_s: float, q_wxyz: np.ndarray, p: np.ndarray, v: np.ndarray,
                bg: np.ndarray, ba: np.ndarray) -> None: ...

    def reanchor(self, t_s: float, q_wxyz: np.ndarray, p: np.ndarray, v: np.ndarray,
                 bg: np.ndarray, ba: np.ndarray, P15: np.ndarray) -> None: ...

    def close(self) -> None: ...
