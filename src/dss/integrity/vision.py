"""Vision degradation detectors: feature-count collapse, motion blur, illumination shock."""

from __future__ import annotations

from collections import deque

import cv2
import numpy as np


def laplacian_var(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_32F).var())


class VisionMonitor:
    def __init__(self, n_max: int = 200, collapse_frac: float = 0.2, blur_ratio: float = 0.35, shock_frac: float = 0.35,
                 hold_frames: int = 20, median_win: int = 60):
        self.n_max = n_max
        self.collapse = collapse_frac * n_max
        self.blur_ratio = blur_ratio
        self.shock_frac = shock_frac
        self.hist = deque(maxlen=median_win)
        self.prev_mean = None
        self.good_run = 0
        self.hold = hold_frames
        self.state = "OK"  # OK | DEGRADED | FAILED
        self.flags = {"collapse": False, "blur": False, "shock": False, "dark": False}

    def step(self, gray_raw_mean: float, gray: np.ndarray, n_tracked: int | None) -> dict:
        lv = laplacian_var(gray)
        med = float(np.median(self.hist)) if len(self.hist) >= 5 else lv
        blur = lv < self.blur_ratio * med
        if not blur:
            self.hist.append(lv)
        shock = self.prev_mean is not None and abs(gray_raw_mean - self.prev_mean) > self.shock_frac * max(self.prev_mean, 1.0)
        self.prev_mean = gray_raw_mean
        collapse = n_tracked is not None and n_tracked < self.collapse
        dark = gray_raw_mean < 8.0  # below this even max AE gain leaves a noise-dominated image
        self.flags = {"collapse": bool(collapse), "blur": bool(blur), "shock": bool(shock), "dark": bool(dark)}
        bad = collapse or dark
        if bad:
            self.state = "FAILED"
            self.good_run = 0
        else:
            self.good_run += 1
            if self.state == "FAILED" and self.good_run >= self.hold:
                self.state = "OK"
            elif self.state != "FAILED":
                self.state = "DEGRADED" if (blur or shock) else "OK"
        return {"state": self.state, "lapvar": lv, **self.flags}
