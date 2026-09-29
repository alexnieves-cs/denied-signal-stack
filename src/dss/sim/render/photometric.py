"""Photometric camera model applied in numpy after the albedo render.

Chain: illumination scale (events: forest canopy darkness, twilight, shocks) -> vignetting ->
Koschmieder haze -> motion blur (exposure x image-plane speed) -> shot + read noise ->
auto-exposure gain -> 8-bit quantization.
"""

from __future__ import annotations

import cv2
import numpy as np


def illumination_at(t: float, events: list[dict]) -> float:
    s = 1.0
    for e in events or []:
        if e["t0"] <= t < e["t1"]:
            ramp = float(e.get("ramp_s", 0.0))
            if ramp > 0:
                a = min(1.0, (t - e["t0"]) / ramp, (e["t1"] - t) / ramp)
            else:
                a = 1.0
            s *= 1.0 + (float(e["scale"]) - 1.0) * a
    return s


def vignette(h: int, w: int, strength: float = 0.25) -> np.ndarray:
    y, x = np.mgrid[0:h, 0:w]
    r2 = ((x - w / 2) ** 2 + (y - h / 2) ** 2) / ((w / 2) ** 2 + (h / 2) ** 2)
    return 1.0 - strength * r2


class Photometric:
    def __init__(self, width: int, height: int, cfg: dict | None = None):
        c = cfg or {}
        self.exposure_s = float(c.get("exposure_s", 0.004))
        self.full_well = float(c.get("full_well_e", 6000.0))
        self.read_noise = float(c.get("read_noise_e", 6.0))
        self.haze_beta = float(c.get("haze_beta_per_m", 1.0e-4))
        self.airlight = float(c.get("airlight", 200.0))
        self.vig = vignette(height, width, float(c.get("vignetting", 0.2)))
        self.auto_exposure = bool(c.get("auto_exposure", True))
        self.ae_max_gain = float(c.get("ae_max_gain", 8.0))
        self.target_mean = float(c.get("target_mean", 110.0))
        self.gain = 1.0

    def blur_px(self, omega_cam: np.ndarray, v_cam: np.ndarray, depth: float, f: float) -> np.ndarray:
        """Image-plane displacement (px) at the image centre during the exposure."""
        vx = f * (v_cam[0] / max(depth, 0.1)) + f * omega_cam[1]
        vy = f * (v_cam[1] / max(depth, 0.1)) - f * omega_cam[0]
        return np.array([vx, vy]) * self.exposure_s

    def apply(self, gray_albedo: np.ndarray, t: float, events: list[dict], g: np.random.Generator,
              depth: float = 100.0, blur_vec_px: np.ndarray | None = None, extra_blur_px: float = 0.0) -> np.ndarray:
        img = gray_albedo.astype(np.float32) * illumination_at(t, events) * self.vig
        tr = np.exp(-self.haze_beta * depth)
        img = img * tr + self.airlight * (1 - tr)
        bl = np.zeros(2) if blur_vec_px is None else np.asarray(blur_vec_px, float)
        if extra_blur_px > 0:
            bl = bl + np.array([extra_blur_px, 0.0])
        n = float(np.hypot(*bl))
        if n >= 1.0:
            k = int(np.ceil(n)) | 1
            ker = np.zeros((k, k), np.float32)
            c = k // 2
            for s in np.linspace(-0.5, 0.5, 4 * k):
                ker[int(round(c + s * bl[1])), int(round(c + s * bl[0]))] += 1
            img = cv2.filter2D(img, -1, ker / ker.sum(), borderType=cv2.BORDER_REFLECT)
        # sensor: electrons = albedo/255 * full_well * exposure factor
        e = np.clip(img, 0, None) / 255.0 * self.full_well
        e = g.poisson(e).astype(np.float32) + g.normal(0, self.read_noise, e.shape).astype(np.float32)
        dn = e / self.full_well * 255.0
        if self.auto_exposure:
            m = float(np.mean(dn)) + 1e-3
            want = np.clip(self.target_mean / m, 1.0 / self.ae_max_gain, self.ae_max_gain)
            self.gain = 0.7 * self.gain + 0.3 * want  # AE loop lag
        return np.clip(dn * self.gain, 0, 255).astype(np.uint8)
