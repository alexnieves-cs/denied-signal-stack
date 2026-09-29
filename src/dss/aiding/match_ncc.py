"""Tier-A matcher: normalized cross-correlation of an orthorectified patch against the map epoch."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from dss.geo.aoi import Raster


@dataclass
class MatchResult:
    ok: bool
    xy: np.ndarray  # matched position of the patch centre (world)
    cov: np.ndarray  # 2x2
    peak: float
    peak_ratio: float
    reason: str = ""


def _norm(a: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    a = a.astype(np.float32)
    if valid is not None:
        m = a[valid].mean() if valid.any() else 0.0
        a = np.where(valid, a, m)
    # high-pass (removes illumination / seasonal low frequencies), then standardize
    a = a - cv2.GaussianBlur(a, (0, 0), 2.0)
    return (a - a.mean()) / (a.std() + 1e-6)


def map_gray(mp: Raster) -> np.ndarray:
    d = mp.data[..., :3].astype(np.float32)
    return 0.299 * d[..., 0] + 0.587 * d[..., 1] + 0.114 * d[..., 2]


class NccMatcher:
    def __init__(self, mp: Raster, res: float | None = None, reg_floor_m: float = 1.5, min_peak: float = 0.10,
                 min_ratio: float = 1.3):
        self.mp = mp
        self.gray = map_gray(mp)
        self.res = res or mp.res
        self.reg_floor = reg_floor_m
        self.min_peak = min_peak
        self.min_ratio = min_ratio

    def crop(self, cx: float, cy: float, half_m: float) -> tuple[np.ndarray, float, float]:
        r = self.mp.res
        i0, j0 = self.mp.xy_to_ij(cx - half_m, cy + half_m)
        i0, j0 = int(round(i0 + 0.5)), int(round(j0 + 0.5))
        n = int(round(2 * half_m / r))
        h, w = self.gray.shape
        if i0 < 0 or j0 < 0 or i0 + n > h or j0 + n > w:
            return None, 0, 0
        # top-left ground coordinate of the crop
        return self.gray[i0:i0 + n, j0:j0 + n], self.mp.xmin + j0 * r, self.mp.ymax - i0 * r

    def match(self, patch: np.ndarray, valid: np.ndarray, center_xy, search_m: float) -> MatchResult:
        r = self.mp.res
        size_m = patch.shape[0] * r
        crop, cx0, cy0 = self.crop(center_xy[0], center_xy[1], size_m / 2 + search_m)
        if crop is None:
            return MatchResult(False, np.zeros(2), np.eye(2), 0, 0, "outside map")
        if valid.mean() < 0.9:
            return MatchResult(False, np.zeros(2), np.eye(2), 0, 0, "patch invalid")
        tpl = _norm(patch, valid)
        if patch[valid].std() < 3.0:
            return MatchResult(False, np.zeros(2), np.eye(2), 0, 0, "no texture")
        img = _norm(crop)
        res = cv2.matchTemplate(img, tpl, cv2.TM_CCOEFF_NORMED)
        _, peak, _, loc = cv2.minMaxLoc(res)
        j, i = loc
        # second peak outside an exclusion radius
        ex = res.copy()
        rad = max(3, int(round(8.0 / r)))
        ex[max(0, i - rad):i + rad + 1, max(0, j - rad):j + rad + 1] = -1
        second = float(ex.max())
        ratio = peak / max(second, 1e-3)
        # sub-pixel refinement + curvature
        di = dj = 0.0
        cii = cjj = 1e-3
        if 0 < i < res.shape[0] - 1 and 0 < j < res.shape[1] - 1:
            cii = res[i - 1, j] - 2 * res[i, j] + res[i + 1, j]
            cjj = res[i, j - 1] - 2 * res[i, j] + res[i, j + 1]
            di = 0.5 * (res[i - 1, j] - res[i + 1, j]) / cii if cii < 0 else 0.0
            dj = 0.5 * (res[i, j - 1] - res[i, j + 1]) / cjj if cjj < 0 else 0.0
        # matched patch top-left in the crop -> patch centre in world
        x = cx0 + (j + dj) * r + size_m / 2
        y = cy0 - (i + di) * r - size_m / 2
        # covariance from peak curvature (in px^2), scaled by the correlation noise, floored
        n_eff = max(valid.sum() / 25.0, 1.0)
        s_i = np.sqrt(max((1 - peak) / (n_eff * max(-cii, 1e-4)), 0.0)) * r
        s_j = np.sqrt(max((1 - peak) / (n_eff * max(-cjj, 1e-4)), 0.0)) * r
        sx = np.hypot(s_j, self.reg_floor)
        sy = np.hypot(s_i, self.reg_floor)
        cov = np.diag([sx**2, sy**2])
        ok = peak >= self.min_peak and ratio >= self.min_ratio
        return MatchResult(bool(ok), np.array([x, y]), cov, float(peak), float(ratio),
                           "" if ok else f"peak {peak:.2f} ratio {ratio:.2f}")
