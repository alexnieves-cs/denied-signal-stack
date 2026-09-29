"""Epoch co-registration check: tile-wise phase correlation between a map epoch and the render epoch."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from dss.aiding.match_ncc import map_gray
from dss.geo import naip


def coregister(aoi: str = "flagstaff", ref_epoch: int = 2023, map_epoch: int = 2021, tile_m: float = 200.0) -> dict:
    img, info_r = naip.load_image(aoi, ref_epoch)
    mp, info_m = naip.load_image(aoi, map_epoch)
    r = mp.res
    g = map_gray(img)
    g = cv2.resize(g, (int(g.shape[1] * img.res / r), int(g.shape[0] * img.res / r)), interpolation=cv2.INTER_AREA)
    m = map_gray(mp)
    h = min(g.shape[0], m.shape[0])
    w = min(g.shape[1], m.shape[1])
    n = int(tile_m / r)
    win = cv2.createHanningWindow((n, n), cv2.CV_32F)
    shifts = []
    for i in range(0, h - n, n):
        for j in range(0, w - n, n):
            a = g[i:i + n, j:j + n].astype(np.float32)
            b = m[i:i + n, j:j + n].astype(np.float32)
            a = a - cv2.GaussianBlur(a, (0, 0), 2)
            b = b - cv2.GaussianBlur(b, (0, 0), 2)
            (dx, dy), resp = cv2.phaseCorrelate(a, b, win)
            if resp > 0.05:
                shifts.append((dx * r, -dy * r, resp))
    s = np.array(shifts)
    strong = s[s[:, 2] >= 0.15]
    out_strong = {"tiles_strong": int(len(strong))}
    if len(strong):
        ms = np.median(strong[:, :2], axis=0)
        out_strong.update({"strong_median_shift_m": ms.tolist(),
                           "strong_rms_residual_m": float(np.sqrt(np.mean(np.sum((strong[:, :2] - ms) ** 2, 1))))})
    med = np.median(s[:, :2], axis=0)
    resid = s[:, :2] - med
    out = {"ref_epoch": info_r, "map_epoch": info_m, "tiles": int(len(s)), "tile_m": tile_m,
           "median_shift_m": med.tolist(), "rms_shift_m": float(np.sqrt(np.mean(np.sum(s[:, :2] ** 2, 1)))),
           "rms_residual_after_median_m": float(np.sqrt(np.mean(np.sum(resid**2, 1)))),
           "p90_abs_shift_m": float(np.percentile(np.linalg.norm(s[:, :2], axis=1), 90)), **out_strong}
    return out


def displacement_field(aoi: str = "flagstaff", ref_epoch: int = 2023, map_epoch: int = 2021, tile_m: float = 100.0,
                       step_m: float = 50.0, min_resp: float = 0.08):
    """Dense map->reference displacement field (dx, dy in metres on the map grid): overlapping tiles,
    phase correlation, outlier-robust median filtering and smoothing."""
    from scipy.ndimage import gaussian_filter, median_filter

    img, _ = naip.load_image(aoi, ref_epoch)
    mp, _ = naip.load_image(aoi, map_epoch)
    r = mp.res
    g = map_gray(img)
    g = cv2.resize(g, (int(g.shape[1] * img.res / r), int(g.shape[0] * img.res / r)), interpolation=cv2.INTER_AREA)
    m = map_gray(mp)
    h, w = min(g.shape[0], m.shape[0]), min(g.shape[1], m.shape[1])
    n, st = int(tile_m / r), int(step_m / r)
    win = cv2.createHanningWindow((n, n), cv2.CV_32F)
    ci = list(range(0, h - n, st))
    cj = list(range(0, w - n, st))
    D = np.full((len(ci), len(cj), 2), np.nan)
    for a, i in enumerate(ci):
        for b, j in enumerate(cj):
            A = g[i:i + n, j:j + n].astype(np.float32)
            B = m[i:i + n, j:j + n].astype(np.float32)
            A = A - cv2.GaussianBlur(A, (0, 0), 2)
            B = B - cv2.GaussianBlur(B, (0, 0), 2)
            (dx, dy), resp = cv2.phaseCorrelate(A, B, win)
            if resp >= min_resp and abs(dx) < 25 and abs(dy) < 25:
                D[a, b] = (dx, dy)  # pixels: map content is displaced by (dx, dy) relative to the reference
    for k in range(2):
        f = D[..., k]
        med = np.nanmedian(f)
        f = np.where(np.isnan(f), med, f)
        f = median_filter(f, size=5)
        D[..., k] = gaussian_filter(f, 1.0)
    centres_i = np.array(ci) + n / 2
    centres_j = np.array(cj) + n / 2
    return D, centres_i, centres_j, (h, w)


def displacement_field_ncc(aoi: str = "flagstaff", ref_epoch: int = 2023, map_epoch: int = 2021, step_m: float = 40.0,
                           patch_m: float = 96.0, search_m: float = 25.0, min_ratio: float = 1.5):
    """Dense patch-NCC displacement field (reference patch located in the map), robustly smoothed.
    Returns D[a, b] = (dx, dy) world metres such that map(x + dx, y + dy) ~ reference(x, y)."""
    from scipy.ndimage import gaussian_filter, generic_filter

    from dss.aiding.match_ncc import NccMatcher
    from dss.geo.aoi import Raster

    img, _ = naip.load_image(aoi, ref_epoch)
    mp, _ = naip.load_image(aoi, map_epoch)
    d = img.data[:, :, :3].astype(np.float32)
    f = img.res / mp.res
    sm = cv2.resize(d, (int(d.shape[1] * f), int(d.shape[0] * f)), interpolation=cv2.INTER_AREA)
    ref = NccMatcher(Raster(sm.astype(np.uint8), img.xmin, img.ymax, mp.res))
    M = NccMatcher(mp, min_ratio=min_ratio)
    from dss.geo.aoi import AOIS

    a = AOIS[aoi]
    xs = np.arange(a.xmin + 150, a.xmax - 150, step_m)
    ys = np.arange(a.ymax - 150, a.ymin + 150, -step_m)
    D = np.full((len(ys), len(xs), 2), np.nan)
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            patch, _, _ = ref.crop(x, y, patch_m / 2)
            if patch is None or patch.std() < 3:
                continue
            r = M.match(patch, np.ones_like(patch, bool), np.array([x, y]), search_m)
            if r.ok:
                D[i, j] = r.xy - np.array([x, y])
    ok = np.isfinite(D[..., 0])
    out = np.empty_like(D)
    for k in range(2):
        f_ = D[..., k]
        med = generic_filter(f_, np.nanmedian, size=7, mode="nearest")
        med = np.where(np.isfinite(med), med, np.nanmedian(f_))
        out[..., k] = gaussian_filter(med, 1.0)
    return out, xs, ys, float(ok.mean())


def coregistered_map_ncc(aoi: str = "flagstaff", map_epoch: int = 2021, ref_epoch: int = 2023):
    from scipy.interpolate import RegularGridInterpolator

    from dss.geo.aoi import Raster

    D, xs, ys, frac = displacement_field_ncc(aoi, ref_epoch, map_epoch)
    mp, info = naip.load_image(aoi, map_epoch)
    h, w = mp.data.shape[:2]
    ii, jj = np.mgrid[0:h, 0:w].astype(np.float32)
    X = mp.xmin + (jj + 0.5) * mp.res
    Y = mp.ymax - (ii + 0.5) * mp.res
    pts = np.stack([Y.ravel(), X.ravel()], 1)
    fx = RegularGridInterpolator((ys[::-1], xs), D[::-1, :, 0], bounds_error=False, fill_value=None)
    fy = RegularGridInterpolator((ys[::-1], xs), D[::-1, :, 1], bounds_error=False, fill_value=None)
    dx = fx(pts).reshape(h, w)
    dy = fy(pts).reshape(h, w)
    mapx = (jj + dx / mp.res).astype(np.float32)
    mapy = (ii - dy / mp.res).astype(np.float32)
    out = np.stack([cv2.remap(mp.data[..., c], mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                    for c in range(mp.data.shape[2])], -1)
    return Raster(out, mp.xmin, mp.ymax, mp.res), info, D, frac


def coregistered_map(aoi: str = "flagstaff", map_epoch: int = 2021, ref_epoch: int = 2023):
    """Warp the map epoch onto the reference epoch with the smooth displacement field (PLAN 2.1)."""
    from scipy.interpolate import RegularGridInterpolator

    from dss.geo.aoi import Raster

    D, ci, cj, (h, w) = displacement_field(aoi, ref_epoch, map_epoch)
    mp, info = naip.load_image(aoi, map_epoch)
    ii, jj = np.mgrid[0:mp.data.shape[0], 0:mp.data.shape[1]].astype(np.float32)
    fx = RegularGridInterpolator((ci, cj), D[..., 0], bounds_error=False, fill_value=None)
    fy = RegularGridInterpolator((ci, cj), D[..., 1], bounds_error=False, fill_value=None)
    pts = np.stack([ii.ravel(), jj.ravel()], 1)
    mapx = (jj.ravel() + fx(pts)).reshape(ii.shape).astype(np.float32)
    mapy = (ii.ravel() + fy(pts)).reshape(ii.shape).astype(np.float32)
    out = np.stack([cv2.remap(mp.data[..., c], mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                    for c in range(mp.data.shape[2])], -1)
    return Raster(out, mp.xmin, mp.ymax, mp.res), info, D


if __name__ == "__main__":
    res = {e: coregister(map_epoch=e) for e in (2021, 2017)}
    Path("reports/phase2/coregistration.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))
