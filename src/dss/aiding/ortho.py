"""Orthorectify a camera frame onto a local ground patch (flat-ground homography at the DEM height)."""

from __future__ import annotations

import cv2
import numpy as np


def ground_to_image_homography(K: np.ndarray, R_w_c: np.ndarray, p_w_c: np.ndarray, z_ground: float) -> np.ndarray:
    """H maps homogeneous ground (x, y, 1) on plane z=z_ground (world) to image pixels."""
    Rcw = R_w_c.T
    t = -Rcw @ p_w_c
    # X = [x, y, z_g]: x_c = Rcw[:,0] x + Rcw[:,1] y + (Rcw[:,2] z_g + t)
    M = np.stack([Rcw[:, 0], Rcw[:, 1], Rcw[:, 2] * z_ground + t], axis=1)
    return K @ M


def orthorectify(img: np.ndarray, K, R_w_c, p_w_c, z_ground: float, center_xy, size_m: float, res: float) -> tuple[np.ndarray, np.ndarray]:
    """Resample the image onto a north-up grid (row 0 = north) of size_m x size_m at res m/px.

    Returns (patch float32, valid mask).
    """
    n = int(round(size_m / res))
    cx, cy = center_xy
    x0 = cx - size_m / 2
    y1 = cy + size_m / 2
    # patch pixel (j, i) -> ground x = x0 + (j+0.5) res, y = y1 - (i+0.5) res
    A = np.array([[res, 0, x0 + 0.5 * res], [0, -res, y1 - 0.5 * res], [0, 0, 1.0]])
    H = ground_to_image_homography(K, R_w_c, p_w_c, z_ground) @ A
    patch = cv2.warpPerspective(img.astype(np.float32), H, (n, n), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=-1)
    valid = patch >= 0
    return patch, valid


def orthorectify_dem(img: np.ndarray, K, R_w_c, p_w_c, dem, center_xy, size_m: float, res: float):
    """Per-pixel orthorectification onto the DEM surface (no flat-ground assumption, so terrain relief does not
    bias the match). Grid is north-up, row 0 = north."""
    n = int(round(size_m / res))
    cx, cy = center_xy
    xs = cx - size_m / 2 + (np.arange(n) + 0.5) * res
    ys = cy + size_m / 2 - (np.arange(n) + 0.5) * res
    X, Y = np.meshgrid(xs, ys)
    Z = dem.sample(X, Y)
    P = np.stack([X, Y, Z], -1).reshape(-1, 3) - p_w_c
    xc = P @ R_w_c  # = (R_w_c^T P^T)^T
    front = xc[:, 2] > 0.1
    uv = (xc @ K.T)
    u = (uv[:, 0] / np.where(front, uv[:, 2], 1.0)).reshape(n, n).astype(np.float32)
    v = (uv[:, 1] / np.where(front, uv[:, 2], 1.0)).reshape(n, n).astype(np.float32)
    # pixel centres: pixel (i, j) covers [j, j+1); OpenCV remap samples integer coords at pixel centres
    patch = cv2.remap(img.astype(np.float32), u - 0.5, v - 0.5, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                      borderValue=-1)
    valid = (patch >= 0) & front.reshape(n, n)
    return patch, valid
