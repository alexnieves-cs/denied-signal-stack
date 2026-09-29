"""Ground tiles: a textured DEM mesh over the AOI (render epoch imagery)."""

from __future__ import annotations

import numpy as np

from dss.geo.aoi import Raster


def ground_mesh(dem: Raster, xmin, xmax, ymin, ymax, step: float = 5.0):
    xs = np.arange(xmin, xmax + 1e-9, step)
    ys = np.arange(ymin, ymax + 1e-9, step)
    X, Y = np.meshgrid(xs, ys)
    Z = dem.sample(X, Y)
    verts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
    u = (X.ravel() - xmin) / (xmax - xmin)
    v = (Y.ravel() - ymin) / (ymax - ymin)
    uv = np.stack([u, v], 1)
    nx, ny = len(xs), len(ys)
    idx = np.arange(nx * ny).reshape(ny, nx)
    a, b, c, d = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel()
    faces = np.concatenate([np.stack([a, b, d], 1), np.stack([a, d, c], 1)])  # CCW seen from +z
    return verts, uv, faces
