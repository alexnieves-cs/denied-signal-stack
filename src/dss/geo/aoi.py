"""Area of interest: a local ENU frame on the NAD83 / UTM 12N grid (EPSG:26912).

Local x = easting - E0, y = northing - N0, z = orthometric height - H0 (NAVD88 via 3DEP).
Grid convergence (~0.3 deg at Flagstaff) is ignored: 'north' means grid north.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Aoi:
    name: str
    crs: str
    e0: float
    n0: float
    h0: float
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    lat_deg: float
    lon_deg: float

    def utm_bounds(self, margin: float = 0.0):
        return (self.e0 + self.xmin - margin, self.n0 + self.ymin - margin,
                self.e0 + self.xmax + margin, self.n0 + self.ymax + margin)


FLAGSTAFF = Aoi(
    name="flagstaff", crs="EPSG:26912", e0=440600.0, n0=3895800.0, h0=2110.0,
    xmin=-2400.0, xmax=400.0, ymin=-500.0, ymax=500.0, lat_deg=35.1985, lon_deg=-111.6585,
)

AOIS = {"flagstaff": FLAGSTAFF}


class Raster:
    """Georeferenced local raster: value at (x, y) via bilinear interpolation. Row 0 is ymax."""

    def __init__(self, data: np.ndarray, xmin: float, ymax: float, res: float):
        self.data = data
        self.xmin, self.ymax, self.res = float(xmin), float(ymax), float(res)

    @property
    def shape(self):
        return self.data.shape

    def xy_to_ij(self, x, y):
        j = (np.asarray(x) - self.xmin) / self.res - 0.5
        i = (self.ymax - np.asarray(y)) / self.res - 0.5
        return i, j

    def sample(self, x, y):
        i, j = self.xy_to_ij(x, y)
        h, w = self.data.shape[:2]
        i = np.clip(i, 0, h - 1.000001)
        j = np.clip(j, 0, w - 1.000001)
        i0 = np.floor(i).astype(int)
        j0 = np.floor(j).astype(int)
        di, dj = i - i0, j - j0
        d = self.data.astype(np.float64)
        if d.ndim == 3:
            di, dj = di[..., None], dj[..., None]
        return ((1 - di) * (1 - dj) * d[i0, j0] + (1 - di) * dj * d[i0, j0 + 1]
                + di * (1 - dj) * d[i0 + 1, j0] + di * dj * d[i0 + 1, j0 + 1])
