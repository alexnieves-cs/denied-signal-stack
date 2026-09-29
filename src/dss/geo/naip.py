"""NAIP orthoimagery + 3DEP DEM via Planetary Computer STAC, cropped to the AOI local grid.

Tokens last ~45 min, so each crop is extracted once into data/geo/<aoi>/ as .npz.
"""

from __future__ import annotations

import json

import numpy as np

from dss.core.config import REPO_ROOT
from dss.geo.aoi import AOIS, Aoi, Raster

GEO = REPO_ROOT / "data" / "geo"
STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
NAIP_ITEMS = {  # Flagstaff NE quarter-quad; epochs recorded in docs (render 2023, map 2017)
    "flagstaff": {
        2023: "az_m_3511151_ne_12_030_20230615_20240328",
        2021: "az_m_3511151_ne_12_060_20211108",
        2019: "az_m_3511151_ne_12_060_20190930_20200109",
        2017: "az_m_3511151_ne_12_.6_20170627_20171128",
    }
}
DEM_ITEM = "n36w112-13"


def _catalog():
    import planetary_computer as pc
    import pystac_client

    return pystac_client.Client.open(STAC, modifier=pc.sign_inplace)


def _crop(href: str, aoi: Aoi, res: float, bands, resampling) -> np.ndarray:
    import rasterio
    from rasterio.warp import reproject

    xmin, ymin, xmax, ymax = aoi.utm_bounds()
    w = int(round((xmax - xmin) / res))
    h = int(round((ymax - ymin) / res))
    from rasterio.transform import from_origin

    dst_t = from_origin(xmin, ymax, res, res)
    with rasterio.open(href) as src:
        out = np.zeros((len(bands), h, w), dtype=np.float32)
        for k, b in enumerate(bands):
            reproject(rasterio.band(src, b), out[k], dst_transform=dst_t, dst_crs=aoi.crs, resampling=resampling)
    return out


def fetch(aoi_name: str, epochs=(2023,), res_img: float = 0.3, res_map: float = 1.0, dem_res: float = 5.0) -> dict:
    from rasterio.enums import Resampling

    aoi = AOIS[aoi_name]
    d = GEO / aoi_name
    d.mkdir(parents=True, exist_ok=True)
    cat = _catalog()
    info = {}
    for ep in epochs:
        f = d / f"naip_{ep}.npz"
        res = res_img if ep == 2023 else res_map
        if not f.exists():
            it = cat.get_collection("naip").get_item(NAIP_ITEMS[aoi_name][ep])
            a = _crop(it.assets["image"].href, aoi, res, (1, 2, 3, 4), Resampling.average)
            np.savez_compressed(f, rgbn=np.clip(a, 0, 255).astype(np.uint8).transpose(1, 2, 0), res=res,
                                xmin=aoi.xmin, ymax=aoi.ymax, item=it.id, date=str(it.datetime.date()))
        info[ep] = str(f)
    f = d / "dem.npz"
    if not f.exists():
        it = cat.get_collection("3dep-seamless").get_item(DEM_ITEM)
        z = _crop(it.assets["data"].href, aoi, dem_res, (1,), Resampling.bilinear)[0]
        np.savez_compressed(f, z=z - aoi.h0, res=dem_res, xmin=aoi.xmin, ymax=aoi.ymax, item=it.id, h0=aoi.h0)
    info["dem"] = str(f)
    (d / "fetch.json").write_text(json.dumps(info, indent=2) + "\n")
    return info


def load_image(aoi_name: str, epoch) -> tuple[Raster, dict]:
    """epoch: 2023 / 2021 / 2019 / 2017, or e.g. "2017_coreg" (co-registered to the render epoch, dss.geo.coreg)."""
    z = np.load(GEO / aoi_name / f"naip_{epoch}.npz")
    return Raster(z["rgbn"], float(z["xmin"]), float(z["ymax"]), float(z["res"])), {"item": str(z["item"]), "date": str(z["date"])}


def load_dem(aoi_name: str) -> Raster:
    z = np.load(GEO / aoi_name / "dem.npz")
    return Raster(z["z"], float(z["xmin"]), float(z["ymax"]), float(z["res"]))
