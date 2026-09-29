"""`dss doctor`: environment gate (Python, HTTPS, disk, data budget) + advisory --full probes."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from dss.core.config import REPO_ROOT

BUDGET_GB = {"data": 26.0, "runs": 6.0}
FREE_FAIL_GB, FREE_WARN_GB = 15.0, 25.0


def _du_gb(p: Path) -> float:
    if not p.exists():
        return 0.0
    out = subprocess.run(["du", "-sk", str(p)], capture_output=True, text=True).stdout.split()
    return int(out[0]) / 1024 / 1024 if out else 0.0


def checks(full: bool = False) -> list[tuple[str, bool, str, bool]]:
    """Return (name, ok, detail, gating)."""
    res = []
    v = platform.python_version()
    res.append(("python 3.13.15", v == "3.13.15", v, True))
    try:
        import urllib.request

        code = urllib.request.urlopen("https://pypi.org/simple/", timeout=10).status
        res.append(("https", code == 200, str(code), True))
    except Exception as e:  # pragma: no cover - network dependent
        res.append(("https", False, repr(e)[:80], True))
    free = shutil.disk_usage(REPO_ROOT).free / 1e9
    res.append(("free disk >= 15 GB", free >= FREE_FAIL_GB, f"{free:.1f} GB" + (" (WARN <25)" if free < FREE_WARN_GB else ""), True))
    for d, b in BUDGET_GB.items():
        used = _du_gb(REPO_ROOT / d)
        res.append((f"ledger {d}/ <= {b} GB", used <= b, f"{used:.2f} GB", True))
    if full:
        for mod in ("numpy", "scipy", "gtsam", "mujoco", "cv2", "rerun", "pymavlink", "rasterio", "pyproj"):
            try:
                m = __import__(mod)
                res.append((f"import {mod}", True, getattr(m, "__version__", "?"), False))
            except Exception as e:
                res.append((f"import {mod}", False, repr(e)[:80], False))
        try:
            import gtsam

            gtsam.IncrementalFixedLagSmoother(10.0)
            res.append(("IncrementalFixedLagSmoother", True, "constructs", False))
        except Exception as e:
            res.append(("IncrementalFixedLagSmoother", False, repr(e)[:80], False))
        try:
            import cv2
            import numpy as np

            a = np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8)
            cv2.resize(a, (333, 251), interpolation=cv2.INTER_LINEAR)
            res.append(("cv2 CV_8UC3 INTER_LINEAR resize", True, cv2.__version__, False))
        except Exception as e:
            res.append(("cv2 resize", False, repr(e)[:80], False))
        try:
            import pyproj
            import rasterio

            res.append(("rasterio PROJ == pyproj PROJ", True, f"{pyproj.proj_version_str} / gdal {rasterio.__gdal_version__}", False))
        except Exception as e:
            res.append(("rasterio/pyproj", False, repr(e)[:80], False))
        try:
            os.environ.setdefault("MUJOCO_GL", "cgl")
            import mujoco

            m = mujoco.MjModel.from_xml_string("<mujoco><worldbody><geom size='1'/></worldbody></mujoco>")
            r = mujoco.Renderer(m, 64, 64)
            d = mujoco.MjData(m)
            mujoco.mj_forward(m, d)
            r.update_scene(d)
            r.render()
            res.append(("MuJoCo CGL frame", True, mujoco.__version__, False))
        except Exception as e:
            res.append(("MuJoCo CGL frame", False, repr(e)[:80], False))
        ov = Path.home() / ".cache/dss/cpp/build/ovserver/ovserver"
        res.append(("ovserver binary", ov.exists(), str(ov), False))
    return res


def main(full: bool = False) -> int:
    rc = 0
    for name, ok, detail, gating in checks(full):
        tag = "OK  " if ok else ("FAIL" if gating else "warn")
        print(f"[{tag}] {name:40s} {detail}")
        if gating and not ok:
            rc = 1
    print("doctor:", "PASS" if rc == 0 else "FAIL", f"({sys.executable})")
    return rc
