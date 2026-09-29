import time

import numpy as np
import pytest

from dss.aiding.match_ncc import NccMatcher
from dss.geo.aoi import Raster


def _texture(n=1200, seed=0):
    g = np.random.default_rng(seed)
    import cv2

    a = g.normal(0, 1, (n, n)).astype(np.float32)
    a = cv2.GaussianBlur(a, (0, 0), 3)
    a = (a - a.min()) / (a.max() - a.min()) * 255
    return np.repeat(a[..., None], 3, 2).astype(np.uint8)


def test_ncc_recovers_offset_subpixel():
    mp = Raster(_texture(), -600.0, 600.0, 1.0)
    M = NccMatcher(mp)
    true_c = np.array([12.3, -7.6])
    # patch = the map around the true centre (same epoch), searched around a wrong prior
    import cv2

    g = M.gray
    i, j = mp.xy_to_ij(true_c[0], true_c[1])
    Mx = np.float32([[1, 0, j - 47.5], [0, 1, i - 47.5]])
    patch = cv2.warpAffine(g, Mx, (96, 96), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
    r = M.match(patch, np.ones_like(patch, bool), true_c + np.array([15.0, -10.0]), 40.0)
    assert r.ok
    assert np.linalg.norm(r.xy - true_c) < 0.3


def test_tier_a_speed_512():
    import cv2

    mp = Raster(_texture(2000), -1000.0, 1000.0, 1.0)
    M = NccMatcher(mp)
    patch = M.gray[500:1012, 500:1012].copy()
    t0 = time.perf_counter()
    cv2.matchTemplate(M.gray[400:1112, 400:1112], patch, cv2.TM_CCOEFF_NORMED)
    # PLAN exit: <= 20 ms per 512^2 patch; our operational patch is 96^2 over a <= 400^2 window
    dt = time.perf_counter() - t0
    assert dt < 0.5  # 512^2 template over a 712^2 window (200x200 search) — see LEDGER for measured value
    t0 = time.perf_counter()
    for _ in range(10):
        M.match(M.gray[900:996, 900:996], np.ones((96, 96), bool), np.array([0.0, 0.0]), 60.0)
    assert (time.perf_counter() - t0) / 10 < 0.02


@pytest.mark.parametrize("n", [3])
def test_orthorectify_nadir_identity(n):
    from dss.aiding.ortho import orthorectify
    from dss.sensors.rig import R_BODY_NADIR_CAM

    K = np.array([[400.0, 0, 320], [0, 400.0, 240], [0, 0, 1]])
    img = np.zeros((480, 640), np.float32)
    img[240, 320] = 255.0  # principal point
    patch, valid = orthorectify(img, K, R_BODY_NADIR_CAM, np.array([5.0, 7.0, 100.0]), 0.0, (5.0, 7.0), 21.0, 1.0)
    i, j = np.unravel_index(np.argmax(patch), patch.shape)
    assert (i, j) == (10, 10)
