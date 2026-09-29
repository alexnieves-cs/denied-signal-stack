import hashlib
import time

import numpy as np
import pytest

from dss.core.rotations import euler_zyx_to_rot
from dss.geo.aoi import Raster
from dss.sensors.rig import R_BODY_NADIR_CAM

pytestmark = pytest.mark.gpu


def _scene(dots):
    from dss.sim.render.camera import PinholeRenderer
    from dss.sim.render.scene import build_model

    res = 0.1
    img = np.zeros((4000, 4000, 4), np.uint8) + 30  # 400 x 400 m at 0.1 m, xmin=-200, ymax=200
    for x, y in dots:
        i, j = int((200 - y) / res), int((x + 200) / res)
        img[i - 3:i + 4, j - 3:j + 4, :3] = 255
    dem = Raster(np.zeros((81, 81), np.float32), -200.0, 200.0, 5.0)
    m = build_model(Raster(img, -200.0, 200.0, res), dem, (-200, 200, -200, 200), 640, 480, 400.0, mesh_step=10.0)
    return PinholeRenderer(m, 640, 480, 400.0)


def test_projection_residual_below_0p2px():
    dots = [(0.0, 0.0), (20.0, 10.0), (-30.0, 15.0), (25.0, -20.0)]
    r = _scene(dots)
    R = euler_zyx_to_rot(0.05, -0.03, 0.7) @ R_BODY_NADIR_CAM
    p = np.array([1.3, -2.1, 100.0])
    out = r.render(p, R, depth=True)
    g = out["rgb"][..., 0].astype(float)
    res = []
    for x, y in dots:
        X = np.array([x, y, 0.0]) + 0.05 * np.array([1, -1, 0])  # dot centre (pixel-centre convention of the texture)
        xc = R.T @ (X - p)
        uv = (r.K @ xc)[:2] / xc[2]
        u0, v0 = int(round(uv[0])), int(round(uv[1]))
        w = g[v0 - 6:v0 + 7, u0 - 6:u0 + 7]
        w = np.clip(w - 30, 0, None)
        vv, uu = np.mgrid[v0 - 6:v0 + 7, u0 - 6:u0 + 7]
        c = np.array([np.sum(uu * w) / w.sum(), np.sum(vv * w) / w.sum()])
        res.append(np.linalg.norm(c + 0.5 - uv))  # pixel centres at +0.5
    assert max(res) < 0.2, res
    # depth aligned with RGB: depth at the principal point equals the range along the optical axis
    zc = R[:, 2]
    rng_axis = -p[2] / zc[2]
    assert abs(out["depth"][240, 320] - rng_axis) < 0.05


def test_tier_b_identical_frames():
    r = _scene([(0.0, 0.0), (10.0, 5.0)])
    hs = []
    for _ in range(2):
        h = hashlib.sha256()
        for k in range(100):
            R = euler_zyx_to_rot(0.0, 0.0, 0.01 * k) @ R_BODY_NADIR_CAM
            h.update(r.render(np.array([0.1 * k, 0.0, 100.0]), R)["rgb"].tobytes())
        hs.append(h.hexdigest())
    assert hs[0] == hs[1]


@pytest.mark.machine
def test_fps_as_configured_on_route():
    from dss.app import renderer, world
    from dss.sim import trajectory

    W = world()
    r = renderer(W)
    tr = trajectory.route(heading0=np.pi, ground_fn=W["dem"].sample)
    ts = np.arange(30, 30 + 1000 * 0.05, 0.05)
    s = tr.sample(ts)
    t0 = time.perf_counter()
    for k in range(len(ts)):
        r.render(s.p[k], s.R[k] @ R_BODY_NADIR_CAM)
    fps = len(ts) / (time.perf_counter() - t0)
    assert fps >= 20.0, fps
