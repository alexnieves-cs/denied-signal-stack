import numpy as np
import pytest

from dss.eval.align import align_se3, apply
from dss.live.mono_vo import MonoVO

pytestmark = pytest.mark.gpu


def test_loop_closes_after_sim3_on_rendered_loop():
    """A 'desk loop' proxy: an oblique camera flying a closed circle over rendered imagery; after Sim(3)
    alignment the up-to-scale VO trajectory matches within 10% of the loop length."""
    from dss.app import renderer, world
    from dss.core.rotations import euler_zyx_to_rot
    from dss.sim.render.camera import to_gray

    W = world()
    r = renderer(W)
    vo = MonoVO(r.K.copy())
    n = 400
    gt = []
    for k in range(n + 1):
        a = 2 * np.pi * k / n
        p = np.array([-300 + 60 * np.cos(a), 60 * np.sin(a), 0.0])
        p[2] = W["dem"].sample(p[0], p[1]) + 60.0
        R = euler_zyx_to_rot(0.0, 0.0, a) @ np.array([[0.0, -1, 0], [-1, 0, 0], [0, 0, -1]])
        vo.step(to_gray(r.render(p, R)["rgb"]).astype(np.uint8))
        gt.append(p)
    kf = np.asarray(vo.traj)
    assert len(kf) > 20 and vo.reinits <= 1
    # compare keyframe positions with GT at matching progress (uniform resample)
    g = np.asarray(gt)
    idx = np.linspace(0, len(g) - 1, len(kf)).astype(int)
    Rr, tt, s = align_se3(kf, g[idx], with_scale=True)
    err = np.linalg.norm(apply(Rr, tt, s, kf) - g[idx], axis=1)
    L = 2 * np.pi * 60
    assert np.sqrt(np.mean(err**2)) / L < 0.10
