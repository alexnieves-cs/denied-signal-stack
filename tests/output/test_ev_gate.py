import numpy as np

from dss.eval.px4_ev_gate import replay


def test_ev_gate_accepts_consistent_and_rejects_jump():
    t = np.arange(0, 60, 1 / 30)
    p = np.stack([2.0 * t, np.zeros_like(t), np.zeros_like(t)], 1)
    a = np.zeros_like(p)
    g = np.random.default_rng(0)
    r = replay(t, p + g.normal(0, 0.3, p.shape), np.full(len(t), 0.3**2), a)
    assert r["rejections"] == 0
    pj = p.copy()
    pj[900:, 0] += 30.0  # 30 m jump the flight controller would see
    r2 = replay(t, pj, np.full(len(t), 0.3**2), a)
    assert r2["rejections"] > 0
