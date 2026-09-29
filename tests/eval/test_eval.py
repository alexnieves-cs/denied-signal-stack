import numpy as np

from dss.core.rotations import rot_z, so3_exp
from dss.eval import align, metrics
from dss.eval.associate import associate
from dss.eval.consistency import anees_band, anees_test


def _path(g, n=500):
    return np.cumsum(g.normal(size=(n, 3)), axis=0)


def test_posyaw_recovers_yaw_translation(g):
    p = _path(g)
    R, t = rot_z(0.7), np.array([3.0, -2.0, 1.0])
    q = p @ R.T + t
    Re, te, _ = align.align_posyaw(p, q)
    assert np.abs(align.apply(Re, te, 1.0, p) - q).max() < 1e-9


def test_se3_umeyama_recovers(g):
    p = _path(g)
    R, t = so3_exp(np.array([0.3, -0.2, 1.1])), np.array([1.0, 2.0, 3.0])
    Re, te, _ = align.align_se3(p, p @ R.T + t)
    assert np.allclose(Re, R, atol=1e-10) and np.allclose(te, t, atol=1e-9)


def test_rpe_segments_longer_than_path(g):
    p = np.stack([np.linspace(0, 10, 100), np.zeros(100), np.zeros(100)], 1)
    r = metrics.rpe(p, p, lengths=(5, 50))
    assert r[5]["count"] > 0 and r[5]["mean_m"] == 0.0
    assert r[50]["count"] == 0


def test_endpoint_drift_self_test():
    p = np.stack([np.linspace(0, 100, 1001), np.zeros(1001), np.zeros(1001)], 1)
    e = p.copy()
    e[200:, 1] += np.linspace(0, 1.0, 801)  # drift starts after the alignment segment
    r = metrics.endpoint_drift(e, p, 20, 10)
    assert abs(r["endpoint_err_m"] - 0.994) < 0.01 and abs(r["drift_pct"] - 0.994) < 0.01


def test_ttr_self_test():
    t = np.arange(0, 100.0, 1.0)
    err = np.where(t < 40, 20.0, 1.0)
    inside = np.ones_like(t, bool)
    r = metrics.ttr(t, err, inside, t_loss=30.0)
    assert r["ttr_s"] == 10.0 and not r["censored"]
    assert metrics.ttr(t, np.full_like(t, 20.0), inside, 30.0)["censored"]
    assert metrics.ttr(t, np.ones_like(t), inside, 30.0)["ttr_s"] == 0.0


def test_associate_tolerance():
    a = np.array([0, 10_000_000, 20_000_000, 30_000_000])
    b = np.array([1_000_000, 19_000_000, 45_000_000])
    ia, ib = associate(a, b, 0.005)
    assert list(ia) == [0, 2] and list(ib) == [0, 1]


def test_anees_band_values():
    lo, hi = anees_band(3, 50)
    assert abs(lo - 2.36) < 0.01 and abs(hi - 3.72) < 0.01


def test_anees_test_accepts_consistent(g):
    e = g.normal(size=(50, 200, 3))
    nees = np.sum(e**2, -1)
    assert anees_test(nees, 3)["pass"]
    assert not anees_test(4 * nees, 3)["pass"]


def test_ate_drift_pct(g):
    p = np.stack([np.linspace(0, 100, 101), np.zeros(101), np.zeros(101)], 1)
    r = metrics.ate(p + np.array([0, 0.5, 0]), p, "none")
    assert abs(r["rmse_m"] - 0.5) < 1e-12 and abs(r["drift_pct"] - 0.5) < 1e-9
