import numpy as np

from dss.integrity import gnss_gate as gg
from dss.integrity.separation import Publisher, separation_stat


def _run(offset_fn, vel_fn, T=200.0, seed=0):
    g = np.random.default_rng(seed)
    gate = gg.GnssGate()
    P = np.eye(3) * 1.0
    Pv = np.eye(3) * 0.05**2
    e = np.zeros(2)
    phi = np.exp(-0.2 / 60)
    for k in range(int(T / 0.2)):
        t = k * 0.2
        e = phi * e + 1.27 * np.sqrt(1 - phi**2) * g.normal(size=2)
        pos = np.r_[e + offset_fn(t), 0]
        vel = np.r_[g.normal(0, 0.05, 2) + vel_fn(t), 0]
        gate.step(t, True, pos, vel, np.zeros(3), P, np.zeros(3), Pv, None)
    return gate


def test_no_false_alarm_nominal_many_seeds():
    for s in range(20):
        gate = _run(lambda t: 0.0, lambda t: 0.0, seed=s)
        assert gate.alarm_t is None, (s, gate.events)
        assert gate.state == gg.TRUSTED


def test_step_spoof_detected_fast():
    gate = _run(lambda t: np.array([20.0, 0]) * (t >= 100), lambda t: 0.0)
    assert gate.alarm_t is not None and gate.alarm_t - 100 <= 1.2
    assert gate.state == gg.REJECTED


def test_dragoff_detected_via_doppler():
    gate = _run(lambda t: np.array([0, 1.0]) * max(t - 100, 0), lambda t: np.array([0, 1.0]) * (t >= 100))
    assert gate.alarm_t is not None and gate.alarm_t - 100 <= 2.0


def test_probation_blocks_fusion():
    gate = gg.GnssGate()
    used = [gate.step(t * 0.2, True, np.zeros(3), np.zeros(3), np.zeros(3), np.eye(3), np.zeros(3), np.eye(3) * 0.01, None)
            for t in range(100)]
    assert not any(used[: int(34 / 0.2)])


def test_publisher_bleeds_absolute_corrections():
    pub = Publisher(np.zeros(3), np.eye(3), np.eye(3))
    target = np.array([10.0, 0, 0])
    for _ in range(30):
        p0 = pub.p.copy()
        pub.step(1 / 30, target, np.zeros(3), np.eye(3), np.eye(3))
        assert np.linalg.norm(pub.p - p0) <= 0.5 / 30 + 1e-12
    assert pub.hpl() > 5.68 * 9.0  # un-bled 9.5 m offset is carried in the covariance


def test_separation_floor():
    assert separation_stat(np.array([1.0, 0, 0]), np.eye(3) * 0.1, np.zeros(3), np.eye(3) * 0.1) < 1.0
