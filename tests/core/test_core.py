import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from dss.core import rotations as rot
from dss.core.frames import SE3, enu_to_ned, ned_to_enu
from dss.core.rng import rng
from dss.core.time import ns_to_s, s_to_ns

vec = st.lists(st.floats(-3, 3), min_size=3, max_size=3).map(np.array)


@given(vec)
@settings(max_examples=200, deadline=None)
def test_exp_log_roundtrip(phi):
    if np.linalg.norm(phi) > 3.1:
        return
    assert np.allclose(rot.so3_log(rot.so3_exp(phi)), phi, atol=1e-9)


@given(vec)
@settings(max_examples=200, deadline=None)
def test_quat_rot_roundtrip(phi):
    R = rot.so3_exp(phi)
    q = rot.rot_to_quat(R)
    assert np.allclose(rot.quat_to_rot(q), R, atol=1e-12)


def test_quat_mul_matches_matrix(g):
    for _ in range(50):
        a, b = rot.so3_exp(g.normal(size=3)), rot.so3_exp(g.normal(size=3))
        qa, qb = rot.rot_to_quat(a), rot.rot_to_quat(b)
        assert np.allclose(rot.quat_to_rot(rot.quat_mul(qa, qb)), a @ b, atol=1e-12)


def test_jpl_boundary_roundtrip(g):
    q = rot.rot_to_quat(rot.so3_exp(g.normal(size=3)))
    assert np.allclose(rot.jpl_to_hamilton(rot.hamilton_to_jpl(q)), q)


def test_time_ns_roundtrip():
    t = np.array([0.0, 1.5, 1234.000000001])
    assert np.array_equal(s_to_ns(t), np.array([0, 1_500_000_000, 1_234_000_000_001]))
    assert np.allclose(ns_to_s(s_to_ns(t)), t)


def test_enu_ned_involution(g):
    v = g.normal(size=(10, 3))
    assert np.allclose(ned_to_enu(enu_to_ned(v)), v)
    assert np.allclose(enu_to_ned(np.array([1.0, 2.0, 3.0])), [2.0, 1.0, -3.0])


def test_se3_chain_inverse_1e12(g):
    ts = [SE3(rot.so3_exp(g.normal(size=3)), g.normal(size=3)) for _ in range(6)]
    T = ts[0]
    for t in ts[1:]:
        T = T @ t
    Ti = ts[-1].inv()
    for t in reversed(ts[:-1]):
        Ti = Ti @ t.inv()
    assert np.abs((T @ Ti).matrix() - np.eye(4)).max() < 1e-12


def test_rng_streams_stable_and_independent():
    a1 = rng(42, "imu", "imu0").standard_normal(5)
    a2 = rng(42, "imu", "imu0").standard_normal(5)
    b = rng(42, "imu", "imu1").standard_normal(5)
    c = rng(42, "gnss", "imu0").standard_normal(5)
    assert np.array_equal(a1, a2)
    assert not np.allclose(a1, b) and not np.allclose(a1, c)
