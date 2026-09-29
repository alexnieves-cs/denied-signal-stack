import numpy as np

from dss.baselines import dead_reckoning as dr
from dss.core import config
from dss.core.rng import rng
from dss.eval.consistency import anees_test
from dss.fusion.eskf import ImuNoise
from dss.sensors.samples import Stream
from dss.sim import imu, trajectory

IC = config.load("sensors/imu_bmi160.yaml")


def test_dr_growth_matches_half_bias_t2():
    st = trajectory.static(61.0)
    d = imu.simulate_imu(st, imu.ImuParams.from_config(IC), 0, t0=0.0, t1=60.0, noise=False)
    ba = np.array([0.02, -0.01, 0.0])
    d["acc"] = d["acc"] + ba  # constant accelerometer bias, level, no rotation
    s = Stream.from_dict("imu0", "imu", d)
    x0 = dict(t_ns=d["t_ns"][0], p=np.zeros(3), v=np.zeros(3), R=np.eye(3), bg=np.zeros(3), ba=np.zeros(3))
    r = dr.run(s, x0, dr.default_P0(), ImuNoise.from_config(IC), out_hz=1.0)
    t = (r.t_ns - r.t_ns[0]) * 1e-9
    pred = 0.5 * np.linalg.norm(ba) * t**2
    m = t > 5
    assert np.abs(np.linalg.norm(r.p[m], axis=1) / pred[m] - 1).max() < 0.01


def test_dr_position_anees_n50():
    st = trajectory.static(61.0)
    P0 = dr.default_P0(0.1, 0.05, np.deg2rad(0.5), 1e-4, 0.01)
    nees = []
    for seed in range(50):
        d = imu.simulate_imu(st, imu.ImuParams.from_config(IC), seed, t0=0.0, t1=60.0)
        s = Stream.from_dict("imu0", "imu", d)
        x = st.sample([0.0])
        p, v, R, bg, ba = dr.perturb_initial(x.p[0], x.v[0], x.R[0], np.zeros(3), np.zeros(3), P0, rng(seed, "scenario", "init"))
        r = dr.run(s, dict(t_ns=d["t_ns"][0], p=p, v=v, R=R, bg=bg, ba=ba), P0, ImuNoise.from_config(IC), out_hz=1.0)
        e = r.p - st.sample(r.t_ns * 1e-9).p
        nees.append(np.einsum("ti,tij,tj->t", e, np.linalg.inv(r.P_pos), e)[1:])
    res = anees_test(np.array(nees), 3)
    assert res["pass"], {k: v for k, v in res.items() if k != "anees_t"}
