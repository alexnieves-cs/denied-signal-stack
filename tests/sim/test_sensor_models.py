import numpy as np
import pytest

from dss.baselines import dead_reckoning as dr
from dss.core import config
from dss.core.rotations import quat_to_rot, so3_log
from dss.fusion.eskf import ImuNoise
from dss.sensors.samples import Stream
from dss.sim import baro, flow, gnss, imu, mag, rangefinder, trajectory
from dss.sim.runner import flat_surface


@pytest.fixture(scope="module")
def route():
    return trajectory.route()


def test_noise_free_strapdown_closes(route):
    ic = config.load("sensors/imu_bmi160.yaml")
    d = imu.simulate_imu(route, imu.ImuParams.from_config(ic), 1, t0=25.0, t1=85.0, noise=False)
    s = Stream.from_dict("imu0", "imu", d)
    x = route.sample([25.0])
    x0 = dict(t_ns=d["t_ns"][0], p=x.p[0], v=x.v[0], R=x.R[0], bg=np.zeros(3), ba=np.zeros(3))
    r = dr.run(s, x0, dr.default_P0(), ImuNoise.from_config(ic))
    gt = route.sample(r.t_ns * 1e-9)
    assert np.linalg.norm(r.p[-1] - gt.p[-1]) < 0.01
    assert np.linalg.norm(so3_log(quat_to_rot(r.q[-1]).T @ gt.R[-1])) < 1e-4


def test_flatness_rates_match_finite_differences(route):
    t = np.linspace(1.0, route.t1 - 1.0, 400)
    w = route.body_rates(t)
    h = 1e-4  # 10 kHz
    R0, R1 = route.rotation(t - h / 2), route.rotation(t + h / 2)
    w_fd = np.stack([so3_log(a.T @ b) / h for a, b in zip(R0, R1)])
    assert np.abs(w - w_fd).max() < 1e-6


def test_no_rate_jumps_at_joints(route):
    # continuity: the largest sample-to-sample change must shrink linearly with the step
    d1 = np.abs(np.diff(route.body_rates(np.arange(0.5, route.t1 - 0.5, 0.004)), axis=0)).max()
    d2 = np.abs(np.diff(route.body_rates(np.arange(0.5, route.t1 - 0.5, 0.001)), axis=0)).max()
    assert d2 < 0.3 * d1


def test_route_geometry(route):
    t = np.arange(0, route.t1, 0.01)
    s = route.sample(t)
    L = np.sum(np.linalg.norm(np.diff(s.p[:, :2], axis=0), axis=1))
    assert abs(L - 2000.0) < 1.0
    assert abs(s.p[:, 2].max() - 100.0) < 1e-6


def test_isa_roundtrip():
    h = np.linspace(-100, 5000, 200)
    assert np.abs(baro.isa_altitude(baro.isa_pressure(h)) - h).max() < 1e-6


def test_wmm_declination_flagstaff():
    c = config.load("sensors/mag_default.yaml")
    r = mag.wmm_field(c["lat_deg"], c["lon_deg"], c["alt_km"], c["decimal_year"])
    # WMM2025 at Flagstaff: ~9.8 deg E, inclination ~61 deg, F ~47.6 uT (sanity, not the NOAA fixture)
    assert 9.0 < r.d < 10.5 and 60 < r.i < 62 and 47000 < r.f < 48200


def test_gnss_static_statistics():
    c = config.load("sensors/gnss_m9n.yaml")
    t = np.arange(0, 3600, 0.2)
    p = np.zeros((len(t), 3))
    errs, verr = [], []
    for seed in range(20):
        d = gnss.simulate_gnss(t, p, p, c, seed)
        errs.append(np.linalg.norm(d["pos"][:, :2], axis=1))
        verr.append(d["vel"])
    cep = np.median(np.concatenate(errs))
    assert abs(cep / 1.5 - 1) < 0.05
    sv = np.concatenate(verr).std()
    assert abs(sv / 0.05 - 1) < 0.10
    # fitted tau_pos from the lag-1 autocorrelation within +-20%
    taus = []
    for seed in range(20):
        e = gnss.simulate_gnss(t, p, p, c, seed)["err_pos"][:, 0]
        rho = np.corrcoef(e[:-1], e[1:])[0, 1]
        taus.append(-0.2 / np.log(rho))
    assert abs(np.median(taus) / c["tau_pos_s"] - 1) < 0.2


def test_gnss_jamming_and_hold():
    c = config.load("sensors/gnss_m9n.yaml")
    t = np.arange(0, 120, 0.2)
    p = np.zeros((len(t), 3))
    jam = {"t0": 30.0, "t_full": 30.0, "t1": 60.0, "drop_dbhz": 30.0}
    d = gnss.simulate_gnss(t, p, p, c, 1, jam=jam)
    lost = t[~d["valid"]]
    assert lost.min() == pytest.approx(30.0, abs=0.21)  # no fix within one epoch of C/N0 < min
    back = t[(t > 60) & d["valid"]].min()
    assert back >= 60.0 + c["reacquire_hold_s"] - 1e-9


def test_spoof_profiles_exact():
    c = config.load("sensors/gnss_m9n.yaml")
    t = np.arange(0, 200, 0.2)
    p = np.zeros((len(t), 3))
    spoof = {"kind": "dragoff", "t0": 120.0, "rate_mps": 1.0, "drag_delay_s": 5.0, "dir": [0, 1, 0]}
    d = gnss.simulate_gnss(t, p, p, c, 3, spoof=spoof)
    exp = np.clip(t - 125.0, 0, None) * (t >= 120)
    assert np.allclose(d["spoof_offset"][:, 1], exp, atol=1e-12)
    assert np.allclose(d["pos"] - d["err_pos"], d["spoof_offset"], atol=1e-12)
    step = gnss.spoof_offset(t, {"kind": "step", "t0": 50.0, "offset_m": 20.0}, p, p)[0]
    assert np.allclose(step[:, 0], 20.0 * (t >= 50.0))
    acc = gnss.spoof_offset(t, {"kind": "dragoff", "t0": 10.0, "accel_mps2": 0.05}, p, p)[0]
    assert np.allclose(acc[:, 0], 0.025 * np.clip(t - 10, 0, None) ** 2 * (t >= 10))


def test_flow_recovers_velocity(route):
    c = config.load("sensors/flow_px4flow.yaml")
    surf = flat_surface(0.0)
    tr = trajectory.route(agl_m=15.0)
    t = np.arange(30.0, 200.0, 0.05)
    d = flow.simulate_flow(tr, t, c, 1, surf)
    s = tr.sample(t)
    h_perp = s.p[:, 2] * s.R[:, 2, 2]
    v = flow.flow_to_velocity(d["flow_xy"], d["gyro_xyz"], 0.05, h_perp)
    # truth: mean body velocity over the integration interval
    vb = np.mean([np.einsum("nji,nj->ni", tr.sample(t - 0.05 * (k + 0.5) / 10).R, tr.sample(t - 0.05 * (k + 0.5) / 10).v)
                  for k in range(10)], axis=0)[:, :2]
    rms = np.sqrt(np.mean(np.sum((v - vb) ** 2, 1))) / np.sqrt(np.mean(np.sum(vb**2, 1)))
    assert rms < 0.05
    assert d["valid"].all()


def test_lidar_sigma_and_invalid(route):
    c = config.load("sensors/rangefinder_sf11c.yaml")
    t = np.arange(0, route.t1, 0.05)
    s = route.sample(t)
    d = rangefinder.simulate_rangefinder(t, s.p, s.R, c, 1, flat_surface(0.0))
    ok = d["valid"]
    r = d["range_true"][ok]
    assert np.allclose(d["sigma_m"][ok], np.sqrt(0.1**2 + (c["sigma_rel"] * r) ** 2))
    z = (d["range_m"][ok] - r) / d["sigma_m"][ok]
    assert abs(z.std() - 1) < 0.05
    far = trajectory.route(agl_m=150.0)
    s2 = far.sample(np.array([150.0]))
    d2 = rangefinder.simulate_rangefinder(np.array([150.0]), s2.p, s2.R, c, 1, flat_surface(0.0))
    assert not d2["valid"][0] and np.isnan(d2["range_m"][0])
