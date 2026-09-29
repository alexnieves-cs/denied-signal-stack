import os
import shutil
import signal
from pathlib import Path

import numpy as np
import pytest

from dss.core.rotations import quat_to_rot, rot_to_quat, skew, so3_exp, so3_log
from dss.eval.consistency import anees_test
from dss.vio import protocol as pr
from dss.vio.ovclient import DEFAULT_BINARY, OvClient
from dss.vio.protocol import VioFailure
from dss.vio.synthetic import SyntheticVio

REPO = Path(__file__).resolve().parents[2]
CFG = REPO / "cpp/ovserver/config/tum_vi_mono/estimator_config.yaml"
ROOM1 = REPO / "data/tumvi/dataset-room1_512_16"
need_server = pytest.mark.skipif(not Path(DEFAULT_BINARY).exists(), reason="ovserver not built")


# --- pure codec / convention tests (CI-safe) ------------------------------------------------
def _R_jpl(q_xyzw):
    """JPL rotation matrix (Trawny & Roumeliotis) of q = [qv, q4]."""
    qv, q4 = np.asarray(q_xyzw[:3]), q_xyzw[3]
    return (2 * q4**2 - 1) * np.eye(3) - 2 * q4 * skew(qv) + 2 * np.outer(qv, qv)


def test_jpl_quaternion_is_hamilton_of_transpose(g):
    for _ in range(50):
        R_ItoG = so3_exp(g.normal(size=3))
        q_h = rot_to_quat(R_ItoG)
        q_jpl = pr.hamilton_to_jpl(q_h)
        assert np.allclose(_R_jpl(q_jpl), R_ItoG.T, atol=1e-12)  # JPL q describes R_GtoI
        assert np.allclose(pr.jpl_to_hamilton(q_jpl), q_h)


def test_jpl_error_equals_local_theta_monte_carlo(g):
    """OpenVINS JPL error dtheta (R_GtoI_true = (I - [dth]x) R_GtoI_est) equals our local theta."""
    errs = []
    for _ in range(2000):
        R_est = so3_exp(g.normal(size=3))
        dth = g.normal(0, 1e-3, 3)
        R_GtoI_true = (np.eye(3) - skew(dth)) @ R_est.T
        u, _, vt = np.linalg.svd(R_GtoI_true)
        R_GtoI_true = u @ vt
        theta_local = so3_log(R_est.T @ R_GtoI_true.T)
        errs.append(theta_local - dth)
    rel = np.linalg.norm(errs, axis=1).max() / 1e-3
    assert rel < 0.02  # second-order residual only


def test_cov15_reorder_roundtrip(g):
    A = g.normal(size=(15, 15))
    P = A @ A.T
    Pov = pr.cov15_ours_to_ov(P)
    assert np.allclose(pr.cov15_ov_to_ours(Pov), P)
    # ours theta (6:9) lands in ov 0:3, ours p (0:3) in ov 3:6
    assert np.allclose(Pov[0:3, 0:3], P[6:9, 6:9]) and np.allclose(Pov[3:6, 3:6], P[0:3, 0:3])
    assert np.allclose(Pov[0:3, 3:6], P[6:9, 0:3])


def test_state_decoder_schema():
    vals = np.zeros(47)
    vals[0] = 1.5
    vals[1:5] = [0, 0, 0, 1]
    vals[11:47] = np.eye(6).ravel()
    b = bytes([0]) + vals.astype("<f8").tobytes() + np.array([2, 5, 1], "<u4").tobytes() + np.array([2], "<u2").tobytes()
    b += np.array([(7, 1.0, 2.0), (9, 3.0, 4.0)], dtype=[("i", "<u4"), ("u", "<f4"), ("v", "<f4")]).tobytes()
    s = pr.dec_state(b)
    assert s.ok and s.n_tracked == 2 and s.n_msckf == 5 and s.n_slam == 1
    assert s.tracks.shape == (2, 3) and s.tracks[1, 0] == 9 and np.allclose(s.q, [1, 0, 0, 0])
    with pytest.raises(VioFailure):
        pr.dec_state(b[:-1])


def test_synthetic_vio_nees_consistent():
    nees = []
    t = np.arange(0, 120, 0.1)
    for seed in range(50):
        v = SyntheticVio(seed, drift_frac=0.01)
        v.init_gt(0.0, [1, 0, 0, 0], [0, 0, 0], [5, 0, 0])
        e = []
        for tk in t:
            p = np.array([5.0 * tk, 0.0, 0.0])
            s = v.step(tk, p, [1, 0, 0, 0], [5, 0, 0])
            dth = so3_log(quat_to_rot(s.q))
            err = np.r_[s.p - p, dth]
            e.append(err @ np.linalg.solve(s.cov6, err))
        nees.append(e[10::10])
    r = anees_test(np.array(nees), 6)
    assert r["pass"], (r["mean"], r["frac_inside"])


# --- ovserver process tests ------------------------------------------------------------------
@pytest.fixture
def client():
    c = OvClient(CFG)
    yield c
    c.close()


def _frames(n=50):
    rng = np.random.default_rng(0)
    base = (rng.integers(0, 255, (560, 560)) > 200).astype(np.uint8) * 200 + 20
    import cv2

    base = cv2.GaussianBlur(base, (3, 3), 0)
    return [np.ascontiguousarray(base[k % 40: k % 40 + 512, : 512]) for k in range(n)]


@pytest.mark.machine
@need_server
def test_10k_frames_verbosity_all_zero_framing_errors(tmp_path):
    """Framing under maximum OpenVINS printing. A static scene with a static IMU keeps the frames
    physically consistent (moving synthetic frames + static IMU drive OpenVINS into its own
    std::exit on a negative covariance diagonal, which the client reports as VioFailure)."""
    d = tmp_path / "cfg"
    shutil.copytree(CFG.parent, d)
    y = d / "estimator_config.yaml"
    y.write_text(y.read_text().replace('verbosity: "WARNING"', 'verbosity: "ALL"'))
    base = _frames(1)[0]
    noise = np.random.default_rng(1)
    fr = [np.clip(base.astype(int) + noise.integers(-2, 3, base.shape), 0, 255).astype(np.uint8) for _ in range(20)]
    c = OvClient(y, stderr_path=os.devnull)
    try:
        c.init()
        t = 100.0
        c.imu(t, [0, 0, 0], [0, 0, 9.81])
        c.init_gt(t, [1, 0, 0, 0], [0, 0, 0], [0, 0, 0])
        statuses = set()
        for k in range(10_000):
            for j in range(10):
                c.imu(t + 0.005 * (j + 1), [0, 0, 0], [0, 0, 9.81])
            t += 0.05
            s = c.cam(t, fr[k % len(fr)])
            statuses.add(s.status)
            assert s.tracks.shape[0] == min(s.n_tracked, 200)
        assert c.framing_errors == 0
        assert statuses <= {pr.ST_OK, pr.ST_NOT_INITIALIZED, pr.ST_ERROR}
        assert c.ping().startswith("ovserver")
    finally:
        c.close()


@pytest.mark.machine
@need_server
def test_malformed_rejected_without_exit(client):
    client.init()
    client.imu(1.0, [0, 0, 0], [0, 0, 9.81])
    r = client.cam(1.0, np.zeros((100, 100), np.uint8))  # wrong size
    assert r.status == pr.ST_BAD_REQUEST
    rtyp, b = client.request(pr.CAM, b"\x00\x01")  # truncated
    assert rtyp == pr.RSP_STATE and b[0] == pr.ST_BAD_REQUEST
    rtyp, b = client.request(42, b"junk")
    assert rtyp == pr.RSP_ACK and b[0] == pr.ST_BAD_REQUEST
    with pytest.raises(ValueError):
        client.imu(0.5, [0, 0, 0], [0, 0, 9.81])  # non-increasing time
    assert client.ping().startswith("ovserver")
    assert client.proc.poll() is None


@pytest.mark.machine
@need_server
def test_killed_server_raises_vio_failure_within_one_frame(client):
    client.init()
    client.imu(1.0, [0, 0, 0], [0, 0, 9.81])
    os.kill(client.proc.pid, signal.SIGKILL)
    client.proc.wait()
    with pytest.raises(VioFailure):
        client.cam(1.0, np.zeros((512, 512), np.uint8))


@pytest.mark.machine
@need_server
def test_state_schema_on_real_frames(client):
    fr = _frames(30)
    client.init()
    t = 10.0
    client.imu(t, [0, 0, 0], [0, 0, 9.81])
    client.init_gt(t, [1, 0, 0, 0], [0, 0, 0], [0, 0, 0])
    for k in range(30):
        for j in range(10):
            client.imu(t + 0.005 * (j + 1), [0, 0, 0], [0, 0, 9.81])
        t += 0.05
        s = client.cam(t, fr[k])
        assert s.status == pr.ST_OK
        assert s.p.shape == (3,) and s.q.shape == (4,) and abs(np.linalg.norm(s.q) - 1) < 1e-9
        assert np.allclose(s.cov6, s.cov6.T, atol=1e-9) and np.linalg.eigvalsh(s.cov6).min() > 0
        assert s.tracks.shape == (min(s.n_tracked, 200), 3)
        assert np.all((s.tracks[:, 1] >= 0) & (s.tracks[:, 1] < 512))


@pytest.mark.machine
@need_server
def test_reanchor_covariance_used(client):
    fr = _frames(20)
    client.init()
    t = 10.0
    for j in range(700):
        client.imu(t + 0.005 * j, [0, 0, 0], [0, 0, 9.81])
    t += 0.005 * 699
    P = np.diag(np.r_[[2.4**2] * 2, 1.0, [0.2**2] * 3, [np.deg2rad(0.5) ** 2] * 2, np.deg2rad(1) ** 2, [1e-6] * 3, [1e-4] * 3])
    client.reanchor(t, [1, 0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], P)
    for j in range(10):
        client.imu(t + 0.005 * (j + 1), [0, 0, 0], [0, 0, 9.81])
    s = client.cam(t + 0.05, fr[0])
    assert s.ok and abs(np.sqrt(s.cov6[0, 0]) - 2.4) < 0.3 and abs(np.sqrt(s.cov6[2, 2]) - 1.0) < 0.2
    assert abs(np.rad2deg(np.sqrt(s.cov6[5, 5])) - 1.0) < 0.3


@pytest.mark.machine
@pytest.mark.data
@need_server
@pytest.mark.skipif(not ROOM1.exists(), reason="TUM-VI room1 not fetched")
def test_room1_determinism_and_tracks():
    from dss.vio.datasets_run import run_dataset

    a = run_dataset(ROOM1, "mono", max_frames=400)
    b = run_dataset(ROOM1, "mono", max_frames=400)
    assert a.log_sha256() == b.log_sha256()
    assert a.framing_errors == 0 and (a.status == 0).all() and a.n_tracked.min() > 20
