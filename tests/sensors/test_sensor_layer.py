from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from dss.core.rng import rng
from dss.sensors import calibration, timesync
from dss.sensors.samples import Stream
from dss.sensors.stream import merge_order

OV = Path.home() / ".cache/dss/cpp/ws/src/open_vins/config"
TUMVI = Path(__file__).resolve().parents[2] / "data/tumvi/dataset-room1_512_16"


def test_offset_and_jitter_recovered():
    t = np.arange(0, 60.0, 0.005)
    sig = np.sin(2 * np.pi * 0.7 * t) + 0.5 * np.sin(2 * np.pi * 1.9 * t + 1) + 0.3 * np.sin(2 * np.pi * 3.1 * t)
    t_ns = (t * 1e9).astype(np.int64)
    for off in (0.0123, -0.0047):
        t2 = timesync.inject(t_ns, offset_s=off, jitter_s=0.0005, g=rng(1, "camera", "jit"))
        est = timesync.estimate_offset(t_ns, sig, t2, sig, max_offset_s=0.05, grid_hz=4000)
        assert abs(est - off) < 1e-4


@given(st.lists(st.tuples(st.integers(0, 50), st.sampled_from(["imu0", "cam0", "gnss0", "baro0"])), min_size=1, max_size=60))
@settings(max_examples=100, deadline=None)
def test_merged_stream_ordered_and_deterministic(evts):
    kinds = {"imu0": "imu", "cam0": "cam", "gnss0": "gnss", "baro0": "baro"}
    streams = []
    for name in kinds:
        ts = sorted(t for t, n in evts if n == name)
        streams.append(Stream(name, kinds[name], np.array(ts, dtype=np.int64)))
    a = merge_order(streams)
    b = merge_order(list(reversed(streams)))
    assert np.array_equal(a, b)
    assert np.all(np.diff(a["t"]) >= 0)
    # IMU precedes the camera at equal timestamps
    for t in np.unique(a["t"]):
        pr = a["prio"][a["t"] == t]
        assert np.all(np.diff(pr) >= 0)


def test_slerp_interp_endpoints():
    q = np.array([[1.0, 0, 0, 0], [np.cos(0.5), np.sin(0.5), 0, 0]])
    t = np.array([0, 10], dtype=np.int64)
    _, qq = timesync.interp_pose(np.array([0, 5, 10]), t, np.zeros((2, 3)), q)
    assert np.allclose(qq[1], [np.cos(0.25), np.sin(0.25), 0, 0])


FIX = Path(__file__).resolve().parents[1] / "fixtures"


@pytest.mark.machine
def test_euroc_calibration_matches_openvins():
    """EuRoC sensor.yaml chain T_imu_cam equals OpenVINS's shipped kalibr T_imu_cam to 1e-9."""
    ov = calibration.load_openvins_imucam(OV / "euroc_mav" / "kalibr_imucam_chain.yaml")
    t_ic = calibration.euroc_T_imu_cam(FIX / "euroc_imu0_sensor.yaml", FIX / "euroc_cam0_sensor.yaml")
    diff = np.abs(t_ic.matrix() - ov[0].T_imu_cam.matrix()).max()
    assert diff < 1e-9, diff
