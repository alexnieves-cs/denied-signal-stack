import os

import numpy as np
import pytest

os.environ["MAVLINK20"] = "1"


@pytest.mark.mavlink
def test_vpe_frame_roundtrip():
    from pymavlink.dialects.v20 import common as mav

    from dss.core.rotations import rot_z
    from dss.output.mavlink_vpe import VpeEncoder

    enc = VpeEncoder()
    b = enc.encode(123456, np.array([1.0, 2.0, -3.0]), rot_z(0.3), np.diag([4.0, 1.0, 0.25]), np.ones(3) * 1e-4, 0)
    assert b[0] == 0xFD
    m = mav.MAVLink(None)
    msg = m.parse_char(b)
    assert msg.get_type() == "VISION_POSITION_ESTIMATE"
    assert msg.usec == 123456
    # ENU (1, 2, -3) -> NED (2, 1, 3)
    assert np.allclose([msg.x, msg.y, msg.z], [2.0, 1.0, 3.0])
    assert len(msg.covariance) == 21 and all(np.isfinite(msg.covariance))
    # conservative horizontal diagonal: both axes carry the largest horizontal variance
    assert msg.covariance[0] == pytest.approx(4.0, rel=1e-6) and msg.covariance[6] == pytest.approx(4.0, rel=1e-6)
    # yaw: ENU yaw 0.3 (from east, CCW) -> NED yaw pi/2 - 0.3
    assert msg.yaw == pytest.approx(np.pi / 2 - 0.3, abs=1e-6)


def test_import_contract_only_gate_builds_gps_factors():
    """Only the integrity gate path may feed GPS into fusion: `.gps(` is called from app.py only
    under `use = gate.step(...)`, and no other module calls NavFilter.gps."""
    from pathlib import Path

    src = Path(__file__).resolve().parents[2] / "src" / "dss"
    callers = [p for p in src.rglob("*.py") if "allf.gps(" in p.read_text() or "ref.gps(" in p.read_text()]
    assert [p.name for p in callers] == ["app.py"]
    txt = (src / "app.py").read_text()
    assert "ref.gps(" not in txt  # REF never sees GPS
    i = txt.index("allf.gps(")
    assert "use = gate.step(" in txt[max(0, i - 900):i]
