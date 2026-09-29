from dss.core import config
from dss.sim import runner


def test_tier_a_identical_hashes():
    scn = config.load("scenarios/dr_route_2km.yaml")
    a = runner.simulate(scn, 42).hashes()
    b = runner.simulate(scn, 42).hashes()
    assert a == b
    c = runner.simulate(scn, 43).hashes()
    assert all(a[k] != c[k] for k in a if k != "gt")


def test_adding_stream_changes_no_existing_hash():
    scn = config.load("scenarios/dr_route_2km.yaml")
    base = dict(scn, enable=["imu", "baro", "mag"])
    a = runner.simulate(base, 42).hashes()
    b = runner.simulate(scn, 42).hashes()
    for k, v in a.items():
        assert b[k] == v
