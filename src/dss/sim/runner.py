"""Generate every non-visual sensor stream + ground truth for a scenario (determinism tier A)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from dss.core import config as cfgmod
from dss.core.time import s_to_ns
from dss.sensors.samples import Stream
from dss.sim import baro, flow, gnss, imu, mag, rangefinder, trajectory


def flat_surface(z0: float = 0.0):
    return lambda x, y: np.full(np.shape(x), z0, dtype=float)


@dataclass
class SimOutput:
    scenario: dict
    seed: int
    traj: trajectory.Trajectory
    streams: dict[str, Stream] = field(default_factory=dict)
    gt: Stream | None = None

    def hashes(self) -> dict[str, str]:
        h = {k: s.sha256() for k, s in sorted(self.streams.items())}
        if self.gt is not None:
            h["gt"] = self.gt.sha256()
        return h


def _sensor_cfg(scn: dict, key: str, default: str) -> dict:
    ref = scn.get("sensors", {}).get(key, default)
    return cfgmod.load(ref) if isinstance(ref, str) else ref


def simulate(scn: dict, seed: int, surface_z=None, gt_hz: float = 100.0) -> SimOutput:
    surface_z = surface_z or flat_surface(scn.get("ground_z", 0.0))
    traj = trajectory.from_config(scn["trajectory"], ground_fn=surface_z)
    out = SimOutput(scn, seed, traj)
    enabled = scn.get("enable", ["imu", "baro", "mag", "gnss", "lidar", "flow"])
    faults = scn.get("faults", {})

    # ground truth
    tg = np.arange(traj.t0, traj.t1 + 1e-9, 1.0 / gt_hz)
    s = traj.sample(tg)
    out.gt = Stream("gt", "gt", s_to_ns(tg), {"p": s.p, "v": s.v, "q": s.q, "omega_b": s.omega_b, "f_b": s.f_b})

    if "imu" in enabled:
        ic = _sensor_cfg(scn, "imu", "sensors/imu_bmi160.yaml")
        ip = imu.ImuParams.from_config(ic)
        d = imu.simulate_imu(traj, ip, seed, noise=scn.get("imu_noise", True))
        out.streams["imu0"] = Stream.from_dict("imu0", "imu", d)
    if "baro" in enabled:
        bc = _sensor_cfg(scn, "baro", "sensors/baro_bmp390.yaml")
        tb = np.arange(traj.t0, traj.t1, 1.0 / bc["rate_hz"])
        sb = traj.sample(tb)
        elev = float(scn.get("site_elevation_m", 0.0))
        d = baro.simulate_baro(tb, sb.p[:, 2], bc, seed, elevation_offset_m=elev)
        if "baro_step" in faults:
            f = faults["baro_step"]
            d["alt_m"] = d["alt_m"] + np.where(tb >= f["t0"], f["step_m"], 0.0)
        out.streams["baro0"] = Stream.from_dict("baro0", "baro", d)
    if "mag" in enabled:
        mc = _sensor_cfg(scn, "mag", "sensors/mag_default.yaml")
        tm = np.arange(traj.t0, traj.t1, 1.0 / mc["rate_hz"])
        sm = traj.sample(tm)
        anomaly = None
        if "mag_anomaly" in faults:
            f = faults["mag_anomaly"]
            on = (tm >= f["t0"]) & (tm < f["t1"])
            anomaly = np.zeros((len(tm), 3))
            anomaly[on] = np.asarray(f["field_nt"], dtype=float)
        d = mag.simulate_mag(tm, sm.R, mc, seed, anomaly_nt=anomaly)
        out.streams["mag0"] = Stream.from_dict("mag0", "mag", d)
    if "gnss" in enabled:
        gc = _sensor_cfg(scn, "gnss", "sensors/gnss_m9n.yaml")
        tn = np.arange(traj.t0, traj.t1, 1.0 / gc["rate_hz"])
        sn = traj.sample(tn)
        nlos = None
        if "nlos" in scn:
            nl = scn["nlos"]
            on = (tn >= nl["t0"]) & (tn < nl["t1"])
            nlos = np.zeros((len(tn), 3))
            nlos[on] = np.asarray(nl["bias_m"], dtype=float)
        d = gnss.simulate_gnss(tn, sn.p, sn.v, gc, seed, jam=scn.get("jam"), spoof=scn.get("spoof"), traj=traj,
                               nlos_bias=nlos)
        out.streams["gnss0"] = Stream.from_dict("gnss0", "gnss", d)
    if "lidar" in enabled:
        lc = _sensor_cfg(scn, "lidar", "sensors/rangefinder_sf11c.yaml")
        tl = np.arange(traj.t0, traj.t1, 1.0 / lc["rate_hz"])
        sl = traj.sample(tl)
        dropout = tuple(faults["lidar_dropout"]) if "lidar_dropout" in faults else None
        d = rangefinder.simulate_rangefinder(tl, sl.p, sl.R, lc, seed, surface_z, dropout=dropout,
                                             bias_m=faults.get("lidar_bias_m", 0.0))
        out.streams["lidar0"] = Stream.from_dict("lidar0", "lidar", d)
    if "flow" in enabled:
        fc = _sensor_cfg(scn, "flow", "sensors/flow_px4flow.yaml")
        tf = np.arange(traj.t0 + 1.0 / fc["rate_hz"], traj.t1, 1.0 / fc["rate_hz"])
        d = flow.simulate_flow(traj, tf, fc, seed, surface_z, scale_error=faults.get("flow_scale_error", 0.0))
        out.streams["flow0"] = Stream.from_dict("flow0", "flow", d)
    return out


def load_scenario(path: str) -> dict:
    return cfgmod.load(path)
