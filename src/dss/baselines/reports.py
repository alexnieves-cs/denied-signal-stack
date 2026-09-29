"""Phase 0 dead-reckoning reports: sim route, TUM-VI room1, EuRoC MH_01 (if fetched)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from dss.baselines import dead_reckoning as dr
from dss.core import config
from dss.core.manifest import build_manifest, git_sha
from dss.core.rotations import quat_to_rot
from dss.datasets import fetch
from dss.datasets.asl import AslSequence
from dss.eval import plots
from dss.eval.report import write_metrics
from dss.fusion.eskf import ImuNoise
from dss.sim import mag as magmod
from dss.sim import runner

REPORTS = config.REPO_ROOT / "reports" / "phase0"


def sim_report(scn_path: str, seed: int, out: Path) -> dict:
    scn = config.load(scn_path)
    so = runner.simulate(scn, seed)
    ic = config.load(scn["sensors"]["imu"])
    imu = so.streams["imu0"]
    gt = so.gt
    x0 = dr.state_from_gt(gt, int(imu.t_ns[0]))
    res = {}
    field_w = magmod.field_enu(config.load("sensors/mag_default.yaml"))
    variants = {
        "dr": {},
        "dr+baro+mag": {"baro": so.streams.get("baro0"), "baro_var": 1.2**2,
                         "mag_yaw": (so.streams["mag0"].t_ns, so.streams["mag0"]["field_nt"], field_w)},
    }
    series, errs = {"GT": gt["p"]}, {}
    for name, kw in variants.items():
        r = dr.run(imu, x0, dr.default_P0(), ImuNoise.from_config(ic), out_hz=10.0, **kw)
        i = np.searchsorted(gt.t_ns, r.t_ns).clip(0, len(gt) - 1)
        e = r.p - gt["p"][i]
        res[name] = {"final_err_m": float(np.linalg.norm(e[-1])), "err_60s_m": float(np.linalg.norm(e[min(600, len(e) - 1)])),
                     "rmse_m": float(np.sqrt(np.mean(np.sum(e**2, 1)))), "vert_rmse_m": float(np.sqrt(np.mean(e[:, 2] ** 2)))}
        series["DR " + name] = r.p
        errs[name] = (r.t_ns * 1e-9, np.linalg.norm(e, axis=1))
    plots.trajectory_plot(out / f"dr_{scn['name']}_traj.png", series, f"Dead reckoning, {scn['name']} seed {seed}")
    t = errs["dr"][0]
    plots.error_plot(out / f"dr_{scn['name']}_err.png", t, {k: v[1] for k, v in errs.items()},
                     f"DR 3-D error, {scn['name']}", ylabel="3-D error [m]", logy=True)
    return res


def dataset_report(key: str, out: Path, seconds: float = 60.0) -> dict | None:
    root = fetch.path(key)
    if not (root / "mav0" / "imu0" / "data.csv").exists():
        return None
    seq = AslSequence(root)
    imu = seq.imu()
    gt = seq.groundtruth()
    ic = config.load("sensors/imu_bmi160.yaml" if key.startswith("tumvi") else "sensors/imu_adis16448.yaml")
    t0 = max(int(imu.t_ns[0]), int(gt.t_ns[0])) + int(1e9)
    i = int(np.searchsorted(gt.t_ns, t0))
    # datasets: GT is the body (mocap/IMU) pose; TUM-VI mocap is the IMU frame after its T_imu_mocap
    x0 = {"t_ns": int(gt.t_ns[i]), "p": gt["p"][i], "v": gt["v"][i], "R": quat_to_rot(gt["q"][i]),
          "bg": np.zeros(3), "ba": np.zeros(3)}
    r = dr.run(imu, x0, dr.default_P0(), ImuNoise.from_config(ic), t_end_ns=x0["t_ns"] + int(seconds * 1e9), out_hz=20.0)
    j = np.searchsorted(gt.t_ns, r.t_ns).clip(0, len(gt) - 1)
    e = np.linalg.norm(r.p - gt["p"][j], axis=1)
    name = key.replace(":", "_")
    t = (r.t_ns - r.t_ns[0]) * 1e-9
    plots.trajectory_plot(out / f"dr_{name}_traj.png", {"GT": gt["p"][j], "DR": r.p}, f"Dead reckoning, {key} ({seconds:.0f} s, GT init)")
    plots.error_plot(out / f"dr_{name}_err.png", t, {"DR": e}, f"DR error, {key}", ylabel="3-D error [m]", logy=True)
    return {"seconds": seconds, "err_10s_m": float(e[min(200, len(e) - 1)]), "final_err_m": float(e[-1]),
            "note": "raw IMU, zero initial bias; consumer IMU DR diverges quadratically, as expected"}


def main(a) -> int:
    out = Path(a.out) if a.out else REPORTS
    out.mkdir(parents=True, exist_ok=True)
    summary = {"manifest": build_manifest(a.seed, {"cmd": "baseline dr"}, {"git_sha": git_sha()})}
    if a.scenario:
        summary["sim"] = sim_report(a.scenario, a.seed, out)
    if a.dataset:
        summary[a.dataset] = dataset_report(a.dataset, out)
    if not a.scenario and not a.dataset:
        summary["sim"] = sim_report("scenarios/dr_route_2km.yaml", a.seed, out)
        for k in ("tumvi:room1", "euroc:MH_01"):
            summary[k] = dataset_report(k, out)
    write_metrics(out / "dr_metrics.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "manifest"}, indent=1, default=str))
    return 0
