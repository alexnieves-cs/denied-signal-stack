"""Run OpenVINS (via ovserver) on an ASL-format sequence (TUM-VI EuRoC export) with GT init."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from dss.core.config import REPO_ROOT
from dss.datasets.asl import AslSequence
from dss.eval import metrics
from dss.eval.associate import associate
from dss.vio.ovclient import OvClient

CONFIG_DIR = REPO_ROOT / "cpp" / "ovserver" / "config"


def load_gray8(path: Path) -> np.ndarray:
    """16-bit PNGs (TUM-VI 512_16) are mapped to 8 bit by range (>> 8); 8-bit images pass through."""
    a = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if a is None:
        raise FileNotFoundError(path)
    if a.ndim == 3:
        a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    if a.dtype == np.uint16:
        a = (a >> 8).astype(np.uint8)
    return a


@dataclass
class DatasetRun:
    name: str
    mode: str
    t: np.ndarray  # state times (s, IMU clock)
    p: np.ndarray
    q: np.ndarray
    cov6: np.ndarray
    n_tracked: np.ndarray
    n_msckf: np.ndarray
    status: np.ndarray
    frame_ms: np.ndarray
    wall_s: float
    duration_s: float
    rss_mb: list = field(default_factory=list)
    framing_errors: int = 0

    def log_bytes(self) -> bytes:
        return np.concatenate([self.t[:, None], self.p, self.q, self.cov6.reshape(len(self.t), 36)], 1).astype("<f8").tobytes()

    def log_sha256(self) -> str:
        return hashlib.sha256(self.log_bytes()).hexdigest()


def run_dataset(seq_root: str | Path, mode: str = "mono", init: str = "gt", config: str | Path | None = None,
                binary: str | None = None, max_frames: int | None = None, t_skip_s: float = 1.0,
                stderr_path: str | None = None, init_speed_mps: float = 0.2) -> DatasetRun:
    """GT init happens at motion onset (first GT speed > ``init_speed_mps``), as OpenVINS's own
    static initializer waits for excitation: MSCKF cannot triangulate during a stationary start and
    ZUPT is off in the shipped TUM-VI config."""
    if init != "gt":
        raise NotImplementedError("only GT init is supported")
    seq = AslSequence(seq_root)
    cfg = Path(config) if config else CONFIG_DIR / f"tum_vi_{mode}" / "estimator_config.yaml"
    imu = seq.imu()
    gt = seq.groundtruth()
    tc0, files0 = seq.camera("cam0")
    files1 = seq.camera("cam1")[1] if mode == "stereo" else None
    t_imu = imu.t_ns * 1e-9
    # first frame at least t_skip after both IMU and GT start
    t_start = max(t_imu[0], gt.t_ns[0] * 1e-9) + t_skip_s
    moving = np.nonzero(np.linalg.norm(gt["v"], axis=1) > init_speed_mps)[0]
    if len(moving):
        t_start = max(t_start, gt.t_ns[moving[0]] * 1e-9)
    k0 = int(np.searchsorted(tc0 * 1e-9, t_start))
    frames = range(k0, len(tc0) if max_frames is None else min(len(tc0), k0 + max_frames))
    t_first = tc0[k0] * 1e-9
    # GT init at the IMU sample just before the first frame
    i_init = int(np.searchsorted(t_imu, t_first)) - 1
    t_init = float(t_imu[i_init])
    g = int(np.argmin(np.abs(gt.t_ns * 1e-9 - t_init)))
    out = {k: [] for k in ("t", "p", "q", "cov6", "n_tracked", "n_msckf", "status", "ms")}
    rss = []
    wall0 = time.perf_counter()
    with OvClient(cfg, binary=binary, stderr_path=stderr_path) as c:
        c.init()
        ii = 0
        while ii <= i_init:
            c.imu(float(t_imu[ii]), imu["gyro"][ii], imu["acc"][ii])
            ii += 1
        c.init_gt(t_init, gt["q"][g], gt["p"][g], gt["v"][g])
        for n, k in enumerate(frames):
            tc = tc0[k] * 1e-9
            while ii < len(t_imu) and t_imu[ii] <= tc + 0.02:
                c.imu(float(t_imu[ii]), imu["gyro"][ii], imu["acc"][ii])
                ii += 1
            if ii >= len(t_imu):
                break
            img0 = load_gray8(files0[k])
            t0 = time.perf_counter()
            if files1 is not None:
                s = c.cam_stereo(tc, img0, load_gray8(files1[k]))
            else:
                s = c.cam(tc, img0)
            out["ms"].append(1e3 * (time.perf_counter() - t0))
            out["status"].append(s.status)
            if s.ok:
                out["t"].append(s.t)
                out["p"].append(s.p)
                out["q"].append(s.q)
                out["cov6"].append(s.cov6)
                out["n_tracked"].append(s.n_tracked)
                out["n_msckf"].append(s.n_msckf)
            if n % 200 == 0:
                rss.append((tc - t_first, c.rss_mb()))
        rss.append((tc - t_first, c.rss_mb()))
        fe = c.framing_errors
    wall = time.perf_counter() - wall0
    return DatasetRun(name=Path(seq_root).name, mode=mode, t=np.array(out["t"]), p=np.array(out["p"]), q=np.array(out["q"]),
                      cov6=np.array(out["cov6"]), n_tracked=np.array(out["n_tracked"]), n_msckf=np.array(out["n_msckf"]),
                      status=np.array(out["status"]), frame_ms=np.array(out["ms"]), wall_s=wall,
                      duration_s=float(tc - t_first), rss_mb=rss, framing_errors=fe)


def evaluate(run: DatasetRun, seq_root: str | Path) -> dict:
    gt = AslSequence(seq_root).groundtruth()
    ia, ib = associate((run.t * 1e9).astype(np.int64), gt.t_ns, 0.01)
    pe, pg = run.p[ia], gt["p"][ib]
    a = metrics.ate(pe, pg, "posyaw")
    r = metrics.rpe(a["aligned"], pg, lengths=(8, 16, 24, 32, 40))
    # divergence: nan, or >1 m error (datasets)
    diverged = bool(~np.isfinite(run.p).all() or a["max_m"] > 1.0 and a["rmse_m"] > 1.0)
    return {
        "ate_rmse_m": a["rmse_m"], "ate_median_m": a["median_m"], "ate_max_m": a["max_m"], "path_m": a["path_m"],
        "drift_pct": a["drift_pct"], "rpe": {str(k): v for k, v in r.items()},
        "n_assoc": int(len(ia)), "n_frames": int(len(run.status)), "n_ok": int((run.status == 0).sum()),
        "frame_ms_p50": float(np.percentile(run.frame_ms, 50)), "frame_ms_p99": float(np.percentile(run.frame_ms, 99)),
        "rtf": float(run.duration_s / run.wall_s), "rss_mb_max": float(max(r_[1] for r_ in run.rss_mb)),
        "framing_errors": run.framing_errors, "diverged": diverged,
        "mean_n_tracked": float(run.n_tracked.mean()) if len(run.n_tracked) else 0.0,
        "_aligned": a["aligned"], "_gt": pg,
    }


# --- Phase 1 dataset gates ----------------------------------------------------------------------
TUMVI_DIR = REPO_ROOT / "data" / "tumvi"
REPORT_DIR = REPO_ROOT / "reports" / "phase1"


def available_rooms() -> list[str]:
    out = []
    for n in range(1, 7):
        d = TUMVI_DIR / f"dataset-room{n}_512_16" / "mav0"
        if (d / "mocap0" / "data.csv").exists() and (d / "cam1" / "data.csv").exists() and (d / "imu0" / "data.csv").exists():
            # a streaming extract writes cam0 images first; require the last listed image to exist
            ok = (d / "cam0" / "data.csv").exists()
            for cam in ("cam0", "cam1"):
                if ok:
                    _, files = AslSequence(d.parent).camera(cam)
                    ok = bool(files) and files[-1].exists()
            if ok:
                out.append(f"room{n}")
    return out


def gates_report(rooms: list[str] | None = None) -> dict:
    import json

    from dss.eval import plots
    from dss.eval.report import _clean, md_table

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rooms = rooms or available_rooms()
    rows, res = [], {}
    for room in rooms:
        root = TUMVI_DIR / f"dataset-{room}_512_16"
        for mode in ("mono", "stereo"):
            run = run_dataset(root, mode)
            e = evaluate(run, root)
            plots.trajectory_plot(REPORT_DIR / f"tumvi_{room}_{mode}_traj.png", {"GT": e["_gt"], "est (posyaw)": e["_aligned"]},
                                  f"OpenVINS {mode}, TUM-VI {room}: ATE {e['ate_rmse_m']:.3f} m")
            e = {k: v for k, v in e.items() if not k.startswith("_")}
            e["log_sha256"] = run.log_sha256()
            res[f"{room}/{mode}"] = e
            rows.append({"seq": room, "mode": mode, "ATE m": e["ate_rmse_m"], "drift %": e["drift_pct"],
                         "RPE8 %": e["rpe"]["8"]["pct"], "p50 ms": e["frame_ms_p50"], "p99 ms": e["frame_ms_p99"],
                         "RTF": e["rtf"], "RSS MB": e["rss_mb_max"], "diverged": e["diverged"]})
    # determinism: second full room1 mono run
    det = None
    if "room1" in rooms:
        r2 = run_dataset(TUMVI_DIR / "dataset-room1_512_16", "mono")
        det = r2.log_sha256() == res["room1/mono"]["log_sha256"]
    mono = [res[f"{r}/mono"]["ate_rmse_m"] for r in rooms if r != "room6"]
    stereo = [res[f"{r}/stereo"]["ate_rmse_m"] for r in rooms if r != "room6"]
    gates = {
        "exit1_protocol": {"pass": all(v["framing_errors"] == 0 for v in res.values()),
                           "note": "10k frames @ verbosity ALL, malformed-frame, kill->VioFailure, schema: tests/vio/test_protocol.py"},
        "exit2_determinism": {"pass": bool(det), "note": "two full TUM-VI room1 mono runs, byte-identical pose+cov log"},
        "exit3_euroc_stereo": {"pass": None, "note": "CUT: EuRoC unavailable (ETH HTTP 429 / timeout)"},
        "exit4_euroc_mono": {"pass": None, "note": "CUT: EuRoC unavailable (ETH HTTP 429 / timeout)"},
        "exit5_drift": {"pass": all(v["drift_pct"] <= 1.0 for v in res.values()),
                        "max_drift_pct": max(v["drift_pct"] for v in res.values()),
                        "note": f"TUM-VI {', '.join(rooms)} only; corridor4 and missing rooms not fetched"},
        "exit6_tumvi_rooms": {"pass": bool(np.mean(mono) <= 0.10 and np.mean(stereo) <= 0.12), "mono_avg_m": float(np.mean(mono)),
                              "stereo_avg_m": float(np.mean(stereo)), "rooms": rooms,
                              "note": "needs rooms 1-5; partial if fewer are available"},
        "exit7_timing": {"pass": bool(max(v["frame_ms_p99"] for k, v in res.items() if k.endswith("mono")) <= 50
                                      and min(v["rtf"] for v in res.values()) >= 1.0),
                         "note": "proxy on TUM-VI (EuRoC V1_01 unavailable); release build, lockstep single-thread; "
                                 "RSS slope over a 25-min soak is a sim gate (not run here)"},
    }
    out = {"runs": res, "gates": gates}
    (REPORT_DIR / "vio_gates.json").write_text(json.dumps(_clean(out), indent=2, sort_keys=True) + "\n")
    lines = ["# Phase 1 VIO dataset gates (TUM-VI)", "",
             "OpenVINS @69488123 via ovserver (release, lockstep, `num_opencv_threads: 0`), shipped `tum_vi` config, "
             "GT init at motion onset (GT speed > 0.2 m/s), posyaw ATE vs mocap0, association 0.01 s.", "",
             md_table(rows, ["seq", "mode", "ATE m", "drift %", "RPE8 %", "p50 ms", "p99 ms", "RTF", "RSS MB", "diverged"]), "",
             "| gate | result | detail |", "|---|---|---|"]
    for k, v in gates.items():
        tag = "CUT" if v["pass"] is None else ("PASS" if v["pass"] else "FAIL")
        extra = {kk: vv for kk, vv in v.items() if kk not in ("pass", "note")}
        lines.append(f"| {k} | **{tag}** | {v['note']} {json.dumps(_clean(extra)) if extra else ''} |")
    lines += ["", f"Determinism (room1 mono, 2 full runs): {'identical' if det else 'DIFFERENT'} "
              f"(sha256 {res['room1/mono']['log_sha256'][:16]}…)" if "room1" in rooms else ""]
    (REPORT_DIR / "vio_gates.md").write_text("\n".join(lines) + "\n")
    return out


if __name__ == "__main__":
    import sys

    o = gates_report(sys.argv[1:] or None)
    for k, v in o["gates"].items():
        print(k, v)
