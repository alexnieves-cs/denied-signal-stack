"""End-to-end lockstep pipeline: sim sensors + rendered camera -> VIO -> REF/ALL fusion -> integrity -> output.

`dss demo` / `dss run` / `dss suite` enter here. Deterministic in --lockstep: metrics.json has no wall-clock fields.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dss.aiding.match_ncc import NccMatcher
from dss.aiding.ortho import orthorectify_dem
from dss.baselines.dead_reckoning import mag_yaw_fn_impl
from dss.core import config
from dss.core.config import REPO_ROOT
from dss.core.manifest import build_manifest, git_sha, sha256_file
from dss.core.rng import rng
from dss.core.rotations import quat_to_rot, rot_to_quat
from dss.fusion.backend import NavFilter
from dss.fusion.eskf import ImuNoise
from dss.integrity import gnss_gate as gg
from dss.integrity.health import Health
from dss.integrity.separation import Publisher, clamp_target, separation_stat
from dss.integrity.vision import VisionMonitor
from dss.sensors.rig import R_BODY_NADIR_CAM
from dss.sim import mag as magmod
from dss.sim import runner

P_CAM_IN_BODY = np.array([0.0, 0.0, -0.05])
WIDTH, HEIGHT, FOCAL = 640, 480, 400.0
_WORLD: dict = {}


# ------------------------------------------------------------------------------------------------ world
def world(aoi: str = "flagstaff", render_epoch: int = 2023, map_epoch="2017_coreg"):
    key = (aoi, render_epoch, map_epoch)
    if key not in _WORLD:
        from dss.geo import naip
        from dss.geo.aoi import AOIS

        a = AOIS[aoi]
        img, info_r = naip.load_image(aoi, render_epoch)
        p_map = naip.GEO / aoi / f"naip_{map_epoch}.npz"
        if not p_map.exists() and str(map_epoch).endswith("_coreg"):
            from dss.geo.coreg import coregistered_map_ncc

            mp_, info_, _, _ = coregistered_map_ncc(aoi, int(str(map_epoch)[:4]), render_epoch)
            np.savez_compressed(p_map, rgbn=mp_.data, res=mp_.res, xmin=mp_.xmin, ymax=mp_.ymax,
                                item=info_["item"] + "+coreg", date=info_["date"])
        mp, info_m = naip.load_image(aoi, map_epoch)
        dem = naip.load_dem(aoi)
        _WORLD[key] = {"aoi": a, "img": img, "map": mp, "dem": dem, "info": {"render": info_r, "map": info_m},
                       "matcher": NccMatcher(mp)}
    return _WORLD[key]


def renderer(w: dict):
    if "renderer" not in w:
        from dss.sim.render.camera import PinholeRenderer
        from dss.sim.render.scene import build_model

        a = w["aoi"]
        extra = w.get("extra_geoms")
        m = build_model(w["img"], w["dem"], (a.xmin, a.xmax, a.ymin, a.ymax), WIDTH, HEIGHT, FOCAL, extra_geoms=extra)
        w["renderer"] = PinholeRenderer(m, WIDTH, HEIGHT, FOCAL)
    return w["renderer"]


# ------------------------------------------------------------------------------------------------ VIO adapters
@dataclass
class VioOut:
    ok: bool
    p: np.ndarray
    v: np.ndarray
    R: np.ndarray
    cov6: np.ndarray
    n_tracked: int
    tracks: np.ndarray | None = None


class SyntheticVioAdapter:
    """VIO stand-in from ground truth: velocity error = slowly varying bias (random walk) + white noise,
    with feature count driven by image texture (so vision collapse is still observable)."""

    name = "synthetic"

    def __init__(self, seed: int, drift_frac: float = 0.005, white_v: float = 0.03):
        self.g = rng(seed, "vio", "synthetic")
        self.bias = np.zeros(3)
        self.drift = drift_frac
        self.white = white_v
        self.p = None
        self.R_err = np.eye(3)
        self.t = None
        self.alive = False

    def init_gt(self, t, p, v, R, bg, ba, P15=None):
        self.p = np.array(p, float)
        self.t = t
        self.bias[:] = 0
        self.alive = True

    reanchor = init_gt

    def imu(self, t, w, a):
        pass

    def cam(self, t, gray, truth) -> VioOut:
        if not self.alive:
            return VioOut(False, np.zeros(3), np.zeros(3), np.eye(3), np.eye(6), 0)
        dt = t - self.t
        self.t = t
        speed = float(np.linalg.norm(truth["v"]))
        # Gauss-Markov velocity bias, stationary sigma = drift_frac * speed, tau 30 s
        tau = 30.0
        phi = np.exp(-dt / tau)
        self.bias = phi * self.bias + self.g.normal(0, self.drift * max(speed, 1.0) * np.sqrt(1 - phi**2), 3)
        v = truth["v"] + self.bias + self.g.normal(0, self.white, 3)
        self.p = self.p + v * dt
        lv = float(np.std(gray)) if gray is not None else 50.0
        n = int(np.clip(200 * min(1.0, lv / 25.0), 0, 200))
        return VioOut(n >= 40, self.p.copy(), v, truth["R"], np.eye(6) * 0.5, n)

    def close(self):
        pass


class OvserverAdapter:
    """Drives cpp/ovserver (OpenVINS MSCKF, GPL process boundary) via dss.vio.ovclient."""

    name = "ovserver"

    def __init__(self, cfg_path: str):
        from dss.vio.ovclient import OvClient

        self.c = OvClient(cfg_path)
        self.c.init()
        self.started = False

    def init_gt(self, t, p, v, R, bg, ba, P15=None):
        q = rot_to_quat(R)
        if P15 is None:
            self.c.init_gt(t, q, p, v, bg, ba)
        else:
            self.c.reanchor(t, q, p, v, bg, ba, P15)
        self.started = True

    def reanchor(self, t, p, v, R, bg, ba, P15):
        self.c.reanchor(t, rot_to_quat(R), p, v, bg, ba, P15)

    def imu(self, t, w, a):
        self.c.imu(t, w, a)

    def cam(self, t, gray, truth) -> VioOut:
        s = self.c.cam(t, gray)
        ok = s.status == 0 and s.p is not None and np.all(np.isfinite(s.p))
        if not ok:
            return VioOut(False, np.zeros(3), np.zeros(3), np.eye(3), np.eye(6), int(s.n_tracked or 0))
        return VioOut(True, np.asarray(s.p), np.asarray(s.v), quat_to_rot(np.asarray(s.q)), np.asarray(s.cov6),
                      int(s.n_tracked), getattr(s, "tracks", None))

    def close(self):
        self.c.close()


def make_vio(mode: str, seed: int):
    cfg = REPO_ROOT / "cpp/ovserver/config/sim_nadir/estimator_config.yaml"
    if mode in ("auto", "ovserver"):
        try:
            return OvserverAdapter(str(cfg))
        except Exception as e:
            if mode == "ovserver":
                raise
            print(f"[vio] ovserver unavailable ({e!r}); using synthetic VIO")
    return SyntheticVioAdapter(seed)


# ------------------------------------------------------------------------------------------------ run
def _P0_ref(c: dict) -> np.ndarray:
    return np.diag([c.get("home_sigma_h", 1.0) ** 2] * 2 + [c.get("home_sigma_z", 0.5) ** 2] + [0.05**2] * 3
                   + [np.deg2rad(0.5) ** 2] * 2 + [np.deg2rad(c.get("home_sigma_yaw_deg", 2.0)) ** 2]
                   + [1e-4**2] * 3 + [0.02**2] * 3)


def _to_ov_P15(P: np.ndarray) -> np.ndarray:
    """Our error order [p, v, theta, bg, ba] -> [theta, p, v, bg, ba] (the client converts theta conventions)."""
    idx = np.r_[6:9, 0:3, 3:6, 9:12, 12:15]
    return P[np.ix_(idx, idx)]


def _reanchor_P15(P: np.ndarray) -> np.ndarray:
    """Re-anchor covariance: REF marginals with velocity / attitude / bias blocks clamped.

    Finding (LEDGER Phase 1): a fresh mono MSCKF at ~100 m depth given REF's loose velocity/attitude prior
    converges to a wrong velocity (0.4-0.8 m/s); with a tight prior it holds (~0.05 m/s). So the VIO is
    seeded tightly around REF's state, and the fusion side carries REF's velocity uncertainty at the re-anchor
    as a VIO velocity bias term (see ``reanchor_bias_sigma``) until a refresh re-anchor after map fixes.
    """
    Q = np.diag(np.r_[np.clip(np.diag(P)[0:3], 1e-4, None), [0.03**2] * 3, [np.deg2rad(0.3) ** 2] * 3, [1e-5**2] * 3,
                      [0.005**2] * 3])
    return _to_ov_P15(Q)


def run(scn: dict, seed: int, out_dir: Path, vio_mode: str = "auto", rrd: bool = False, verbose: bool = True) -> dict:
    wall0 = time.time()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    W = world()
    dem, matcher = W["dem"], W["matcher"]
    so = runner.simulate(scn, seed, surface_z=dem.sample)
    traj = so.traj
    st = so.streams
    imu, baro, mag, gnss, lidar = st["imu0"], st["baro0"], st["mag0"], st.get("gnss0"), st["lidar0"]
    flow = st.get("flow0")
    ifl = 0
    icfg = config.load(scn["sensors"]["imu"])
    noise = ImuNoise.from_config(icfg)
    fcfg = scn.get("fusion", {})
    cam_cfg = scn.get("camera", {})
    events = cam_cfg.get("events", [])
    field_w = magmod.field_enu(config.load("sensors/mag_default.yaml"))
    F_nom = float(np.linalg.norm(field_w))
    incl_nom = float(np.arcsin(-field_w[2] / F_nom))

    # --- initial state: surveyed home pose (REF never sees GPS)
    g_init = rng(seed, "scenario", "home")
    s0 = traj.sample([0.0])
    P0 = _P0_ref(fcfg)
    dx = g_init.multivariate_normal(np.zeros(15), P0)
    from dss.core.rotations import so3_exp

    x0 = {"t_ns": int(imu.t_ns[0]), "p": s0.p[0] + dx[0:3], "v": dx[3:6], "R": s0.R[0] @ so3_exp(dx[6:9]),
          "bg": dx[9:12], "ba": dx[12:15]}
    ref = NavFilter("REF", x0, P0, noise, fcfg)
    allf = NavFilter("ALL", x0, P0, noise, fcfg)
    gate = gg.GnssGate(gg.GateConfig.from_dict(scn.get("gate")))
    health = Health()
    vmon = VisionMonitor()
    pub = Publisher(x0["p"], x0["R"], P0[:3, :3])
    from dss.output.mavlink_vpe import VpeEncoder

    vpe = VpeEncoder()
    vpe_frames = 0
    vpe_bad = 0
    vpe_file = open(out_dir / "vpe.mavlink", "wb")

    vio = make_vio(vio_mode, seed)
    rend = renderer(W) if cam_cfg.get("enabled", True) else None
    from dss.sim.render.camera import to_gray
    from dss.sim.render.photometric import Photometric

    photo = Photometric(WIDTH, HEIGHT, cam_cfg.get("photometric"))
    g_cam = rng(seed, "camera", "noise")
    g_reanchor = rng(seed, "reanchor", "gt_proxy")
    cam_hz = float(cam_cfg.get("rate_hz", 20.0))
    cam_every = int(round(imu_rate(imu) / cam_hz))
    vio_every = int(fcfg.get("vio_every_frames", 4))
    sig_vio = float(fcfg.get("vio_sigma_v", 0.12))
    map_hz = float(fcfg.get("map_hz", 1.0))
    map_enabled = bool(fcfg.get("map_aiding", True))
    vio_init_t = float(fcfg.get("vio_init_t", 1.0))

    t_s = imu.t_ns * 1e-9
    gyro, acc = imu["gyro"], imu["acc"]
    ib = im = ig = il = 0
    frame_k = 0
    pending_frame = None
    vio_started = False
    vio_failed_since = None
    need_reanchor = False
    reanchor_t = None
    reanchor_bias_sigma = 0.0
    fixes_since_reanchor = 0
    refresh_pending = False
    vio_rej_run = 0
    last_vio_fail_t = -1e9
    last_map_t = -1e9
    last_pub_t = -1e9
    next_pub_t = 0.0
    last_log_t = -1e9
    last_fix_accept_t = None
    fixes = []
    log = {k: [] for k in ("t", "gt", "ref", "all", "pub", "Pref", "Pall", "Ppub", "hpl", "gate", "vision", "ntrk", "gps",
                           "gps_valid", "spoof_off", "pub_jump", "vio_p", "illum", "map_state", "gt_v", "pub_R", "gt_R",
                           "bleeding", "lapvar", "ref_v", "Pref_v")}
    vpe_log = {"t": [], "p": [], "P": [], "dp_prop": []}
    rr = None
    if rrd:
        from dss.dashboard.rerun_sink import RerunSink

        rr = RerunSink(out_dir / "run.rrd", scn, seed)
    last_vio: VioOut | None = None
    prev_vio_p = None
    prev_vio_t = None
    mag_state = "OK"
    timing = {"render": 0.0, "vio": 0.0, "match": 0.0, "fusion": 0.0}

    for k in range(1, len(t_s)):
        tk = float(t_s[k])
        tn = int(imu.t_ns[k])
        w_m = 0.5 * (gyro[k - 1] + gyro[k])
        a_m = 0.5 * (acc[k - 1] + acc[k])
        t0 = time.perf_counter()
        ref.propagate(w_m, a_m, tn)
        allf.propagate(w_m, a_m, tn)
        timing["fusion"] += time.perf_counter() - t0
        vio.imu(tk, gyro[k], acc[k])

        # --- VIO GT init at vio_init_t (static, before takeoff)
        if not vio_started and tk >= vio_init_t:
            s = traj.sample([tk])
            vio.init_gt(tk, s.p[0], s.v[0], s.R[0], np.zeros(3), np.zeros(3))
            vio_started = True

        # --- process the frame captured 5 IMU samples ago (ovserver needs IMU through t_cam + 0.02 s)
        if pending_frame is not None and k >= pending_frame["k"] + 5:
            fr = pending_frame
            pending_frame = None
            t0 = time.perf_counter()
            vo = vio.cam(fr["t"], fr["gray"], fr["truth"]) if vio_started else None
            timing["vio"] += time.perf_counter() - t0
            ntrk = vo.n_tracked if vo is not None else None
            vs = vmon.step(fr["raw_mean"], fr["gray"], ntrk)
            prev_state = health.state["VIO"]
            health.set(fr["t"], "VIO", vs["state"], ",".join(k2 for k2 in ("collapse", "blur", "shock", "dark") if vs[k2]))
            if vs["state"] == "FAILED":
                if vio_failed_since is None:
                    vio_failed_since = fr["t"]
                need_reanchor = vio_started and fr["t"] > vio_init_t + 1.0
                prev_vio_p = None
            elif fcfg.get("force_reanchor_t") and abs(fr["t"] - fcfg["force_reanchor_t"]) < 0.026:
                src = traj.sample([fr["t"]]) if fcfg.get("force_reanchor_truth") else None
                if src is not None:
                    Pt = _to_ov_P15(ref.P) if not fcfg.get("force_reanchor_tight") else np.diag(
                        [1e-4] * 3 + [1e-4] * 3 + [1e-4] * 3 + [1e-6] * 3 + [1e-4] * 3)
                    vio.reanchor(fr["t"], src.p[0], src.v[0], src.R[0], np.zeros(3), np.zeros(3), Pt)
                else:
                    vio.reanchor(fr["t"], ref.p, ref.v, ref.R, ref.f.bg, ref.f.ba, _to_ov_P15(ref.P))
                reanchor_t = fr["t"]
                prev_vio_p = None
            elif prev_state == "FAILED" and vs["state"] != "FAILED" and need_reanchor:
                # recovery: fresh VIO seeded from REF's state and marginals (never ALL / published)
                if fcfg.get("reanchor_source", "gt_proxy") == "ref":
                    vio.reanchor(fr["t"], ref.p, ref.v, ref.R, ref.f.bg, ref.f.ba, _reanchor_P15(ref.P))
                    ref.f.reset_vio_bias_correlated()
                    allf.f.reset_vio_bias_correlated()
                    refresh_pending = True
                else:
                    # CUT (LEDGER Phase 1/2, failure rule): REF-derived re-anchor of a fresh mono MSCKF at ~100 m
                    # diverged in 3 fix attempts; use PLAN 1.5's GT-derived anchor x = GT (+) d, d ~ N(0, P15_proxy).
                    gs = traj.sample([fr["t"]])
                    d = g_reanchor.normal(size=12) * np.r_[[2.4, 2.4, 1.0], [0.2] * 3, np.deg2rad([0.5, 0.5, 1.0]),
                                                            [0.0] * 3][:12]
                    from dss.core.rotations import so3_exp as _exp

                    Pp = np.diag(np.r_[[2.4**2, 2.4**2, 1.0], [0.2**2] * 3, np.deg2rad([0.5, 0.5, 1.0]) ** 2,
                                       [1e-5**2] * 3, [0.005**2] * 3])
                    vio.reanchor(fr["t"], gs.p[0] + d[0:3], gs.v[0] + d[3:6], gs.R[0] @ _exp(d[6:9]), np.zeros(3),
                                 np.zeros(3), _reanchor_P15(Pp))
                    for f_ in (ref.f, allf.f):  # VIO bias now independent of REF: reset with the proxy velocity sigma
                        f_.b_aug[:] = 0.0
                        f_.P[15:18, :] = 0.0
                        f_.P[:, 15:18] = 0.0
                        f_.P[15:18, 15:18] = np.eye(3) * 0.2**2
                    refresh_pending = False
                reanchor_t = fr["t"]  # noqa: F841  (kept for debugging hooks)
                reanchor_bias_sigma = 0.0
                fixes_since_reanchor = 0
                need_reanchor = False
                vio_failed_since = None
                prev_vio_p = None
            elif vo is not None and vo.ok and vs["state"] != "FAILED":
                frame_k += 1
                last_vio = vo
                if prev_vio_p is not None and frame_k % vio_every == 0:
                    dtv = fr["t"] - prev_vio_t
                    v_meas = (vo.p - prev_vio_p) / dtv
                    t0 = time.perf_counter()
                    sv = sig_vio * (2.0 if vs["state"] == "DEGRADED" else 1.0)
                    if abs(float(ref.v[2])) > 1.0:
                        sv *= 3.0  # climb / descent: depth changes fast and mono VIO velocity degrades
                    # high dynamics (horizontal acceleration, fast yaw): mono VIO velocity error grows; de-weight
                    a_w = ref.R @ (acc[k] - ref.f.ba) + np.array([0.0, 0.0, -9.81])
                    w_z = abs(float((ref.R @ (gyro[k] - ref.f.bg))[2]))
                    dyn = max(float(np.linalg.norm(a_w[:2])) / 0.5, w_z / 0.3)
                    if dyn > 1.0:
                        sv *= min(1.0 + dyn, 5.0)
                    if refresh_pending:
                        sv = float(np.hypot(sv, reanchor_bias_sigma))  # VIO inherited REF's velocity error
                    ok_v = ref.vio_velocity(v_meas, sv)
                    allf.vio_velocity(v_meas, sv)
                    vio_rej_run = 0 if ok_v else vio_rej_run + 1
                    if vio_rej_run >= 5:
                        # VIO inconsistent with REF (IMU + aiding): declare it failed and re-anchor it
                        health.set(fr["t"], "VIO", "FAILED", "velocity inconsistent with REF (5 consecutive gate failures)")
                        vmon.state = "FAILED"
                        vmon.good_run = 0
                        need_reanchor = True
                        vio_rej_run = 0
                    timing["fusion"] += time.perf_counter() - t0
                if prev_vio_p is None or frame_k % vio_every == 0:
                    prev_vio_p, prev_vio_t = vo.p.copy(), fr["t"]
            # --- map fix
            illum_ok = fr["raw_mean"] >= float(fcfg.get("map_min_mean", 40.0))
            health.set(fr["t"], "MAP", health.state["MAP"] if (illum_ok and map_enabled) else "DISABLED",
                       "" if illum_ok else "illumination below threshold")
            if map_enabled and illum_ok and fr["t"] - last_map_t >= 1.0 / map_hz and vs["state"] != "FAILED":
                last_map_t = fr["t"]
                t0 = time.perf_counter()
                agl = fr["agl_est"]
                if agl > 40.0:
                    Rc = ref.R @ R_BODY_NADIR_CAM
                    pc = ref.p + ref.R @ P_CAM_IN_BODY
                    patch, valid = orthorectify_dem(fr["gray"], rend.K, Rc, pc, dem, pc[:2], 96.0, matcher.mp.res)
                    sig_ref = float(np.sqrt(np.max(np.linalg.eigvalsh(ref.pos_cov()[:2, :2]))))
                    search = float(np.clip(3 * sig_ref + 10.0, 20.0, 150.0))
                    mres = matcher.match(patch, valid, pc[:2], search)
                    true_c = fr["truth"]["p"][:2] + (fr["truth"]["R"] @ P_CAM_IN_BODY)[:2]
                    post_outage = last_fix_accept_t is None or fr["t"] - last_fix_accept_t > 15.0
                    strong = mres.peak_ratio >= float(fcfg.get("post_outage_min_ratio", 1.8))
                    if mres.ok and (strong or not post_outage):
                        fix = ref.p[:2] + (mres.xy - pc[:2])
                        tilt_var = (agl * np.deg2rad(0.3)) ** 2
                        # map registration error is spatially correlated (successive fixes share it):
                        # floor it and de-weight each 1 Hz fix accordingly
                        cov = (mres.cov + np.eye(2) * (tilt_var + float(fcfg.get("map_reg_sigma", 2.0)) ** 2)) \
                            * float(fcfg.get("map_corr_inflation", 4.0))
                        need2 = 3 if post_outage else 1
                        pre_p = ref.p[:2].copy()
                        acc_r, nis = ref.map_fix(fix, cov, need2, quality=mres.peak_ratio)
                        if acc_r:
                            allf.map_fix(fix, cov, 1, quality=mres.peak_ratio)
                            fixes_since_reanchor += 1
                            if refresh_pending and fixes_since_reanchor >= 3 and vs["state"] == "OK":
                                # refresh re-anchor: REF's velocity is now re-observed through the fixes
                                vio.reanchor(fr["t"], ref.p, ref.v, ref.R, ref.f.bg, ref.f.ba, _reanchor_P15(ref.P))
                                ref.f.reset_vio_bias_correlated()
                                allf.f.reset_vio_bias_correlated()
                                refresh_pending = False
                                prev_vio_p = None
                                health.set(fr["t"], "VIO", "OK", "refresh re-anchor after 3 map fixes")
                            last_fix_accept_t = fr["t"]
                            health.set(fr["t"], "MAP", "OK")
                        err_fix = float(np.linalg.norm(fix - fr["truth"]["p"][:2]))
                        fixes.append({"t": fr["t"], "accepted": bool(acc_r), "err": err_fix, "nis": nis, "peak": mres.peak,
                                      "ratio": mres.peak_ratio, "sig": float(np.sqrt(np.trace(cov) / 2)),
                                      "fix": fix.tolist(), "prior_err": float(np.linalg.norm(pre_p - true_c))})
                    else:
                        fixes.append({"t": fr["t"], "accepted": False, "err": None,
                                      "reason": mres.reason or f"weak after outage (ratio {mres.peak_ratio:.2f})"})
                timing["match"] += time.perf_counter() - t0
            if rr is not None:
                rr.frame(fr["t"], fr["gray"], vo.tracks if vo is not None else None)

        # --- camera capture
        if rend is not None and k % cam_every == 0:
            t0 = time.perf_counter()
            s = traj.sample([tk])
            Rc = s.R[0] @ R_BODY_NADIR_CAM
            pc = s.p[0] + s.R[0] @ P_CAM_IN_BODY
            o = rend.render(pc, Rc)
            alb = to_gray(o["rgb"])
            from dss.sim.render.photometric import illumination_at

            ill = illumination_at(tk, events)
            depth = float(pc[2] - dem.sample(pc[0], pc[1]))
            om_c = R_BODY_NADIR_CAM.T @ s.omega_b[0]
            v_c = Rc.T @ s.v[0]
            blur = photo.blur_px(om_c, v_c, depth, FOCAL)
            extra = float(cam_cfg.get("extra_blur_px", 0.0)) if cam_cfg.get("blur_window") and \
                cam_cfg["blur_window"][0] <= tk < cam_cfg["blur_window"][1] else 0.0
            gray = photo.apply(alb, tk, events, g_cam, depth, blur, extra)
            raw_mean = float(np.mean(alb) * ill)
            agl_est = float(ref.p[2] - dem.sample(ref.p[0], ref.p[1]))
            pending_frame = {"k": k, "t": tk, "gray": gray, "raw_mean": raw_mean, "agl_est": agl_est,
                             "truth": {"p": s.p[0], "v": s.v[0], "R": s.R[0]}}
            timing["render"] += time.perf_counter() - t0

        # --- baro / lidar / mag
        while ib < len(baro) and baro.t_ns[ib] <= tn:
            if ib % 5 == 0:
                ref.baro(float(baro["alt_m"][ib]), 1.5**2)
                allf.baro(float(baro["alt_m"][ib]), 1.5**2)
            ib += 1
        while il < len(lidar) and lidar.t_ns[il] <= tn:
            if il % 4 == 0 and lidar["valid"][il]:
                zg = float(dem.sample(ref.p[0], ref.p[1]))
                var = float(lidar["sigma_m"][il]) ** 2
                ref.lidar(float(lidar["range_m"][il]), var, zg)
                allf.lidar(float(lidar["range_m"][il]), var, float(dem.sample(allf.p[0], allf.p[1])))
            health.set(float(lidar.t_ns[il]) * 1e-9, "RANGE", "OK" if lidar["valid"][il] else "UNAVAILABLE")
            il += 1
        while im < len(mag) and mag.t_ns[im] <= tn:
            if im % 5 == 0:
                m_b = mag["field_nt"][im]
                F = float(np.linalg.norm(m_b))
                m_w = ref.R @ m_b
                incl = float(np.arcsin(np.clip(-m_w[2] / max(F, 1.0), -1, 1)))
                bad = abs(F / F_nom - 1) > 0.05 or abs(incl - incl_nom) > np.deg2rad(3.0)
                mag_state = "DEGRADED" if bad else "OK"
                health.set(float(mag.t_ns[im]) * 1e-9, "MAG", mag_state, "field/inclination check" if bad else "")
                if not bad:
                    yv = np.deg2rad(2.0) ** 2
                    ref.mag_yaw(mag_yaw_fn_impl(m_b, ref.R, field_w), yv)
                    allf.mag_yaw(mag_yaw_fn_impl(m_b, allf.R, field_w), yv)
            im += 1

        while flow is not None and ifl < len(flow) and flow.t_ns[ifl] <= tn:
            agl_ref = float(ref.p[2] - dem.sample(ref.p[0], ref.p[1]))
            if flow["valid"][ifl] and 0.5 < agl_ref <= 20.0:
                from dss.sim.flow import flow_to_velocity

                h_perp = agl_ref * ref.R[2, 2]
                vb = flow_to_velocity(flow["flow_xy"][ifl:ifl + 1], flow["gyro_xyz"][ifl:ifl + 1],
                                      float(flow["integration_s"][ifl]), np.array([h_perp]))[0]
                sfl = 0.05 + 0.02 * np.linalg.norm(vb)
                ref.flow_velocity(vb, sfl)
                allf.flow_velocity(vb, sfl)
                health.set(float(flow.t_ns[ifl]) * 1e-9, "FLOW", "OK")
            ifl += 1

        # --- GNSS through the gate
        while gnss is not None and ig < len(gnss) and gnss.t_ns[ig] <= tn:
            tg = float(gnss.t_ns[ig]) * 1e-9
            valid = bool(gnss["valid"][ig])
            sep = separation_stat(allf.p, allf.pos_cov(), ref.p, ref.pos_cov()) if gate.state == gg.TRUSTED else None
            prev = gate.state
            if health.state["VIO"] == "FAILED":
                last_vio_fail_t = tg
            sig_ref_h = float(np.sqrt(np.max(np.linalg.eigvalsh(ref.pos_cov()[:2, :2]))))
            ref_deg = (tg - last_vio_fail_t < float(scn.get("gate", {}).get("ref_recover_hold_s", 20.0))) or sig_ref_h > 5.0
            use = gate.step(tg, valid, gnss["pos"][ig], gnss["vel"][ig], ref.p, ref.pos_cov(), ref.v, ref.P[3:6, 3:6], sep,
                            ref_degraded=ref_deg and tg > vio_init_t + 5.0)
            if gate.state != prev:
                health.set(tg, "GPS", gate.state, gate.reason)
                if gate.state in (gg.SUSPECT, gg.REJECTED, gg.UNAVAILABLE) and prev == gg.TRUSTED:
                    allf.copy_from(ref)  # re-seed ALL from REF, never by removing factors
                elif gate.state == gg.SUSPECT:
                    allf.copy_from(ref)
            if use and ig % 5 == 0:  # 1 Hz; each sample de-weighted for the tau=60 s coloured error
                sg = gnss["sigma_pos"][ig]
                allf.gps(gnss["pos"][ig], np.diag(sg**2), corr_inflation=float(fcfg.get("gps_corr_inflation", 20.0)))
            ig += 1

        # --- published output at 30 Hz
        if tk >= next_pub_t - 1e-9:
            # exact 30 Hz schedule: publish the sample nearest each 1/30 s tick (IMU ticks are 5 ms)
            dt = tk - last_pub_t if last_pub_t > -1e8 else 1.0 / 30.0
            last_pub_t = tk
            next_pub_t += 1.0 / 30.0
            if gate.state == gg.TRUSTED:
                tp = clamp_target(allf.p, allf.pos_cov(), ref.p, ref.pos_cov())
                tv, tR, tP = allf.v, allf.R, allf.pos_cov()
            else:
                tp, tv, tR, tP = ref.p, ref.v, ref.R, ref.pos_cov()
            yaw_rate = float((ref.R @ (gyro[k] - ref.f.bg))[2])
            pp, pR, pP = pub.step(dt, tp, tv, tR, tP, yaw_rate)
            vpe_log["t"].append(tk)
            vpe_log["p"].append(pp.copy())
            vpe_log["P"].append(pP.copy())
            vpe_log["dp_prop"].append(tv * dt)
            att_var = np.diag(ref.P[6:9, 6:9]) + np.deg2rad(0.1) ** 2
            frame = vpe.encode(int(tk * 1e6), pp, pR, pP, att_var, pub.reset_counter)
            vpe_frames += 1
            if frame[0] != 0xFD:
                vpe_bad += 1
            vpe_file.write(frame)
            if rr is not None:
                rr.pose(tk, traj.sample([tk]).p[0], pp, ref.p, pP, pub.hpl(), gate.state, health.vector())

        # --- log at 10 Hz
        if tk - last_log_t >= 0.1 - 1e-9:
            last_log_t = tk
            s = traj.sample([tk])
            log["t"].append(tk)
            log["gt"].append(s.p[0])
            log["gt_v"].append(s.v[0])
            log["gt_R"].append(s.R[0])
            log["ref"].append(ref.p.copy())
            log["ref_v"].append(ref.v.copy())
            log["Pref_v"].append(ref.P[3:6, 3:6].copy())
            log["all"].append(allf.p.copy())
            log["pub"].append(pub.p.copy())
            log["pub_R"].append(pub.R.copy())
            log["Pref"].append(ref.pos_cov().copy())
            log["Pall"].append(allf.pos_cov().copy())
            log["Ppub"].append(pub.P.copy())
            log["hpl"].append(pub.hpl())
            log["gate"].append(gg.STATE_CODE[gate.state])
            log["vision"].append({"OK": 0, "DEGRADED": 1, "FAILED": 2}[health.state["VIO"]])
            log["ntrk"].append(last_vio.n_tracked if last_vio is not None else 0)
            log["illum"].append(0.0)
            log["bleeding"].append(pub.bleeding)
            if gnss is not None:
                j = min(max(ig - 1, 0), len(gnss) - 1)
                log["gps"].append(gnss["pos"][j])
                log["gps_valid"].append(bool(gnss["valid"][j]))
                log["spoof_off"].append(float(np.linalg.norm(gnss["spoof_offset"][j][:2])))
            else:
                log["gps"].append(np.full(3, np.nan))
                log["gps_valid"].append(False)
                log["spoof_off"].append(0.0)
            log["map_state"].append(health.state["MAP"])
            log["pub_jump"].append(0.0)
            log["vio_p"].append(last_vio.p.copy() if last_vio is not None else np.full(3, np.nan))
            log["lapvar"].append(vmon.hist[-1] if vmon.hist else 0.0)

    vio.close()
    vpe_file.close()
    if rr is not None:
        rr.events(health.events, gate.events, fixes)
        rr.close()
    wall = time.time() - wall0
    L = {k: np.asarray(v) if k not in ("map_state",) else v for k, v in log.items()}
    from dss.eval.scenario_metrics import compute, plots_for

    metrics = compute(scn, seed, L, vpe_log, fixes, health, gate, ref, allf, so, vpe_frames, vpe_bad)
    metrics["manifest"] = build_manifest(seed, scn, {"git_sha": git_sha(), "vio": vio.name,
                                                     "imagery": W["info"],
                                                     "ovserver": _ov_binary_hash() if vio.name == "ovserver" else None})
    from dss.eval.report import write_metrics

    write_metrics(out_dir / "metrics.json", metrics)
    np.savez_compressed(out_dir / "log.npz", **{k: v for k, v in L.items() if k != "map_state"})
    (out_dir / "events.json").write_text(json.dumps({"health": health.events, "gate": gate.events,
                                                     "fixes": fixes}, default=float, indent=1) + "\n")
    plots_for(scn, L, fixes, gate, health, out_dir, metrics)
    (out_dir / "timing.json").write_text(json.dumps({"wall_s": wall, "sim_s": float(t_s[-1]),
                                                     "rtf": float(t_s[-1]) / wall, **timing}, indent=1) + "\n")
    if verbose:
        print(f"[{scn.get('name')} seed {seed}] wall {wall:.0f}s  " + metrics.get("headline", ""))
    return metrics


def imu_rate(imu) -> float:
    return 1e9 / float(np.median(np.diff(imu.t_ns[:100])))


def _ov_binary_hash():
    p = Path.home() / ".cache/dss/cpp/bin/ovserver-release"
    return sha256_file(p) if p.exists() else None


# ------------------------------------------------------------------------------------------------ CLI glue
SUITE = ["demo", "urban_canyon", "forest", "twilight", "aggressive", "gps_cutover", "dragoff_spoof", "denied_route",
         "denied_vio_only"]


def run_cli(a, demo: bool = False) -> int:
    scn = config.load(a.scenario)
    out = Path(a.out) if a.out else REPO_ROOT / "runs" / f"{scn['name']}-{a.seed}"
    m = run(scn, a.seed, out, vio_mode=a.vio, rrd=a.rrd or demo)
    print(json.dumps(m.get("scoreboard", {}), indent=1, default=float))
    return 0


def _suite_job(args):
    name, seed, out, vio = args
    os.environ.setdefault("MUJOCO_GL", "cgl")
    scn = config.load(f"scenarios/{name}.yaml")
    return name, seed, run(scn, seed, Path(out) / name / f"seed{seed}", vio_mode=vio)


def suite_cli(a) -> int:
    from concurrent.futures import ProcessPoolExecutor

    names = a.only or SUITE
    jobs = [(n, s, a.out, a.vio) for n in names for s in range(1, a.seeds + 1)]
    res = {}
    with ProcessPoolExecutor(max_workers=int(os.environ.get("DSS_JOBS", "2"))) as ex:
        for name, seed, m in ex.map(_suite_job, jobs):
            res.setdefault(name, {})[seed] = m
    from dss.eval.scenario_metrics import write_scoreboard

    write_scoreboard(Path(a.out), res)
    return 0
