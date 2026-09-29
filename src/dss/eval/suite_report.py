"""Aggregate Phase 2 / Phase 3 gate report from eval/results (+ offline gate replays).

`python -m dss.eval.suite_report eval/results` -> reports/phase2/gates.{md,json}, reports/phase3/gates.{md,json},
eval/results/scoreboard.md (regenerated).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from dss.core import config
from dss.eval.consistency import anees_band
from dss.eval.scenario_metrics import write_scoreboard


def _load(results: Path) -> dict:
    res = {}
    for mp in sorted(results.glob("*/seed*/metrics.json")):
        m = json.loads(mp.read_text())
        res.setdefault(m["scenario"], {})[m["seed"]] = m
    return res


def _fixes(results: Path, scn: str, textured_only: bool = True):
    out = []
    c = config.load(f"scenarios/{scn}.yaml")
    tex = [s for s in c.get("segments", []) if s["kind"] == "textured"]
    for ep in sorted((results / scn).glob("seed*/events.json")):
        for f in json.loads(ep.read_text())["fixes"]:
            if textured_only and not any(s["t0"] <= f["t"] < s["t1"] for s in tex):
                continue
            out.append(f)
    return out


def ev_gate(run_dir: Path, scn_name: str) -> dict:
    import os

    os.environ["MAVLINK20"] = "1"
    from pymavlink.dialects.v20 import common as mav

    from dss.eval.px4_ev_gate import replay
    from dss.geo import naip
    from dss.sim import trajectory

    data = (run_dir / "vpe.mavlink").read_bytes()
    m = mav.MAVLink(None)
    msgs = m.parse_buffer(data) or []
    t = np.array([x.usec * 1e-6 for x in msgs])
    p_ned = np.array([[x.x, x.y, x.z] for x in msgs])
    var = np.array([max(x.covariance[0], x.covariance[6], x.covariance[11]) for x in msgs])
    finite = all(np.all(np.isfinite(x.covariance)) and len(x.covariance) == 21 for x in msgs)
    scn = config.load(f"scenarios/{scn_name}.yaml")
    traj = trajectory.from_config(scn["trajectory"], ground_fn=naip.load_dem("flagstaff").sample)
    a_enu = traj._a(t)
    a_ned = np.stack([a_enu[:, 1], a_enu[:, 0], -a_enu[:, 2]], 1)
    r = replay(t, p_ned, var, a_ned)
    r.update({"frames": len(msgs), "rate_hz": float(len(t) / (t[-1] - t[0])), "all_cov_finite_21": bool(finite),
              "first_byte_0xFD": data[:1] == b"\xfd"})
    return r


def main(results: Path):
    results = Path(results)
    res = _load(results)
    write_scoreboard(results, res)
    P2, P3 = {}, {}

    # ---------------- Phase 2
    fx = [f for s in ("denied_route", "demo", "forest", "twilight", "gps_cutover", "dragoff_spoof", "urban_canyon")
          if (results / s).exists() for f in _fixes(results, s)]
    cand = [f for f in fx if f.get("err") is not None or "reason" in f]
    acc = np.array([f.get("accepted", False) for f in cand])
    err = np.array([f["err"] if f.get("err") is not None else np.inf for f in cand])
    ea = err[acc]
    P2["exit3_fix_quality"] = {
        "attempts": int(len(cand)), "accepted": int(acc.sum()),
        "median_m": float(np.median(ea)) if len(ea) else None, "p95_m": float(np.percentile(ea, 95)) if len(ea) else None,
        "success_rate": float(np.mean(acc & (err <= 5))) if len(cand) else None,
        "false_accept_rate": float(np.mean(ea > 20)) if len(ea) else None,
        "gate": "median<=3, p95<=8, success>=0.8, false-accept<=0.01",
    }
    q = P2["exit3_fix_quality"]
    q["pass"] = bool(q["median_m"] is not None and q["median_m"] <= 3 and q["p95_m"] <= 8 and q["success_rate"] >= 0.8
                     and q["false_accept_rate"] <= 0.01)
    # fix NEES (2-DoF) on accepted fixes
    nis = [f.get("nis") for f in cand if f.get("accepted") and f.get("nis") is not None]
    P2["exit3_fix_nis_mean"] = float(np.mean(nis)) if nis else None

    def seeds_of(name):
        return res.get(name, {})

    dr = seeds_of("denied_route")
    vo = seeds_of("denied_vio_only")
    if dr:
        rm = [m["textured"]["rmse"] for m in dr.values()]
        p99 = [m["textured"]["p99"] for m in dr.values()]
        med = [m["textured"]["median"] for m in dr.values()]
        hmi = [m["hmi_epochs"] for m in dr.values()]
        e4 = {"per_seed_textured_rmse_m": rm, "pooled_p99_m": float(np.max(p99)), "median_m": float(np.median(med)),
              "hmi_epochs": int(np.sum(hmi)), "ref_rmse_m": [m["ref_h"]["rmse"] for m in dr.values()]}
        if vo:
            ratio = [vo[s]["textured"]["rmse"] / dr[s]["textured"]["rmse"] for s in dr if s in vo]
            e4["rmse_vio_over_aided"] = ratio
            e4["median_ratio"] = float(np.median(ratio))
        e4["vertical_p95_m"] = float(np.max([m["pub_v"]["p95"] for m in dr.values()]))
        e4["touchdown_vertical_m"] = [m["pub_v"]["touchdown"] for m in dr.values()]
        e4["pass"] = bool(max(rm) <= 5 and e4["pooled_p99_m"] <= 10 and e4["median_m"] <= 3 and e4["hmi_epochs"] == 0
                          and e4.get("median_ratio", 0) >= 3)
        P2["exit4_bounded_error"] = e4
        n = len(dr)
        lo, hi = anees_band(3, n)
        an = float(np.mean([m["ref_nees_mean"] for m in dr.values()]))
        P2["exit5_honest_cov"] = {"N": n, "ref_pos_anees_time_avg": an, "band": [lo, hi], "pass": bool(lo <= an <= hi),
                                  "note": f"N={n} seeds (PLAN: 25)"}
    # forest
    fo = seeds_of("forest")
    if fo:
        P2["forest_leg"] = {s: m["segments"].get("seg_forest") for s, m in fo.items()}
    P2["exit6_robustness"] = {"backend": "ES-EKF (18-state, VIO-bias augmented); GTSAM IFLS not used (CUT, see LEDGER)"}
    geo = Path("reports/phase2/coregistration.json")
    if geo.exists():
        P2["exit1_geo"] = json.loads(geo.read_text())

    # ---------------- Phase 3
    demo = seeds_of("demo")
    if demo:
        P3["exit1_demo"] = {s: {"flagged_s": m.get("spoof", {}).get("latency_from_onset_s"),
                                "final_after_recovery_m": m["final_after_recovery"]["pub_h_m"],
                                "rmse_h_m": m["pub_h"]["rmse"], "hmi": m["hmi_epochs"],
                                "within_hpl_until_flag": m.get("spoof", {}).get("within_hpl_until_flag")}
                            for s, m in demo.items()}
    ds = seeds_of("dragoff_spoof")
    if ds:
        P3["exit2_dragoff_in_suite"] = {s: m.get("spoof") for s, m in ds.items()}
    # continuity + output + honest output over all scenarios
    cont, hmi, nees, evg = {}, {}, {}, {}
    for name, per in res.items():
        cont[name] = float(max(m["continuity"]["max_jump_m"] for m in per.values()))
        hmi[name] = int(sum(m["hmi_epochs"] for m in per.values()))
        nees[name] = float(np.mean([m["pub_nees_mean"] for m in per.values()]))
        rd = results / name / f"seed{min(per)}"
        try:
            evg[name] = ev_gate(rd, name)
        except Exception as e:  # pragma: no cover
            evg[name] = {"error": repr(e)}
    P3["exit4_continuity_max_jump_m"] = cont
    P3["exit4_ev_gate_replay"] = evg
    P3["exit5_hmi_epochs"] = hmi
    P3["exit5_pub_nees_time_avg"] = nees
    P3["exit5_band_N5"] = list(anees_band(3, 5))
    vis = {}
    for name in ("demo", "forest"):
        if name in res:
            vis[name] = {s: m.get("vision") for s, m in res[name].items()}
    P3["exit6_degradation"] = vis
    tw = seeds_of("twilight")
    if tw:
        P3["exit8_twilight_map_disabled"] = {s: any(e[1] == "MAP" and e[2] == "DISABLED" for e in m["health_events"])
                                              for s, m in tw.items()}
    uc = seeds_of("urban_canyon")
    if uc:
        P3["exit8_urban_canyon"] = {s: {"gate_events": m["gate_events"], "hmi": m["hmi_epochs"]} for s, m in uc.items()}
    gc = seeds_of("gps_cutover")
    if gc:
        P3["exit8_gps_cutover_ttr"] = {s: m.get("ttr") for s, m in gc.items()}
    ag = seeds_of("aggressive")
    if ag:
        P3["exit8_aggressive"] = {s: {"rmse_h_m": m["pub_h"]["rmse"], "max_h_m": m["pub_h"]["max"]} for s, m in ag.items()}
    for d, obj in (("phase2", P2), ("phase3", P3)):
        Path(f"reports/{d}").mkdir(parents=True, exist_ok=True)
        Path(f"reports/{d}/gates.json").write_text(json.dumps(obj, indent=1, default=str) + "\n")
    return P2, P3


if __name__ == "__main__":
    p2, p3 = main(Path(sys.argv[1] if len(sys.argv) > 1 else "eval/results"))
    print(json.dumps({"phase2": p2, "phase3": {k: v for k, v in p3.items() if k != "exit4_ev_gate_replay"}}, indent=1,
                     default=str)[:6000])
