"""Per-scenario metrics, plots and the suite scoreboard (definitions in docs/metrics.md)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from dss.eval import metrics as M
from dss.eval import plots

HPL_ALERT_SOURCES = ("VIO", "GPS")


def _h(e):
    return np.linalg.norm(e[:, :2], axis=1)


def _seg_mask(t, seg):
    return (t >= seg["t0"]) & (t < seg["t1"])


def compute(scn, seed, L, vpe_log, fixes, health, gate, ref, allf, so, vpe_frames, vpe_bad) -> dict:
    t = L["t"]
    gt = L["gt"]
    e_pub = _h(L["pub"] - gt)
    e_ref = _h(L["ref"] - gt)
    e_all = _h(L["all"] - gt)
    e_v = np.abs((L["pub"] - gt)[:, 2])
    airborne = t >= scn["trajectory"].get("t_takeoff", 5.0)
    m: dict = {"scenario": scn["name"], "seed": seed}
    m["pub_h"] = {"rmse": float(np.sqrt(np.mean(e_pub[airborne] ** 2))), "p95": float(np.percentile(e_pub[airborne], 95)),
                  "max": float(e_pub[airborne].max()), "final": float(e_pub[-1]), "median": float(np.median(e_pub[airborne]))}
    m["ref_h"] = {"rmse": float(np.sqrt(np.mean(e_ref[airborne] ** 2))), "max": float(e_ref[airborne].max())}
    m["all_h"] = {"rmse": float(np.sqrt(np.mean(e_all[airborne] ** 2)))}
    m["pub_v"] = {"p95": float(np.percentile(e_v[airborne], 95)), "touchdown": float(e_v[-1])}
    m["path_m"] = M.path_length(gt)
    t_fin = float(scn.get("final_numbers_t", 248.0))  # end of the textured tail, after the post-forest recovery
    i_fin = min(int(np.searchsorted(t, t_fin)), len(t) - 1)
    m["final_after_recovery"] = {"t": float(t[i_fin]), "pub_h_m": float(e_pub[i_fin]), "ref_h_m": float(e_ref[i_fin]),
                                 "hpl_m": float(L["hpl"][i_fin])}
    # segments
    segs = {}
    for seg in scn.get("segments", []):
        mk = _seg_mask(t, seg)
        if mk.any():
            segs[seg["id"]] = {"kind": seg["kind"], "rmse": float(np.sqrt(np.mean(e_pub[mk] ** 2))),
                               "p99": float(np.percentile(e_pub[mk], 99)), "median": float(np.median(e_pub[mk])),
                               "max": float(e_pub[mk].max())}
    m["segments"] = segs
    tex = [s for s in scn.get("segments", []) if s["kind"] == "textured"]
    if tex:
        mk = np.any([_seg_mask(t, s) for s in tex], axis=0)
        m["textured"] = {"rmse": float(np.sqrt(np.mean(e_pub[mk] ** 2))), "p99": float(np.percentile(e_pub[mk], 99)),
                         "median": float(np.median(e_pub[mk]))}
    # integrity: HPL / HMI
    hpl = L["hpl"]
    # active alert = vision FAILED or the GNSS gate in SUSPECT (a REJECTED GPS is a settled state, not an alert)
    alert = (L["vision"] == 2) | (L["gate"] == 3)
    hmi = (e_pub > hpl) & ~alert & airborne
    m["hmi_epochs"] = int(hmi.sum())
    m["hpl_exceed_epochs"] = int(((e_pub > hpl) & airborne).sum())
    m["hpl_median"] = float(np.median(hpl[airborne]))
    # NEES of the published position (3-DoF), single run
    err3 = L["pub"] - gt
    nees = np.einsum("ti,tij,tj->t", err3, np.linalg.inv(L["Ppub"] + np.eye(3) * 1e-9), err3)
    m["pub_nees_mean"] = float(np.mean(nees[airborne]))
    err_ref3 = L["ref"] - gt
    m["ref_nees_mean"] = float(np.mean(np.einsum("ti,tij,tj->t", err_ref3, np.linalg.inv(L["Pref"]), err_ref3)[airborne]))
    # continuity of the 30 Hz stream
    vp = np.asarray(vpe_log["p"])
    dpp = np.asarray(vpe_log["dp_prop"])
    jumps = np.linalg.norm(vp[1:] - (vp[:-1] + dpp[1:]), axis=1)
    vt = np.asarray(vpe_log["t"])
    dt = np.diff(vt)
    m["continuity"] = {"max_jump_m": float(jumps.max()), "p99_jump_m": float(np.percentile(jumps, 99)),
                       "max_bleed_mps": float(np.max(jumps / np.maximum(dt, 1e-6))), "reset_counter_increments": 0,
                       "vpe_frames": int(vpe_frames), "vpe_rate_hz": float(len(vt) / (vt[-1] - vt[0])),
                       "vpe_bad_frames": int(vpe_bad)}
    # map fixes
    att = [f for f in fixes if f.get("err") is not None or "reason" in f]
    errs = np.array([f["err"] if f.get("err") is not None else np.inf for f in att])
    accd = np.array([f.get("accepted", False) for f in att])
    m["map"] = M.relocalization(errs, accd) if len(att) else {"attempts": 0}
    if len(att):
        m["map"]["candidates"] = int(np.isfinite(errs).sum())
        if accd.any():
            fe = np.array([f["err"] for f in att if f.get("accepted")])
            m["map"]["accepted_median_m"] = float(np.median(fe))
    # GNSS gate / spoof
    m["gate_events"] = [[float(a), b, c, d] for a, b, c, d in gate.events]
    m["gate_final"] = gate.state
    sp = scn.get("spoof")
    if sp:
        off = L["spoof_off"]
        sig_innov = float(np.sqrt(np.max(np.linalg.eigvalsh(L["Pref"][np.searchsorted(t, sp["t0"])][:2, :2]))
                                  + gate.cfg.gm_sigma_h**2))
        alarm = np.isin(L["gate"], [3, 4])
        st = M.spoof_timing(t, off, alarm, sig_innov)
        st["sigma_innov_m"] = sig_innov
        m["spoof"] = st
        # SID proxy: published deviation vs REF-only during the attack (paired nominal run not simulated)
        mk = t >= sp["t0"]
        m["spoof"]["max_pub_minus_ref_after_onset_m"] = float(_h(L["pub"][mk] - L["ref"][mk]).max()) if mk.any() else 0.0
        m["spoof"]["gps_ever_trusted_after_onset"] = bool(np.any(L["gate"][mk] == 2))
        tf = st.get("t_flag")
        if tf is not None:
            w = (t >= sp["t0"]) & (t <= tf)
            m["spoof"]["within_hpl_until_flag"] = bool(np.all(e_pub[w] <= hpl[w])) if w.any() else True
    # TTR after GPS loss
    jam = scn.get("jam")
    if jam:
        t_loss = float(jam.get("t_full", jam["t0"]))
        inside = np.einsum("ti,tij,tj->t", err3[:, :2], np.linalg.inv(L["Ppub"][:, :2, :2]), err3[:, :2]) <= 9.21
        m["ttr"] = M.ttr(t, e_pub, inside, t_loss)
    # vision degradation
    ev = scn.get("camera", {}).get("events", [])
    dark = [e for e in ev if e["scale"] < 0.2]
    if dark:
        e0 = dark[0]
        failed = t[(L["vision"] == 2) & (t >= e0["t0"])]
        m["vision"] = {"collapse_t": e0["t0"], "failed_t": float(failed[0]) if len(failed) else None,
                       "latency_s": float(failed[0] - e0["t0"]) if len(failed) else None}
        # recovery after the outage: REF within 10 m of truth, 10 s after the next accepted fix
        nxt = [f["t"] for f in fixes if f.get("accepted") and f["t"] > e0["t1"]]
        if nxt:
            w = (t >= nxt[0] + 10.0)
            m["vision"]["first_fix_after_t"] = nxt[0]
            m["vision"]["ref_err_10s_after_fix"] = float(e_ref[w][0]) if w.any() else None
        m["vision"]["max_err_in_outage"] = float(e_pub[(t >= e0["t0"]) & (t < e0["t1"] + 10)].max())
    m["health_events"] = [[float(a), b, c, d] for a, b, c, d in health.events][:400]
    m["filter_stats"] = {"REF": ref.stats, "ALL": allf.stats}
    sb = {
        "pub_rmse_h_m": m["pub_h"]["rmse"], "pub_final_h_m": m["final_after_recovery"]["pub_h_m"],
        "pub_touchdown_h_m": m["pub_h"]["final"], "pub_max_h_m": m["pub_h"]["max"],
        "textured_rmse_h_m": m.get("textured", {}).get("rmse"), "textured_median_h_m": m.get("textured", {}).get("median"),
        "hmi_epochs": m["hmi_epochs"], "max_jump_m": m["continuity"]["max_jump_m"],
        "map_accept": m["map"].get("accepted", 0), "map_median_m": m["map"].get("median_m"),
        "gate_final": gate.state,
    }
    if "spoof" in m:
        sb["spoof_latency_s"] = m["spoof"].get("latency_from_onset_s")
    if "ttr" in m:
        sb["ttr_s"] = m["ttr"]["ttr_s"]
    if "vision" in m:
        sb["vision_fail_latency_s"] = m["vision"].get("latency_s")
    m["scoreboard"] = sb
    m["headline"] = (f"pub RMSE {sb['pub_rmse_h_m']:.2f} m, final(after recovery) {sb['pub_final_h_m']:.2f} m, "
                     f"touchdown {sb['pub_touchdown_h_m']:.1f} m, max {sb['pub_max_h_m']:.1f} m, "
                     f"HMI {sb['hmi_epochs']}, gate {gate.state}"
                     + (f", spoof latency {sb.get('spoof_latency_s')}" if "spoof" in m else ""))
    return m


def plots_for(scn, L, fixes, gate, health, out: Path, m: dict):
    t = L["t"]
    gt = L["gt"]
    events = []
    for f in fixes:
        if f.get("accepted"):
            events.append(("", np.array(f["fix"])))
    series = {"GT": gt, "PUB (published)": L["pub"], "REF (no GPS)": L["ref"]}
    gv = L["gps_valid"].astype(bool)
    if gv.any():
        g = L["gps"].copy()
        g[~gv] = np.nan
        series["GPS (as received)"] = g
    plots.trajectory_plot(out / "trajectory.png", series, f"{scn['name']} seed {m['seed']}: trajectory (grid ENU)")
    spans = []
    jam = scn.get("jam")
    if jam:
        spans.append((jam["t0"], jam.get("t1", t[-1]), "GPS jammed"))
    for e in scn.get("camera", {}).get("events", []):
        if e["scale"] < 0.5:
            spans.append((e["t0"], e["t1"], "vision degraded"))
    vl = []
    sp = scn.get("spoof")
    if sp:
        vl.append((sp["t0"], "spoof onset"))
    if gate.alarm_t is not None:
        vl.append((gate.alarm_t, "spoof flagged"))
    plots.error_plot(out / "error.png", t, {"PUB": _h(L["pub"] - gt), "REF": _h(L["ref"] - gt)},
                     f"{scn['name']} seed {m['seed']}: horizontal error vs HPL", bounds={"HPL(t)": L["hpl"]}, spans=spans,
                     vlines=vl, logy=True)
    # attack / health timeline
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(3, 1, figsize=(10, 5), sharex=True)
    ax[0].step(t, L["gate"], where="post")
    ax[0].set_yticks(range(5), ["UNAVAIL", "PROBATION", "TRUSTED", "SUSPECT", "REJECTED"], fontsize=7)
    ax[0].set_title("GNSS gate state")
    ax[1].step(t, L["vision"], where="post", color="C1")
    ax[1].set_yticks(range(3), ["OK", "DEGRADED", "FAILED"], fontsize=7)
    ax[1].set_title("vision health")
    ax[2].plot(t, L["ntrk"], color="C2", lw=0.8)
    ax[2].set_ylabel("KLT tracks")
    ax[2].set_xlabel("t [s]")
    for a in ax:
        for x, _ in vl:
            a.axvline(x, color="k", ls=":", lw=0.8)
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "timeline.png", dpi=110)
    plt.close(fig)


def write_scoreboard(out: Path, res: dict):
    rows = []
    agg = {}
    for name, per in sorted(res.items()):
        sbs = [m["scoreboard"] for m in per.values()]
        keys = sorted({k for s in sbs for k in s})
        a = {}
        for k in keys:
            vals = [s.get(k) for s in sbs]
            num = [v for v in vals if isinstance(v, (int, float)) and v is not None and np.isfinite(v)]
            a[k] = {"median": float(np.median(num)), "max": float(np.max(num))} if num else vals
        agg[name] = a
        rows.append(f"| {name} | {len(per)} | {a['pub_rmse_h_m']['median']:.2f} | {_fmt(a.get('textured_median_h_m'))} | "
                    f"{_fmt(a.get('pub_final_h_m'))} | {a['pub_max_h_m']['max']:.1f} | "
                    f"{a['hmi_epochs']['max']:.0f} | {a['max_jump_m']['max']:.3f} | "
                    f"{_fmt(a.get('spoof_latency_s'))} | {_fmt(a.get('ttr_s'))} | {_fmt(a.get('vision_fail_latency_s'))} |")
    md = ["# Scenario scoreboard", "", "| scenario | seeds | pub RMSE_h median [m] | textured median err [m] | "
          "final err after recovery [m] | pub max_h [m] | HMI epochs (max) | "
          "max 30 Hz jump [m] | spoof latency median [s] | TTR median [s] | vision FAILED latency [s] |",
          "|---|---|---|---|---|---|---|---|---|---|---|"] + rows
    (out / "scoreboard.md").write_text("\n".join(md) + "\n")
    (out / "scoreboard.json").write_text(json.dumps(agg, indent=1, default=str) + "\n")


def _fmt(a):
    if isinstance(a, dict):
        return f"{a['median']:.2f}"
    return "—"
