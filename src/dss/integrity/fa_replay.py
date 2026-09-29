"""False-alarm replay (set H): the frozen GNSS gate against REF logs from rendered runs, with fresh open-sky
GNSS noise realisations (seeds 2000+; disjoint from tuning). No jam, spoof or NLOS."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from dss.core import config
from dss.integrity import gnss_gate as gg
from dss.sim import gnss


def ref_degraded_mask(L, hold_s: float = 20.0, sigma_max: float = 5.0, t_min: float = 17.0) -> np.ndarray:
    """Same rule as the in-sim gate: vision FAILED within hold_s, or REF horizontal sigma > sigma_max."""
    t, vis, Pref = L["t"], L["vision"], L["Pref"]
    last = -1e9
    out = np.zeros(len(t), bool)
    sig = np.sqrt(np.max(np.linalg.eigvalsh(Pref[:, :2, :2]), axis=1))
    for j in range(len(t)):
        if vis[j] == 2:
            last = t[j]
        out[j] = ((t[j] - last < hold_s) or sig[j] > sigma_max) and t[j] > t_min
    return out


def replay_run(log_path: Path, n_seeds: int, seed0: int = 2000) -> dict:
    L = np.load(log_path)
    t, gt, gv = L["t"], L["gt"], L["gt_v"]
    ref, Pref = L["ref"], L["Pref"]
    if "ref_v" not in L:
        return {"skipped": "no REF velocity in log"}
    rv, Pv = L["ref_v"], L["Pref_v"]
    deg = ref_degraded_mask(L)
    c = config.load("sensors/gnss_m9n.yaml")
    alarms = 0
    exposure = 0.0
    tg = np.arange(t[0], t[-1], 0.2)
    idx = np.clip(np.searchsorted(t, tg), 0, len(t) - 1)
    for s in range(seed0, seed0 + n_seeds):
        d = gnss.simulate_gnss(tg, gt[idx], gv[idx], c, s)
        gate = gg.GnssGate()
        for k in range(len(tg)):
            i = idx[k]
            gate.step(tg[k], bool(d["valid"][k]), d["pos"][k], d["vel"][k], ref[i], Pref[i], rv[i], Pv[i], None,
                      ref_degraded=bool(deg[i]))
        alarms += int(gate.alarm_t is not None)
        exposure += (tg[-1] - tg[0]) / 3600.0
    return {"alarms": alarms, "exposure_h": exposure}


def main(results: Path, n_seeds: int = 20) -> dict:
    tot_a, tot_h, runs = 0, 0.0, 0
    for lp in sorted(Path(results).glob("*/seed*/log.npz")):
        r = replay_run(lp, n_seeds)
        if "alarms" in r:
            tot_a += r["alarms"]
            tot_h += r["exposure_h"]
            runs += 1
    # 95% upper bound on the rate with zero (or k) events: chi2 / (2T)
    from scipy.stats import chi2

    ub = chi2.ppf(0.95, 2 * (tot_a + 1)) / (2 * tot_h) if tot_h > 0 else float("nan")
    out = {"runs": runs, "gnss_seeds_per_run": n_seeds, "exposure_h": tot_h, "alarms": tot_a, "rate_ub95_per_h": ub,
           "note": "REF logs from rendered runs x fresh GNSS noise; thresholds frozen in configs/integrity/thresholds.yaml"}
    return out


if __name__ == "__main__":
    import sys

    res = main(Path(sys.argv[1] if len(sys.argv) > 1 else "eval/results"), int(sys.argv[2]) if len(sys.argv) > 2 else 20)
    Path("reports/phase3").mkdir(parents=True, exist_ok=True)
    Path("reports/phase3/false_alarms.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))


SWEEP = [{"kind": "step", "offset_m": 20.0, "label": "step 20 m"}] + \
    [{"kind": "dragoff", "rate_mps": r, "label": f"{r} m/s"} for r in (0.1, 0.3, 0.5, 1.0, 2.0, 3.0)] + \
    [{"kind": "dragoff", "accel_mps2": a, "label": f"{a} m/s^2"} for a in (0.01, 0.05, 0.2)]
ENVELOPE = {"step 20 m": 1.2, "2.0 m/s": 8.0, "1.0 m/s": 20.0, "0.5 m/s": 35.0}


def spoof_sweep(results: Path, seeds_per_run: int = 4, t_onset: float = 100.0, horizon: float = 150.0, seed0: int = 3000,
                exclude=(), only=None) -> dict:
    """Seamless capture of a TRUSTED receiver at t_onset (probation passed at 35 s), against REF logs."""
    c = config.load("sensors/gnss_m9n.yaml")
    logs = [lp for lp in sorted(Path(results).glob("*/seed*/log.npz")) if "ref_v" in np.load(lp)
            and lp.parts[-3] not in exclude and (only is None or lp.parts[-3] in only)]
    out = {}
    for prof in SWEEP:
        lat, lat_det, sid = [], [], []
        censored = 0
        n = 0
        for lp in logs:
            L = np.load(lp)
            t, gt, gv, ref, Pref, rv, Pv = L["t"], L["gt"], L["gt_v"], L["ref"], L["Pref"], L["ref_v"], L["Pref_v"]
            deg = ref_degraded_mask(L)
            t_end = min(t[-1], t_onset + horizon)
            tg = np.arange(t[0], t_end, 0.2)
            idx = np.clip(np.searchsorted(t, tg), 0, len(t) - 1)
            for s in range(seeds_per_run):
                ang = 2 * np.pi * (s + 0.37) / seeds_per_run
                sp = dict(prof, t0=t_onset, dir=[np.cos(ang), np.sin(ang), 0.0], seamless=True)
                d = gnss.simulate_gnss(tg, gt[idx], gv[idx], c, seed0 + s + 17 * n, spoof=sp)
                gate = gg.GnssGate()
                for k in range(len(tg)):
                    i = idx[k]
                    gate.step(tg[k], bool(d["valid"][k]), d["pos"][k], d["vel"][k], ref[i], Pref[i], rv[i], Pv[i], None,
                              ref_degraded=bool(deg[i]))
                n += 1
                off = np.linalg.norm(d["spoof_offset"][:, :2], axis=1)
                on = tg[off > 1e-6]
                if gate.alarm_t is None or gate.alarm_t < t_onset:
                    censored += int(gate.alarm_t is None)
                    if gate.alarm_t is not None:
                        lat.append(np.nan)  # false alarm before onset
                    continue
                t_on = on[0] if len(on) else t_onset
                lat.append(gate.alarm_t - t_on)
                sig = float(np.sqrt(np.max(np.linalg.eigvalsh(Pref[idx[np.searchsorted(tg, t_on)]][:2, :2])) + 1.27**2))
                det = tg[off > 5.94 * sig]
                if len(det):
                    lat_det.append(gate.alarm_t - det[0])
                # SID proxy: offset reached when flagged (GPS fused into ALL only while TRUSTED)
                sid.append(float(off[np.searchsorted(tg, gate.alarm_t)]))
        la = np.array([x for x in lat if np.isfinite(x)])
        out[prof["label"]] = {
            "segments": n, "detected": int(len(la)), "censored": censored,
            "p50_latency_s": float(np.median(la)) if len(la) else None,
            "p95_latency_s": float(np.percentile(la, 95)) if len(la) else None,
            "p95_from_detectable_s": float(np.percentile(lat_det, 95)) if lat_det else None,
            "max_offset_at_flag_m": float(np.max(sid)) if sid else None,
            "envelope_s": ENVELOPE.get(prof["label"]),
        }
        e = out[prof["label"]]
        e["pass_envelope"] = (e["envelope_s"] is None) or (e["p95_latency_s"] is not None and e["p95_latency_s"] <= e["envelope_s"]
                                                          and censored == 0)
    return out
