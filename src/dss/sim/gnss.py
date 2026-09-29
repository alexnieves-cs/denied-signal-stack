"""GNSS receiver model (position + velocity fixes) with jamming and spoofing.

Errors are coloured (first-order Gauss-Markov) per axis. Jamming lowers C/N0; below
``cn0_min`` there is no fix within one epoch, and a fix returns only after C/N0 has stayed
above threshold for ``reacquire_hold_s``. Spoofing (measurement-consistency level only, no RF):
  step      offset jumps to ``offset_m`` at t0
  dragoff   seamless capture at t0, then offset grows at ``rate_mps`` (or ``accel_mps2``) along ``dir``
  meacon    reported position is the true position delayed by ``delay_s``
The spoofer reports a healthy C/N0 while active (it overpowers the jammer).
"""

from __future__ import annotations

import numpy as np

from dss.core.rng import rng
from dss.core.time import s_to_ns
from dss.sim.baro import gauss_markov

CEP_TO_SIGMA = 1.0 / 1.1774


def spoof_offset(t: np.ndarray, spoof: dict | None, p_true: np.ndarray, v_true: np.ndarray, traj=None):
    """Return (offset_pos (N,3), offset_vel (N,3), active (N,) bool)."""
    n = len(t)
    off = np.zeros((n, 3))
    offv = np.zeros((n, 3))
    active = np.zeros(n, bool)
    if not spoof:
        return off, offv, active
    t0 = float(spoof["t0"])
    t1 = float(spoof.get("t1", np.inf))
    active = (t >= t0) & (t < t1)
    kind = spoof["kind"]
    d = np.asarray(spoof.get("dir", [1.0, 0.0, 0.0]), dtype=float)
    d = d / np.linalg.norm(d)
    tau = np.clip(t - t0, 0.0, None)
    if kind == "step":
        off[active] = spoof["offset_m"] * d
    elif kind == "dragoff":
        delay = float(spoof.get("drag_delay_s", 0.0))
        tt = np.clip(tau - delay, 0.0, None)
        if "accel_mps2" in spoof:
            a = float(spoof["accel_mps2"])
            mag, magv = 0.5 * a * tt**2, a * tt
        else:
            r = float(spoof["rate_mps"])
            mag, magv = r * tt, np.where(tau > delay, r, 0.0)
        off[active] = (mag[:, None] * d)[active]
        offv[active] = (magv[:, None] * d)[active]
    elif kind == "meacon":
        if traj is None:
            raise ValueError("meacon needs the trajectory")
        delay = float(spoof["delay_s"])
        td = np.clip(t - delay, traj.t0, None)
        s = traj.sample(td)
        off[active] = (s.p - p_true)[active]
        offv[active] = (s.v - v_true)[active]
    else:
        raise ValueError(kind)
    return off, offv, active


def cn0_profile(t: np.ndarray, cfg: dict, jam: dict | None) -> np.ndarray:
    cn0 = np.full(len(t), float(cfg["cn0_nominal_dbhz"]))
    if jam:
        t0, t_full = float(jam["t0"]), float(jam.get("t_full", jam["t0"]))
        t1 = float(jam.get("t1", np.inf))
        drop = float(jam.get("drop_dbhz", 30.0))
        ramp = np.clip((t - t0) / max(t_full - t0, 1e-9), 0.0, 1.0)
        ramp[t >= t1] = 0.0
        cn0 = cn0 - drop * ramp
    return cn0


def simulate_gnss(t: np.ndarray, p_true: np.ndarray, v_true: np.ndarray, cfg: dict, seed: int,
                  jam: dict | None = None, spoof: dict | None = None, traj=None, nlos_bias: np.ndarray | None = None,
                  name: str = "gnss0") -> dict:
    g = rng(seed, "gnss", name)
    n = len(t)
    dt = float(np.median(np.diff(t))) if n > 1 else 0.2
    sh = cfg["cep_m"] * CEP_TO_SIGMA
    sig = np.array([sh, sh, cfg["sigma_v_up_m"]])
    ep = np.stack([gauss_markov(g, n, dt, s, cfg["tau_pos_s"]) for s in sig], axis=1)
    ev = np.stack([gauss_markov(g, n, dt, cfg["sigma_v"], cfg["tau_vel_s"]) for _ in range(3)], axis=1)
    off, offv, sp_active = spoof_offset(t, spoof, p_true, v_true, traj)
    cn0 = cn0_profile(t, cfg, jam)
    if spoof:
        cn0 = np.where(sp_active, float(cfg["cn0_nominal_dbhz"]), cn0)
    above = cn0 >= cfg["cn0_min_dbhz"]
    # reacquisition hold: fix valid only when C/N0 has been above threshold for hold seconds
    hold_n = int(round(cfg["reacquire_hold_s"] / dt))
    valid = np.zeros(n, bool)
    run = hold_n  # start "locked" at t0
    for k in range(n):
        run = run + 1 if above[k] else 0
        valid[k] = above[k] and run > hold_n
    if spoof and spoof.get("seamless", True):
        valid |= sp_active  # a seamless capture keeps a fix flowing
    p = p_true + ep + off
    if nlos_bias is not None:
        p = p + nlos_bias
    v = v_true + ev + offv
    return {
        "t_ns": s_to_ns(t),
        "pos": p,
        "vel": v,
        "valid": valid,
        "cn0": cn0,
        "spoofed": sp_active,
        "spoof_offset": off,
        "err_pos": ep,
        "sigma_pos": np.tile(sig, (n, 1)),
        "sigma_vel": np.full((n, 3), cfg["sigma_v"]),
    }
