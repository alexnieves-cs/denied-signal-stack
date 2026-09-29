"""GNSS gate: the ONLY path from GPS into fusion (import contract, tested).

FSM: UNAVAILABLE -> PROBATION -> TRUSTED -> SUSPECT -> REJECTED.
Tests (all against REF, which never sees GPS):
  - position CUSUM on the Gauss-Markov-whitened normalized innovation (1 Hz decimation)
  - velocity CUSUM: GPS Doppler velocity vs REF (VIO-aided) velocity
  - chi2_2 pre-gate (13.82) on each sample: excludes only that sample; 3 consecutive -> SUSPECT
  - ALL-REF separation > K sigma_ss (K=5) -> SUSPECT
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

UNAVAILABLE, PROBATION, TRUSTED, SUSPECT, REJECTED = "UNAVAILABLE", "PROBATION", "TRUSTED", "SUSPECT", "REJECTED"
STATE_CODE = {UNAVAILABLE: 0, PROBATION: 1, TRUSTED: 2, SUSPECT: 3, REJECTED: 4}


@dataclass
class Cusum:
    k: float
    h: float
    g: float = 0.0

    def step(self, x: float) -> float:
        self.g = max(0.0, self.g + x - self.k)
        return self.g

    def reset(self):
        self.g = 0.0


@dataclass
class GateConfig:
    probation_s: float = 35.0
    sigma_ref_max: float = 5.0
    pos_k: float = 2.0  # CUSUM drift on d^2/2 (chi2_2 mean -> 1)
    pos_h: float = 12.0
    vel_k: float = 2.0
    vel_h: float = 10.0
    pregate: float = 13.82
    sep_K: float = 5.0
    suspect_to_reject_s: float = 1.0
    gps_vel_sigma: float = 0.05
    ref_vel_floor: float = 0.08
    gm_tau_s: float = 60.0
    gm_sigma_h: float = 1.27
    cusum_gm_inflation: float = 2.0

    @staticmethod
    def from_dict(d: dict | None) -> GateConfig:
        c = GateConfig()
        for k, v in (d or {}).items():
            setattr(c, k, v)
        return c


@dataclass
class GnssGate:
    cfg: GateConfig = field(default_factory=GateConfig)
    state: str = UNAVAILABLE
    t_enter: float = 0.0
    consec_fail: int = 0
    last_decim_t: float = -1e9
    events: list = field(default_factory=list)
    alarm_t: float | None = None
    reason: str = ""

    def __post_init__(self):
        self.cpos = Cusum(self.cfg.pos_k, self.cfg.pos_h)
        self.cvel = Cusum(self.cfg.vel_k, self.cfg.vel_h)
        self.prev_r = None

    def _go(self, t: float, s: str, why: str):
        if s != self.state:
            self.events.append((t, self.state, s, why))
            self.state = s
            self.t_enter = t
            self.reason = why
            if s in (SUSPECT, REJECTED) and self.alarm_t is None:
                self.alarm_t = t

    def step(self, t: float, fix_valid: bool, pos, vel, ref_p, ref_P, ref_v, ref_Pv, sep_stat: float | None) -> bool:
        """Return True if this GPS sample may be fused into ALL."""
        c = self.cfg
        if self.state == REJECTED:
            return False
        if not fix_valid:
            if self.state in (PROBATION, TRUSTED):
                self._go(t, UNAVAILABLE, "no fix")
                self.cpos.reset()
                self.cvel.reset()
                self.prev_r = None
            return False
        if self.state == UNAVAILABLE:
            self._go(t, PROBATION, "fix acquired")
        # --- consistency statistics vs REF
        r = np.asarray(pos[:2]) - ref_p[:2]
        S = ref_P[:2, :2] + np.eye(2) * c.gm_sigma_h**2
        d2 = float(r @ np.linalg.solve(S, r))  # pre-gate statistic (single sample)
        # CUSUM statistic: the coloured (tau=60 s) GNSS error is not white at 1 Hz, so its variance is
        # inflated (x cusum_gm_inflation^2) so that minute-long 2-sigma excursions do not accumulate
        Sc = ref_P[:2, :2] + np.eye(2) * (c.cusum_gm_inflation * c.gm_sigma_h) ** 2
        d2c = float(r @ np.linalg.solve(Sc, r))
        rv = np.asarray(vel[:2]) - ref_v[:2]
        Sv = ref_Pv[:2, :2] + np.eye(2) * (c.gps_vel_sigma**2 + c.ref_vel_floor**2)
        dv2 = float(rv @ np.linalg.solve(Sv, rv))
        if t - self.last_decim_t >= 1.0:  # 1 Hz decimation for the correlated position test
            self.last_decim_t = t
            self.cpos.step(0.5 * d2c)
            self.cvel.step(0.5 * dv2)
        pre_ok = d2 <= c.pregate
        self.consec_fail = 0 if pre_ok else self.consec_fail + 1
        why = None
        if self.cpos.g > c.pos_h:
            why = f"position CUSUM {self.cpos.g:.1f} > {c.pos_h}"
        elif self.cvel.g > c.vel_h:
            why = f"velocity CUSUM {self.cvel.g:.1f} > {c.vel_h}"
        elif self.consec_fail >= 3:
            why = "3 consecutive pre-gate failures"
        elif sep_stat is not None and sep_stat > c.sep_K:
            why = f"ALL-REF separation {sep_stat:.1f} sigma"
        if why and self.state in (PROBATION, TRUSTED):
            self._go(t, SUSPECT, why)
        if self.state == SUSPECT:
            if t - self.t_enter >= c.suspect_to_reject_s:
                self._go(t, REJECTED, "suspect persisted: " + self.reason)
            return False
        if self.state == PROBATION:
            sig_ref = float(np.sqrt(np.max(np.linalg.eigvalsh(ref_P[:2, :2]))))
            if (t - self.t_enter >= c.probation_s and self.cpos.g < c.pos_h / 2 and self.cvel.g < c.vel_h / 2
                    and sig_ref <= c.sigma_ref_max):
                self._go(t, TRUSTED, "probation passed")
            else:
                return False
        return self.state == TRUSTED and pre_ok
