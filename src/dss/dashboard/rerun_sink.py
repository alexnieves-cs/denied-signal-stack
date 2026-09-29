"""Rerun adapter (pinned rerun-sdk 0.38): one fixed blueprint, written to run.rrd for demo / --debug runs.

Columns: (1) 3D view: drone, GT / published / REF trajectories, 3-sigma ellipsoid;
(2) camera + KLT tracks, error vs HPL, GNSS gate + vision state timelines;
(3) scoreboard markdown, health bars, event log.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from dss.integrity import gnss_gate as gg
from dss.integrity.health import LEVEL, SOURCES


class RerunSink:
    def __init__(self, path: Path, scn: dict, seed: int, spawn: bool = False):
        import rerun as rr
        import rerun.blueprint as rrb

        self.rr = rr
        rr.init(f"dss/{scn.get('name')}/{seed}", spawn=False)
        bp = rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial3DView(origin="/world", name="3D: drone, GT vs estimate, 3σ"),
                rrb.Vertical(
                    rrb.Spatial2DView(origin="/camera", name="camera + tracks"),
                    rrb.TimeSeriesView(origin="/error", name="horizontal error vs HPL"),
                    rrb.TimeSeriesView(origin="/state", name="GNSS gate / vision state"),
                ),
                rrb.Vertical(
                    rrb.TextDocumentView(origin="/scoreboard", name="scoreboard"),
                    rrb.BarChartView(origin="/health", name="health vector (0 ok … 3 failed)"),
                    rrb.TextLogView(origin="/events", name="event log"),
                ),
                column_shares=[2, 2, 1],
            ),
            collapse_panels=True,
        )
        rr.save(str(path), default_blueprint=bp)
        self.scn = scn
        self.traj_gt, self.traj_pub, self.traj_ref = [], [], []
        self.k = 0
        self.last_state = None

    def _t(self, t):
        self.rr.set_time("sim_time", duration=float(t))

    def frame(self, t, gray, tracks):
        if int(t * 20) % 2:  # log camera at 10 Hz to keep the .rrd small
            return
        self._t(t)
        rr = self.rr
        rr.log("camera/image", rr.Image(gray[::2, ::2]))
        if tracks is not None and len(tracks):
            tr = np.asarray(tracks, float)
            rr.log("camera/tracks", rr.Points2D(tr[:, 1:3] / 2.0, radii=1.5, colors=[255, 200, 0]))

    def pose(self, t, p_gt, p_pub, p_ref, P_pub, hpl, gate_state, hv):
        self.k += 1
        if self.k % 3:
            return  # 10 Hz
        rr = self.rr
        self._t(t)
        self.traj_gt.append(p_gt)
        self.traj_pub.append(p_pub)
        self.traj_ref.append(p_ref)
        if self.k % 30 == 0:
            rr.log("world/gt", rr.LineStrips3D([np.asarray(self.traj_gt)], colors=[40, 40, 40]))
            rr.log("world/pub", rr.LineStrips3D([np.asarray(self.traj_pub)], colors=[148, 103, 189]))
            rr.log("world/ref", rr.LineStrips3D([np.asarray(self.traj_ref)], colors=[44, 160, 44]))
        rr.log("world/drone", rr.Points3D([p_gt], radii=4.0, colors=[30, 30, 30]))
        w, V = np.linalg.eigh(0.5 * (P_pub + P_pub.T))
        half = 3.0 * np.sqrt(np.maximum(w, 1e-6))
        from dss.core.rotations import rot_to_quat

        q = rot_to_quat(V if np.linalg.det(V) > 0 else -V)
        rr.log("world/estimate", rr.Ellipsoids3D(centers=[p_pub], half_sizes=[half],
                                                  quaternions=[rr.Quaternion(xyzw=[q[1], q[2], q[3], q[0]])],
                                                  colors=[148, 103, 189]))
        e = float(np.linalg.norm((p_pub - p_gt)[:2]))
        rr.log("error/published", rr.Scalars(e))
        rr.log("error/REF", rr.Scalars(float(np.linalg.norm((p_ref - p_gt)[:2]))))
        rr.log("error/HPL", rr.Scalars(hpl))
        rr.log("state/gnss_gate", rr.Scalars(gg.STATE_CODE[gate_state]))
        rr.log("state/vision", rr.Scalars({"OK": 0, "DEGRADED": 1, "FAILED": 2}[hv["VIO"]]))
        rr.log("health", rr.BarChart([LEVEL.get(hv[s], 1) for s in SOURCES]))
        if gate_state != self.last_state:
            rr.log("events", rr.TextLog(f"t={t:6.1f}s GNSS gate -> {gate_state}", level="WARN" if gate_state in
                                        (gg.SUSPECT, gg.REJECTED) else "INFO"))
            self.last_state = gate_state
        if self.k % 30 == 0:
            md = (f"## {self.scn.get('name')}\n| | |\n|---|---|\n| t | {t:.1f} s |\n| published err_h | {e:.2f} m |\n"
                  f"| HPL | {hpl:.1f} m |\n| GNSS gate | **{gate_state}** |\n| vision | {hv['VIO']} |\n| map | {hv['MAP']} |\n")
            rr.log("scoreboard", rr.TextDocument(md, media_type=rr.MediaType.MARKDOWN))

    def events(self, health_events, gate_events, fixes):
        rr = self.rr
        for t, src, st, why in health_events:
            self._t(t)
            rr.log("events", rr.TextLog(f"t={t:6.1f}s {src} -> {st} {why}", level="WARN" if st in ("FAILED", "SUSPECT", "REJECTED")
                                        else "INFO"))

    def close(self):
        self.rr.disconnect() if hasattr(self.rr, "disconnect") else None
