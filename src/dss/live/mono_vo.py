"""Live monocular VO: FAST/Shi-Tomasi corners + pyramidal KLT tracks, essential-matrix relative pose.

Trajectory is UP TO SCALE ("scale: arbitrary"): each keyframe step has unit translation length times the
median-flow ratio, so shape is meaningful but metric scale is not (PLAN F4). Re-initializes (counted) when
tracking collapses.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class MonoVO:
    K: np.ndarray
    min_tracks: int = 80
    max_corners: int = 300
    kf_min_flow_px: float = 8.0
    R: np.ndarray = field(default_factory=lambda: np.eye(3))
    t: np.ndarray = field(default_factory=lambda: np.zeros(3))
    reinits: int = 0

    def __post_init__(self):
        self.prev = None
        self.pts = None
        self.kf_pts = None
        self.traj = [self.t.copy()]
        self.n_tracks = 0

    def _detect(self, g):
        p = cv2.goodFeaturesToTrack(g, self.max_corners, 0.01, 8)
        return p if p is not None else np.empty((0, 1, 2), np.float32)

    def step(self, gray: np.ndarray) -> dict:
        if self.prev is None:
            self.prev = gray
            self.pts = self._detect(gray)
            self.kf_pts = self.pts.copy()
            return {"tracks": self.pts.reshape(-1, 2), "pose": (self.R, self.t)}
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(self.prev, gray, self.pts, None, winSize=(21, 21), maxLevel=3)
        ok = st.ravel() == 1
        self.pts, self.kf_pts = nxt[ok], self.kf_pts[ok]
        self.n_tracks = int(ok.sum())
        self.prev = gray
        flow = np.linalg.norm((self.pts - self.kf_pts).reshape(-1, 2), axis=1)
        if len(self.pts) >= 8 and np.median(flow) >= self.kf_min_flow_px:
            E, inl = cv2.findEssentialMat(self.kf_pts, self.pts, self.K, cv2.RANSAC, 0.999, 1.0)
            if E is not None and E.shape == (3, 3):
                _, Rr, tr, _ = cv2.recoverPose(E, self.kf_pts, self.pts, self.K, mask=inl)
                # camera motion from keyframe to current: X_cur = Rr X_kf + tr -> T_kf_cur = inv
                Rk = Rr.T
                tk = -Rr.T @ tr.ravel()
                self.t = self.t + self.R @ tk * 1.0
                self.R = self.R @ Rk
                self.traj.append(self.t.copy())
            self.kf_pts = self.pts.copy()
        if len(self.pts) < self.min_tracks:
            if len(self.pts) < 8:
                self.reinits += 1
            fresh = self._detect(gray)
            self.pts = np.concatenate([self.pts, fresh]) if len(self.pts) else fresh
            self.kf_pts = self.pts.copy()
        return {"tracks": self.pts.reshape(-1, 2), "pose": (self.R, self.t), "n_tracks": self.n_tracks}
