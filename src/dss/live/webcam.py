"""Live camera mode: `python -m dss.live.webcam [--video FILE] [--seconds 60]`.

Opens the FaceTime camera (needs the macOS Camera TCC grant) or a video file, runs KLT + up-to-scale mono VO,
and logs tracks + the trajectory (labelled "scale: arbitrary") to a Rerun viewer / .rrd.
"""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from dss.live.mono_vo import MonoVO


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=None)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--rrd", default="runs/live.rrd")
    a = ap.parse_args(argv)
    import rerun as rr

    rr.init("dss/live", spawn=False)
    rr.save(a.rrd)
    cap = cv2.VideoCapture(a.video if a.video else 0)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("camera/video not available (grant Camera access in System Settings > Privacy)")
    h, w = frame.shape[:2]
    f = 0.9 * w  # FaceTime HD ~ 65 deg HFOV
    vo = MonoVO(np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]]))
    t0 = time.time()
    n = 0
    while time.time() - t0 < a.seconds:
        ok, frame = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        out = vo.step(g)
        n += 1
        rr.set_time("frame", sequence=n)
        rr.log("camera/image", rr.Image(g))
        rr.log("camera/tracks", rr.Points2D(out["tracks"], radii=2, colors=[255, 200, 0]))
        rr.log("vo/trajectory", rr.LineStrips3D([np.asarray(vo.traj)]))
        rr.log("vo/label", rr.TextDocument("scale: arbitrary (monocular VO, up to scale)"))
    fps = n / max(time.time() - t0, 1e-6)
    print(f"frames {n}, {fps:.1f} fps, re-inits {vo.reinits}, keyframes {len(vo.traj)}")


if __name__ == "__main__":
    main()
