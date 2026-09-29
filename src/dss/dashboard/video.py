"""Offline one-screen dashboard video (demo.mp4) from a run directory.

Same 3-column layout as the Rerun blueprint:
  (1) map view: NAIP render epoch, GT / published / REF trajectories, 3-sigma ellipse, drone
  (2) camera frame (re-rendered deterministically) + error vs HPL + GNSS-gate / vision timelines
  (3) scoreboard, health vector, event log
Frames are produced at ``fps_sim`` per simulated second and written at 30 fps with frame repetition, so
playback is 1x real time. Piped straight into ffmpeg; nothing is persisted but the mp4.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Ellipse  # noqa: E402

from dss.core import config  # noqa: E402

GATE_NAMES = ["UNAVAIL", "PROBATION", "TRUSTED", "SUSPECT", "REJECTED"]
GATE_COLORS = ["#999999", "#e6b800", "#2ca02c", "#ff7f0e", "#d62728"]


def make(run_dir: Path, out: Path | None = None, fps_sim: int = 5, width_px: int = 1920, height_px: int = 1080):
    run_dir = Path(run_dir)
    out = out or run_dir / "demo.mp4"
    m = json.loads((run_dir / "metrics.json").read_text())
    ev = json.loads((run_dir / "events.json").read_text())
    L = dict(np.load(run_dir / "log.npz"))
    scn_name = m["scenario"]
    scn = config.load(f"scenarios/{scn_name}.yaml")
    from dss.app import P_CAM_IN_BODY, renderer, world
    from dss.sensors.rig import R_BODY_NADIR_CAM
    from dss.sim import trajectory
    from dss.sim.render.camera import to_gray
    from dss.sim.render.photometric import illumination_at

    W = world()
    rend = renderer(W)
    traj = trajectory.from_config(scn["trajectory"], ground_fn=W["dem"].sample)
    t = L["t"]
    gt, pub, ref = L["gt"], L["pub"], L["ref"]
    e_pub = np.linalg.norm((pub - gt)[:, :2], axis=1)
    e_ref = np.linalg.norm((ref - gt)[:, :2], axis=1)
    hpl = L["hpl"]
    dpi = 100
    fig = plt.figure(figsize=(width_px / dpi, height_px / dpi), dpi=dpi, facecolor="#101418")
    gsp = fig.add_gridspec(3, 3, width_ratios=[1.35, 1.2, 0.8], height_ratios=[1.3, 1, 0.55], wspace=0.18, hspace=0.35,
                           left=0.03, right=0.99, top=0.93, bottom=0.05)
    txt = dict(color="#e8e8e8")
    fig.suptitle(f"denied-signal-stack · {scn_name} · seed {m['seed']} · Flagstaff AZ, 2.0 km @ 100 m AGL", fontsize=15, **txt)
    # (1) map
    axm = fig.add_subplot(gsp[:, 0])
    img = W["img"]
    pad = 150
    x0, x1 = gt[:, 0].min() - pad, gt[:, 0].max() + pad
    y0, y1 = gt[:, 1].min() - pad, gt[:, 1].max() + pad
    i0, j0 = img.xy_to_ij(x0, y1)
    i1, j1 = img.xy_to_ij(x1, y0)
    bg = img.data[int(i0):int(i1):4, int(j0):int(j1):4, :3]
    axm.imshow(bg, extent=[x0, x1, y0, y1], alpha=0.85)
    axm.plot(gt[:, 0], gt[:, 1], color="white", lw=0.6, ls="--", alpha=0.6)
    l_gt, = axm.plot([], [], color="white", lw=1.6, label="ground truth")
    l_ref, = axm.plot([], [], color="#2ca02c", lw=1.2, label="REF (no GPS)")
    l_pub, = axm.plot([], [], color="#c77dff", lw=1.6, label="published")
    l_gps, = axm.plot([], [], ".", color="#ff4d4d", ms=2, label="GPS as received")
    drone, = axm.plot([], [], "o", color="white", mec="k", ms=9)
    ell = Ellipse((0, 0), 1, 1, fill=False, color="#c77dff", lw=1.5)
    axm.add_patch(ell)
    axm.set_xlim(x0, x1)
    axm.set_ylim(y0, y1)
    axm.set_aspect("equal")
    axm.legend(loc="lower left", fontsize=9, facecolor="#202428", labelcolor="#e8e8e8")
    axm.set_title("map view (grid ENU, m) · 3σ ellipse ×10", fontsize=11, **txt)
    axm.tick_params(colors="#bbbbbb", labelsize=7)
    # (2) camera
    axc = fig.add_subplot(gsp[0, 1])
    cam_im = axc.imshow(np.zeros((480, 640)), cmap="gray", vmin=0, vmax=255)
    axc.set_axis_off()
    cam_title = axc.set_title("nadir camera 640×480", fontsize=11, **txt)
    # error vs HPL
    axe = fig.add_subplot(gsp[1, 1])
    axe.set_facecolor("#181c20")
    axe.semilogy(t, np.maximum(e_pub, 1e-2), color="#c77dff", lw=1.0, label="published err_h")
    axe.semilogy(t, np.maximum(e_ref, 1e-2), color="#2ca02c", lw=0.8, label="REF err_h")
    axe.semilogy(t, hpl, color="#4db8ff", lw=0.9, ls="--", label="HPL(t)")
    cur = axe.axvline(0, color="white", lw=0.8)
    axe.set_ylim(0.05, max(300, hpl.max() * 1.2))
    axe.legend(fontsize=8, loc="upper left", facecolor="#202428", labelcolor="#e8e8e8", ncol=3)
    axe.set_title("horizontal error vs protection level [m]", fontsize=11, **txt)
    axe.tick_params(colors="#bbbbbb", labelsize=8)
    # attack timeline
    axt = fig.add_subplot(gsp[2, 1])
    axt.set_facecolor("#181c20")
    g = L["gate"].astype(int)
    for code, col in enumerate(GATE_COLORS):
        mk = g == code
        axt.fill_between(t, 1.1, 2.0, where=mk, color=col, step="post", lw=0)
    vcol = ["#2ca02c", "#e6b800", "#d62728"]
    for code, col in enumerate(vcol):
        axt.fill_between(t, 0.0, 0.9, where=L["vision"] == code, color=col, step="post", lw=0)
    axt.set_yticks([0.45, 1.55], ["vision", "GNSS gate"], color="#e8e8e8", fontsize=9)
    sp = scn.get("spoof")
    marks = []
    jam = scn.get("jam")
    if jam:
        marks.append((jam["t0"], "jam"))
    if sp:
        marks.append((sp["t0"], "spoofer capture"))
        if sp.get("drag_delay_s"):
            marks.append((sp["t0"] + sp["drag_delay_s"], "drag-off"))
    if m.get("spoof", {}).get("t_flag") is not None:
        marks.append((m["spoof"]["t_flag"], "FLAGGED"))
    for x, lab in marks:
        axt.axvline(x, color="white", lw=0.8, ls=":")
        axt.text(x + 1, 2.08, lab, color="white", fontsize=7, rotation=0)
    curt = axt.axvline(0, color="white", lw=1.0)
    axt.set_xlim(t[0], t[-1])
    axt.set_ylim(0, 2.5)
    axt.set_title("attack / failover timeline", fontsize=11, **txt)
    axt.tick_params(colors="#bbbbbb", labelsize=8)
    # (3) scoreboard + health + events
    axs = fig.add_subplot(gsp[0, 2])
    axs.set_axis_off()
    sb_txt = axs.text(0.0, 1.0, "", va="top", ha="left", family="monospace", fontsize=10.5, color="#e8e8e8",
                      transform=axs.transAxes)
    axh = fig.add_subplot(gsp[1, 2])
    axh.set_facecolor("#181c20")
    srcs = ["IMU", "VIO", "MAP", "GPS", "BARO", "MAG", "RANGE", "FLOW"]
    bars = axh.barh(srcs, [0] * len(srcs), color="#2ca02c")
    axh.set_xlim(0, 3)
    axh.set_xticks([0, 1, 2, 3], ["ok", "degr.", "susp.", "failed"], fontsize=8, color="#bbbbbb")
    axh.tick_params(colors="#e8e8e8", labelsize=9)
    axh.invert_yaxis()
    axh.set_title("health vector", fontsize=11, **txt)
    axl = fig.add_subplot(gsp[2, 2])
    axl.set_axis_off()
    log_txt = axl.text(0.0, 1.0, "", va="top", ha="left", family="monospace", fontsize=8, color="#e8e8e8",
                       transform=axl.transAxes)
    axl.set_title("event log", fontsize=11, **txt)

    hev = sorted({(round(float(a), 2), s_, st, why) for a, s_, st, why in ev["health"]})
    from dss.integrity.health import LEVEL

    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s",
                           f"{width_px}x{height_px}", "-framerate", str(fps_sim), "-i", "-", "-r", "30", "-pix_fmt", "yuv420p",
                           "-c:v", "libx264", "-crf", "23", str(out)], stdin=subprocess.PIPE)
    events = scn.get("camera", {}).get("events", [])
    ts = np.arange(t[0], t[-1], 1.0 / fps_sim)
    g_cam = np.random.default_rng(0)
    fin = m.get("final_after_recovery", {})
    for tk in ts:
        i = min(int(np.searchsorted(t, tk)), len(t) - 1)
        l_gt.set_data(gt[:i + 1, 0], gt[:i + 1, 1])
        l_ref.set_data(ref[:i + 1, 0], ref[:i + 1, 1])
        l_pub.set_data(pub[:i + 1, 0], pub[:i + 1, 1])
        gv = L["gps_valid"][:i + 1].astype(bool)
        l_gps.set_data(L["gps"][:i + 1][gv, 0], L["gps"][:i + 1][gv, 1])
        drone.set_data([gt[i, 0]], [gt[i, 1]])
        P = L["Ppub"][i][:2, :2]
        w, V = np.linalg.eigh(P)
        ell.set_center((pub[i, 0], pub[i, 1]))
        ell.width, ell.height = 2 * 3 * 10 * np.sqrt(np.maximum(w, 1e-6))
        ell.angle = float(np.degrees(np.arctan2(V[1, 1], V[0, 1])))
        s = traj.sample([tk])
        Rc = s.R[0] @ R_BODY_NADIR_CAM
        pc = s.p[0] + s.R[0] @ P_CAM_IN_BODY
        alb = to_gray(rend.render(pc, Rc)["rgb"]) * illumination_at(tk, events)
        gain = 1.0 if alb.mean() > 60 else min(8.0, 110.0 / max(alb.mean(), 1.0))
        cam = np.clip(alb * gain + g_cam.normal(0, 3.0 * gain, alb.shape), 0, 255)
        cam_im.set_data(cam)
        cam_title.set_text(f"nadir camera · t = {tk:6.1f} s · KLT tracks {int(L['ntrk'][i])}")
        cur.set_xdata([tk, tk])
        curt.set_xdata([tk, tk])
        gs = GATE_NAMES[int(g[i])]
        vs = ["OK", "DEGRADED", "FAILED"][int(L["vision"][i])]
        lines = [f"t          {tk:7.1f} s", f"err_h pub  {e_pub[i]:7.2f} m", f"err_h REF  {e_ref[i]:7.2f} m",
                 f"HPL        {hpl[i]:7.1f} m", f"alt err    {abs((pub - gt)[i, 2]):7.2f} m", "",
                 f"GNSS gate  {gs}", f"vision     {vs}", f"map        {L.get('map_state', ['?'])[i] if 'map_state' in L else ''}"]
        if sp and m.get("spoof", {}).get("t_flag") and tk >= m["spoof"]["t_flag"]:
            lines += ["", f"spoof flagged {m['spoof']['latency_from_onset_s']:.1f} s", "  after onset"]
        if tk >= fin.get("t", 1e9):
            lines += ["", "FINAL (after recovery)", f"  err_h {fin['pub_h_m']:.2f} m", f"  RMSE_h {m['pub_h']['rmse']:.2f} m",
                      f"  HMI epochs {m['hmi_epochs']}"]
        sb_txt.set_text("\n".join(lines))
        state = {s_: "OK" for s_ in srcs}
        for a, s_, st, _ in hev:
            if a <= tk and s_ in state:
                state[s_] = st
        for b, s_ in zip(bars, srcs):
            lv = LEVEL.get(state[s_], 1)
            b.set_width(max(lv, 0.08))
            b.set_color(["#2ca02c", "#e6b800", "#ff7f0e", "#d62728"][min(lv, 3)])
        recent = [f"{a:6.1f} {s_:5s} {st}" for a, s_, st, _ in hev if a <= tk][-9:]
        log_txt.set_text("\n".join(recent))
        fig.canvas.draw()
        ff.stdin.write(np.asarray(fig.canvas.buffer_rgba()).tobytes())
    ff.stdin.close()
    ff.wait()
    plt.close(fig)
    return out


if __name__ == "__main__":
    print(make(Path(sys.argv[1])))
