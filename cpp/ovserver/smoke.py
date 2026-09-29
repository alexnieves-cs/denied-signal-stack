#!/usr/bin/env python3
"""ovserver protocol smoke test (stdlib only). See PROTOCOL.md.

Usage: smoke.py [ovserver_binary] [estimator_config.yaml]
Defaults: $HOME/.cache/dss/cpp/bin/ovserver and the Task 0.1 probe config (nadir mono 640x480).
Checks: PING, INIT, IMU stream, INIT_GT, CAM replies with a state, malformed CAM/IMU/unknown/oversized
requests are rejected without exiting, REANCHOR, SHUTDOWN exits 0.
"""

import math
import os
import random
import struct
import subprocess
import sys

HOME = os.path.expanduser("~")
BIN = sys.argv[1] if len(sys.argv) > 1 else f"{HOME}/.cache/dss/cpp/bin/ovserver"
CFG = sys.argv[2] if len(sys.argv) > 2 else f"{HOME}/.cache/dss/cpp/probe/flat/seed1/estimator_config.yaml"
W, H = 640, 480

INIT, IMU, CAM, INIT_GT, REANCHOR, SHUTDOWN, CAM_STEREO, PING = 1, 2, 3, 4, 5, 6, 7, 8
RSP_ACK, RSP_STATE = 0x81, 0x83


class Client:
    def __init__(self) -> None:
        self.p = subprocess.Popen([BIN], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=open("/tmp/ovserver_smoke.err", "w"))

    def req(self, typ: int, payload: bytes) -> tuple[int, bytes]:
        self.p.stdin.write(struct.pack("<IB", len(payload), typ) + payload)
        self.p.stdin.flush()
        hdr = self.p.stdout.read(5)
        if len(hdr) < 5:
            raise RuntimeError("ovserver closed the pipe")
        n, t = struct.unpack("<IB", hdr)
        return t, self.p.stdout.read(n)


def parse_state(b: bytes) -> dict:
    st = b[0]
    vals = struct.unpack_from("<" + "d" * 47, b, 1)
    off = 1 + 47 * 8
    n_tracked, n_msckf, n_slam, n_list = struct.unpack_from("<IIIH", b, off)
    off += 14
    assert len(b) == off + n_list * 12, "track list length mismatch"
    return {"status": st, "t": vals[0], "q": vals[1:5], "p": vals[5:8], "v": vals[8:11], "cov": vals[11:47],
            "n_tracked": n_tracked, "n_msckf": n_msckf, "n_slam": n_slam, "n_list": n_list}


def texture(seed: int) -> list[tuple[int, int, int]]:
    """Random 3x3 dots (FAST-detectable blobs) on a wide canvas: (x, y, intensity)."""
    rnd = random.Random(seed)
    return [(rnd.randrange(W + 200), rnd.randrange(H), rnd.randrange(120, 256)) for _ in range(2500)]


def frame(tex: list[tuple[int, int, int]], shift: int) -> bytes:
    out = bytearray([30]) * (W * H)
    for x0, y, val in tex:
        # right half moves 3x faster than the left: parallax so the F-matrix RANSAC is not degenerate
        x = x0 - (shift if x0 < W // 2 + 100 else 3 * shift)
        if 1 <= x < W - 2 and 1 <= y < H - 2:
            for dy in (-1, 0, 1):
                base = (y + dy) * W + x
                out[base - 1:base + 2] = bytes([val]) * 3
    return bytes(out)


def main() -> int:
    c = Client()
    ok = lambda rsp: rsp[0] == RSP_ACK and rsp[1][0] == 0
    checks = {}
    checks["ping"] = ok(c.req(PING, b""))
    checks["imu_before_init_rejected"] = c.req(IMU, struct.pack("<7d", 0, 0, 0, 0, 0, 0, 9.81))[1][0] == 4
    checks["init"] = ok(c.req(INIT, CFG.encode()))
    checks["bad_config_rejected"] = c.req(INIT, b"/nonexistent.yaml")[1][0] == 4
    checks["init_again"] = ok(c.req(INIT, CFG.encode()))
    t0, dt = 100.0, 1.0 / 200
    k = 0

    def imu_until(t_end: float) -> bool:
        nonlocal k
        good = True
        while t0 + k * dt <= t_end + 1e-9:
            good &= ok(c.req(IMU, struct.pack("<7d", t0 + k * dt, 0, 0, 0, 0, 0, 9.81)))
            k += 1
        return good

    checks["imu_stream"] = imu_until(t0 + 0.5)
    checks["nonincreasing_imu_rejected"] = c.req(IMU, struct.pack("<7d", t0, 0, 0, 0, 0, 0, 9.81))[1][0] == 4
    gt = struct.pack("<17d", t0 + 0.5, 0, 0, 0, 1, 0, 0, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    checks["init_gt"] = ok(c.req(INIT_GT, gt))
    tex = texture(1)
    states = []
    for i in range(10):
        tc = t0 + 0.55 + 0.05 * i
        early = c.req(CAM, struct.pack("<dBII", tc, 0, W, H) + frame(tex, i)) if i == 0 else None
        if early is not None:
            checks["cam_without_imu_needs_imu"] = early[0] == RSP_STATE and early[1][0] == 3
        imu_until(tc + 0.01)
        t, b = c.req(CAM, struct.pack("<dBII", tc, 0, W, H) + frame(tex, i))
        assert t == RSP_STATE
        states.append(parse_state(b))
    checks["cam_replies_ok"] = all(s["status"] == 0 for s in states)
    checks["state_finite"] = all(math.isfinite(x) for s in states for x in (*s["p"], *s["q"], *s["cov"]))
    checks["tracks_reported"] = (states[-1]["n_tracked"] > 20 and states[-1]["n_tracked"] > states[0]["n_tracked"]
                                 and all(s["n_list"] == min(s["n_tracked"], 200) for s in states))
    t, b = c.req(CAM, struct.pack("<dBII", t0 + 2, 0, 320, 240) + bytes(320 * 240))
    checks["wrong_size_rejected"] = t == RSP_STATE and b[0] == 4
    t, b = c.req(CAM, struct.pack("<dBII", t0 + 2, 0, W, H) + bytes(100))
    checks["short_pixels_rejected"] = t == RSP_STATE and b[0] == 4
    checks["unknown_type_rejected"] = c.req(42, b"xyz")[1][0] == 4
    checks["oversize_rejected"] = False
    c.p.stdin.write(struct.pack("<IB", (64 << 20) + 1, CAM) + bytes((64 << 20) + 1))
    c.p.stdin.flush()
    n, typ = struct.unpack("<IB", c.p.stdout.read(5))
    checks["oversize_rejected"] = c.p.stdout.read(n)[0] == 4
    cov = [0.0] * 225
    for i, s in enumerate([0.01] * 3 + [2.4, 2.4, 1.0] + [0.2] * 3 + [1e-3] * 3 + [1e-2] * 3):
        cov[i * 15 + i] = s * s
    tr = t0 + 0.55 + 0.05 * 10
    imu_until(tr)
    checks["reanchor"] = ok(c.req(REANCHOR, struct.pack("<17d", tr, 0, 0, 0, 1, 0, 0, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0)
                                  + struct.pack("<225d", *cov)))
    bad = list(cov)
    bad[0] = -1.0
    checks["reanchor_nonpd_rejected"] = c.req(REANCHOR, struct.pack("<17d", tr, 0, 0, 0, 1, 0, 0, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0)
                                               + struct.pack("<225d", *bad))[1][0] == 4
    post = []
    for i in range(5):
        tc = tr + 0.05 * (i + 1)
        imu_until(tc + 0.01)
        post.append(parse_state(c.req(CAM, struct.pack("<dBII", tc, 0, W, H) + frame(tex, 10 + i))[1]))
    checks["cam_after_reanchor_ok"] = all(s["status"] == 0 for s in post)
    checks["reanchor_cov_used"] = abs(math.sqrt(post[0]["cov"][0]) - 2.4) < 0.5
    checks["shutdown"] = ok(c.req(SHUTDOWN, b""))
    checks["exit0"] = c.p.wait(timeout=10) == 0
    for name, v in checks.items():
        print(f"{'PASS' if v else 'FAIL'} {name}")
    print("statuses:", [(s["status"], s["n_tracked"]) for s in states], [(s["status"], s["n_tracked"], round(math.sqrt(max(s["cov"][0],0)),3)) for s in post])
    print("last state:", {k: v for k, v in states[-1].items() if k != "cov"})
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
