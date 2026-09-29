#!/usr/bin/env python3
"""Large-depth probe trajectory (stdlib only).

2 km S-turn at 100 m AGL, 9 m/s, coordinated-turn bank, written as
"t x y z qx qy qz qw" (space separated, z-up ENU, q = Hamilton body(FLU)->world) at 50 Hz,
the format OpenVINS' Simulator reads via sim_traj_path.

Usage: make_traj.py OUT.txt [--length 2000] [--speed 9] [--alt 100] [--rate 50]
"""

import argparse
import math


def quat_from_euler_zyx(yaw: float, pitch: float, roll: float) -> tuple[float, float, float, float]:
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return x, y, z, w


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--length", type=float, default=2000.0)
    ap.add_argument("--speed", type=float, default=9.0)
    ap.add_argument("--alt", type=float, default=100.0)
    ap.add_argument("--rate", type=float, default=50.0)
    ap.add_argument("--yaw-amp-deg", type=float, default=40.0)
    ap.add_argument("--period", type=float, default=55.0)
    a = ap.parse_args()

    g = 9.81
    dt = 1.0 / a.rate
    duration = a.length / a.speed
    n = int(duration * a.rate) + 1
    amp = math.radians(a.yaw_amp_deg)
    w = 2 * math.pi / a.period
    x = y = 0.0
    t0 = 1_000_000.0  # arbitrary epoch, seconds
    with open(a.out, "w") as f:
        f.write("# t x y z qx qy qz qw  (ENU, body FLU->world, Hamilton)\n")
        for k in range(n):
            t = k * dt
            # smooth speed ramp over the first 2 s so the spline has no velocity step
            ramp = 0.5 - 0.5 * math.cos(math.pi * min(t / 2.0, 1.0))
            v = a.speed * (0.2 + 0.8 * ramp)
            yaw = amp * math.sin(w * t)
            yaw_rate = amp * w * math.cos(w * t)
            roll = -math.atan(v * yaw_rate / g)
            z = a.alt + 0.5 * math.sin(2 * math.pi * t / 20.0)
            vz = 0.5 * 2 * math.pi / 20.0 * math.cos(2 * math.pi * t / 20.0)
            pitch = -math.atan2(vz, v)  # nose follows the flight path (FLU: +pitch = nose down)
            qx, qy, qz, qw = quat_from_euler_zyx(yaw, pitch, roll)
            f.write(f"{t0 + t:.6f} {x:.6f} {y:.6f} {z:.6f} {qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}\n")
            x += v * math.cos(yaw) * dt
            y += v * math.sin(yaw) * dt


if __name__ == "__main__":
    main()
