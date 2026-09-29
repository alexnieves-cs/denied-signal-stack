# denied-signal-stack
Software-only positioning stack that keeps drones located when GPS is jammed or spoofed: visual-inertial odometry fused with orthoimagery aiding and an integrity monitor, proven in a 3D simulator with live metrics.

## Quickstart (macOS arm64)
```
bash scripts/bootstrap_macos.sh          # brew bottles, uv CPython 3.13.15, venv, doctor
uv run pytest                            # tests
bash cpp/spike/build_openvins_macos.sh   # OpenVINS + deps into ~/.cache/dss/cpp (GPL, separate process)
bash cpp/ovserver/build.sh
uv run dss demo --seed 42 --lockstep     # the SPEC demo -> runs/demo-42/
uv run dss suite --seeds 5               # scenario suite -> eval/results/
```
See `PLAN.md`, `SPEC.md`, `LEDGER.md` (build log with verdicts and numbers) and `docs/`.

## Results (sim, Flagstaff AZ 2.0 km @ 100 m AGL, real OpenVINS on rendered NAIP imagery)
| | |
|---|---|
| **GPS denied** for the whole flight (map aiding) | horizontal RMSE **1.5 m**, textured median 1.0 m, 0 HMI; VIO-only 10.6 m (5.9× worse) |
| **SPEC demo** (jam 60 s, spoof 120 s, forest 150–195 s) | RMSE 5.7 m, final 2.1 m, spoof flagged 6 s after drag-off, 0 HMI, max jump 1.8 cm |
| **Map matcher**, cross-epoch (2017 map vs 2023 world) | fix median 0.98 m, p95 3.1 m, 89% success, 0 false accepts |
| **TUM-VI** rooms 1–5 (OpenVINS mono/stereo) | ATE 0.064 / 0.077 m, drift ≤0.074% |

Full scoreboard: `eval/results/scoreboard.md`. The gates and every cut are in `LEDGER.md`.
