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
