# cpp/ — GPL-3.0 boundary

Everything under `cpp/` that links OpenVINS is **GPL-3.0**: `spike/` and `ovserver/`.
The rest of this repository is MIT. The Python package (`src/dss`) never links or imports GPL
code. It only talks to the `ovserver` binary over a pipe (see `ovserver/PROTOCOL.md`), so the
process boundary is also the license boundary.

| path | what |
|---|---|
| `spike/build_openvins_macos.sh` | Task 0.1 hermetic build into `$HOME/.cache/dss/cpp`: OpenCV 4.14 (minimal, KleidiCV off), Ceres 2.2 (miniglog), stock OpenVINS @`69488123`, then the spike tests and helpers |
| `spike/sim_rmse.cpp` | OpenVINS simulator + MSCKF with GT init: RMSE, drift %, NEES, per-epoch update counts |
| `spike/klt_smoke.cpp` | TrackKLT on synthetic parallax frames + `cv::imread` check |
| `spike/make_traj.py`, `spike/probe_large_depth.sh` | large-depth probe: nadir mono at ~100 m over a 2 km S-turn |
| `patches/open_vins/` | numbered patch series applied by the build script (empty: stock OpenVINS builds unpatched) |
| `ovserver/` | the OpenVINS process behind the stdin/stdout protocol; `build.sh [test\|release]`, `smoke.py` |

Build order:
```
bash cpp/spike/build_openvins_macos.sh          # deps, openvins, tests, helpers
bash cpp/spike/probe_large_depth.sh 5           # optional: large-depth probe
bash cpp/ovserver/build.sh test && bash cpp/ovserver/build.sh release
python3 cpp/ovserver/smoke.py                   # protocol smoke test
```
Binaries: `$HOME/.cache/dss/cpp/bin/ovserver` (test build, asserts live) and `ovserver-release`.
The C++ trees live outside the repo because the repo path contains a space.
