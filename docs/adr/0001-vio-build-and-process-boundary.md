# ADR-0001: VIO build and process boundary

- **Status:** Accepted (2026-09-28, Phase 0 Task 0.1)
- **Verdict:** Stock OpenVINS @`69488123` builds natively on macOS 27 / Apple clang 21 / arm64 with an **empty patch series**. All Task 0.1 build checks pass. The large-depth probe shows **no divergence** in 10/10 runs (ATE/length 0.19–0.89%), but position NEES averages ≈9. The filter is overconfident at 100 m depth, which is R12, not R11. The container fallback is not needed. We keep the native build and the process boundary.

## Decision
1. **VIO core:** OpenVINS @`69488123ed9362dd44b6f28e7f4680abbff1442b`, built natively by `cpp/spike/build_openvins_macos.sh` into `$HOME/.cache/dss/cpp` against:
   - Homebrew `eigen@3` 3.4.1 and `boost` 1.92.0;
   - OpenCV 4.14.0 built from source: minimal module list, KleidiCV/OpenCL/TBB/Eigen off;
   - Ceres 2.2.0 built from source: miniglog, no SuiteSparse, LAPACK on.

   `cmake` is 4.4.3 with `-DCMAKE_POLICY_VERSION_MINIMUM=3.5`; ninja is 1.13.2 on a sanitized PATH.
2. **Patch series:** empty (`cpp/patches/open_vins/` holds only `.gitkeep`). Every R1 cause was absorbed by the recipe's CMake flags; no source edits.
3. **Process boundary:** OpenVINS runs only inside `ovserver` (GPL-3.0, `cpp/ovserver/`). It speaks the length-prefixed stdin/stdout protocol in `cpp/ovserver/PROTOCOL.md`. The MIT Python package never links GPL code. The boundary also contains OpenVINS's `std::exit`s and stdout printing (fd 1 → stderr; protocol on a saved fd).
4. **Container fallback** (`ubuntu:24.04` linux/arm64): not built. Native passed, and PLAN.md §10 forbids switching for recipe errors.

## Evidence (logs in `$HOME/.cache/dss/cpp/logs/`)
| Check (PLAN Phase 0 exit 1) | Result |
|---|---|
| Deps: OpenCV 4.14.0 + Ceres 2.2.0 from SHA-256-pinned tarballs | built and installed; OpenVINS configure log shows `OPENCV: 4.14.0 \| BOOST: 1.92.0 \| CERES: 2.2.0` |
| Stock OpenVINS build (no ROS, no ArUco, asserts live) | **pass**, ~55 s at `-j4` |
| `nm -u test_sim_repeat` shows `___assert_rtn` (built without NDEBUG) | **pass** |
| `test_sim_repeat rpng_sim` prints `success! they all are the same!` | **pass** |
| `run_simulation rpng_sim` exits 0 | **pass** |
| `sim_rmse rpng_sim`: finite position RMSE ≤0.5 m and finite NEES | **pass**: RMSE 0.039 m over 296 m (0.013%), NEES ori 1.55 / pos 0.36 |
| `klt_smoke`: ≥100 tracks over 20 synthetic parallax frames; `cv::imread` works | **pass**: min 178 continuing tracks on frames 1–19, imread OK |
| `ovserver` test and release builds + `cpp/ovserver/smoke.py` (22 checks) | **pass** on both builds |

## Large-depth probe (R11 first reading; informational)
Setup (`cpp/spike/probe_large_depth.sh 5`, run 2026-09-28):
- **Simulator:** OpenVINS's own, GT init.
- **Camera:** nadir mono, 640×480, f = 400 px, no distortion, image top toward travel.
- **Rates:** 20 Hz camera, 200 Hz IMU.
- **IMU noise:** BMI160 class (accel 2.0e-3, gyro 2.4e-4 white; RW 5e-4 / 2e-5).
- **Trajectory:** 2 km S-turn at 100 m, 9 m/s, coordinated bank (±40° yaw, 55 s period).
- **Estimator config:** `fi_max_dist 200`; calibration of intrinsics, IMU intrinsics and g-sensitivity off; extrinsics and time offset still calibrated (rpng_sim defaults); MSCKF + SLAM as shipped.
- **Variants:** feature depth 98–102 m ("flat") and 80–120 m ("relief"), `sim_seed_measurements` 1–5.

| variant | seed | RMSE m | ATE/len % | final err m | NEES ori (3) | NEES pos (3) | 0-update epochs | diverged |
|---|---|---|---|---|---|---|---|---|
| flat | 1 | 11.15 | 0.56 | 18.6 | 3.08 | 13.10 | 13 (all in first 0.7 s) | no |
| flat | 2 | 12.70 | 0.64 | 23.6 | 2.57 | 13.26 | 13 | no |
| flat | 3 | 7.55 | 0.38 | 11.7 | 2.20 | 5.28 | 13 | no |
| flat | 4 | 3.80 | 0.19 | 5.6 | 2.75 | 1.36 | 16 | no |
| flat | 5 | 15.02 | 0.76 | 29.9 | 3.04 | 9.91 | 13 | no |
| relief | 1 | 8.71 | 0.44 | 20.8 | 2.71 | 6.25 | 15 | no |
| relief | 2 | 4.13 | 0.21 | 6.4 | 1.74 | 3.03 | 13 | no |
| relief | 3 | 6.99 | 0.35 | 9.8 | 2.81 | 5.48 | 14 | no |
| relief | 4 | 11.04 | 0.56 | 20.0 | 4.47 | 2.75 | 16 | no |
| relief | 5 | 17.74 | 0.89 | 32.5 | 2.78 | 34.44 | 12 | no |
| **mean** flat / relief | | 10.0 / 9.7 | **0.51 / 0.49** | 17.9 / 17.9 | 2.73 / 2.90 | **8.6 / 10.4** | | **0 / 10** |

Path length is 1,990 m and each run has 4,429 camera epochs. The zero-update epochs all fall in the first ~0.7 s, while the clone window fills. Because they are not the whole first 10 s, the probe is configured correctly.

### Reading
- **No divergence** at 100 m depth, so R11 does **not** force a Phase 1 re-plan. The nadir-mono rig stays the starting point for the Phase 1 sweep.
- **Drift ≈0.5% of distance**, with a 10–30 m end error after 2 km. That is under the sim target of ≤2%, but it is not "low single meters", so map aiding (Phase 2) is load-bearing, as planned.
- **Position covariance is optimistic** (NEES_pos ≈ 9 vs 3 expected; one seed at 34). Orientation is roughly consistent (≈2.8 vs 3). Consequences:
  - Phase 1 exit 9 (ANEES ≤ 4.03) is at risk;
  - the fusion backend must not trust raw VIO covariance. Inflate by α, tuned on denied segments (PLAN 2.4), and plan for R12 mitigations.
- **Caveats:**
  - OpenVINS's simulator gives ideal point features: no appearance, blur or outliers, and perfectly known feature depth bands. Rendered imagery will be worse.
  - GT init with OpenVINS's static covariance, and extrinsics/time offset still estimated.

## ovserver (Phase 1.1 groundwork, built here)
- `cpp/ovserver/` is a separate CMake project. It imports `libov_msckf_lib.dylib`; there is no upstream CMake package.
- Builds: `build.sh test` (no NDEBUG) → `$HOME/.cache/dss/cpp/bin/ovserver`, and `build.sh release` (`-O3 -DNDEBUG`) → `ovserver-release`.
- Two findings from the smoke test, both handled in the server:
  1. `initialize_with_gt()` leaves TrackKLT at `init_max_features` (15). Only the dynamic initializer raises it to `num_pts`, so `INIT_GT`/`REANCHOR` call `set_num_features(num_pts / num_cameras)`.
  2. `VioManager::initialized()` also requires a first update. The server therefore reports state from `is_initialized_vio`.
- `REANCHOR` builds a fresh `VioManager`, replays the last 3 s of IMU, calls `initialize_with_gt`, then overwrites the static covariance with the supplied 15×15 matrix, which must be positive definite.

## Consequences
- Phase 1 builds on `ovserver`. Still to do in ovserver:
  - clone-pair Δpose + joint covariance output;
  - `fast_state_propagate` (PR #549 backport, as patch `0001`);
  - determinism overrides (`num_opencv_threads: 0`).
- The C++ tree costs ~1.5 GB in `$HOME/.cache/dss/cpp` (ledger budget ~3 GB).
