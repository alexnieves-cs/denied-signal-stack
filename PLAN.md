# PLAN — denied-signal-stack

> **Verdict:** Buildable on this M1 Pro / macOS 27: every Python dependency has a cp313 arm64 or pure wheel today.
> - The first unknown is a native OpenVINS build, never done publicly on macOS 27 / clang 21. Phase 0 opens with a one-day build spike that also takes a first, renderer-free reading on the bigger risk: MSCKF VIO at 100 m altitude over 2 km.
> - One SPEC metric is infeasible as written (CPU on target hardware), one target is ill-posed (spoof "within seconds"), and one claim is untraceable (~2.8 m). All are in §2 with replacements. Nothing is cut without your say-so.

## Context
- **Repo today:** `LICENSE` (MIT), `README.md`, untracked `SPEC.md`, one commit on `main`, no code.
- **Ask:** survey the machine; plan SPEC Phases 0–3 with libraries/files/modules per component, exit criteria, machine risks and the exact Phase 0 tree.
- **Plan-mode constraint:** only this file could be written. On approval it goes into the repo as `PLAN.md`, committed only when you say so.
- **Evidence:** three adversarial workflows back this plan.
  1. Five researchers plus five verifiers checked library/API/dataset/accuracy claims against live sources: PyPI, GitHub source at pinned commits, docs, and this Mac's `ioreg` and CGL probes. They verified 137 claims, covering all 130 load-bearing ones: 105 confirmed, 30 corrected, 1 refuted, and 1 unverifiable (the spoof-latency Monte Carlo, treated here as a hypothesis).
  2. Four critics attacked the first draft and raised 7 blockers and 39 majors.
  3. A re-check verified this version's build commands line by line against the pinned OpenVINS, OpenCV 4.14 and Ceres 2.2 sources, and re-audited the fixes.
  Every finding is addressed below.

## 1. Machine survey (2026-09-28)
| Item | Finding | Consequence |
|---|---|---|
| OS / HW | macOS 27.0 (Darwin 27, Homebrew `arm64_golden_gate`), MacBookPro18,3 M1 Pro (6P+2E), 14-core GPU (Metal 4), **16 GB RAM** | C++ risk lives in the new OS/compiler; ≤2 parallel sim workers |
| Disk | **75 GB free** (volume 92% used); Docker VM already 28 GB | Budget ledger (§8); never persist rendered frames |
| Python | python.org 3.13.15 is the default and has **no CA bundle** (urllib HTTPS fails); 3.11.0 has a polluted global env; brew 3.12/3.14; no uv/conda | **uv-managed CPython 3.13.15** venv; never pip into global interpreters |
| Why 3.13 | numpy 2.5, scipy 1.18, rasterio and pyproj need ≥3.12; moderngl stops at cp313; gtsam, mujoco, torch, rerun and pymavlink ship cp313 arm64 | 3.13 is the intersection |
| Toolchain | Xcode 27.0 (Apple clang 21), xcode-select → Xcode; CLT 26.6 (Homebrew refuses *source* builds with it); **no cmake**; pip `ninja` from the 3.11 env shadows PATH | Bottles only; C++ scripts use a sanitized PATH and explicit `-G Ninja` |
| Homebrew | `opencv` is **5.0.0**; `opencv@4` is a 2.9 GB closure, **deprecated 2026-10-09**; `eigen` is 5.0.1; `eigen@3` 3.4.1 keg; `ceres-solver` is locked to Eigen 5; `boost` 1.92 (still resolves the `system` component); `cmake` 4.4.3 | `eigen@3` + source-built minimal OpenCV 4.14 + Ceres 2.2 |
| Containers | Docker Desktop 4.86, daemon off; **no Rosetta 2** (macOS 27 is its last full release) | linux/arm64 only; Docker is a fallback, not the default |
| Sensors | FaceTime HD camera in the lid. `ioreg` shows an SPU accel + gyro in the base (root-only, unsynced, hinge-dependent extrinsic) | Live metric VIO is experimental (F4) |
| Simulator | None on disk or in GitHub (`alexnieves-cs`) | We build `dss.sim` |
| Repo path | `…/Spoofed Drone Nav/…` contains a space | Quote everything; C++ builds live in `$HOME/.cache/dss/cpp` |
| CI | Nearest hosted runner is GitHub `macos-26` arm64 (3 vCPU, 7 GB); no macOS 27 runner | CI covers pure-Python tests only |

## 2. SPEC feasibility flags (flagged first; approve or veto each)
| # | SPEC item | Verdict | Why | Proposed replacement |
|---|---|---|---|---|
| F1 | "CPU load on the target hardware" | **Infeasible as written** | No $399-class module exists before Phase 4, and M1 timings don't transfer | Host proxy: stack-only RTF, per-module p50/p95, E-core run (`taskpolicy -b`), a labelled ±2× target estimate; real number in Phase 4 |
| F2 | "Spoof flagged within seconds of onset" | **Ill-posed** | GPS-vs-reference consistency can't see offsets below ~10–16 m (≈4–6σ of the separation; single-epoch MDB ≈16 m), so slow drag-offs take tens of seconds. Coloured GNSS noise makes it worse | Committed attack envelope (steps and fast drags in seconds), a bounded spoof-induced deviation for slow drags, and latency measured from detectability (Phase 3 exit 2) |
| F3 | "~2.8 m median error" (student work) | **Untraceable** | No UAV source found. Nearest: NGPS 2.94 m RMSE on Google imagery (unusable here); a UAV-VisLoc student project at 3.85 m median | Measured target: sim cross-epoch fix median ≤3 m, p95 ≤8 m; a report-only real-data check (ALTO subset) against NAIP's ±4 m contract tolerance |

**Feasible with caveats.** The plan absorbs these; veto any you disagree with.
- **F4 Live camera mode.** Deliver KLT tracks plus a live **up-to-scale** monocular VO trajectory labelled "scale: arbitrary". That meets the letter of the feature. Metric live VIO is an optional 1-day spike using the base SPU IMU via a root helper; M1 Pro was reported working by one user on macOS 26.3, untested on 27. A real metric mode waits for Phase 4.
- **F5 Optical flow + rangefinder.** The SPEC's aiding layer needs a rangefinder that its sensor list omits, so we add a 120 m lidar. Flow is modelled and fused **only below 20 m AGL** (takeoff/landing): low-resolution flow modules oscillate above that (PX4 docs), and flow from the VIO camera would double-count its pixels. Range and flow refer to the DSM surface, not bare earth.
- **F6 "Your 3D simulator"** doesn't exist, so we build it: kinematic flatness trajectories, and MuJoCo only to draw pixels and cast rays. No physics engine.
- **F7 "Satellite" imagery.** Google, Bing, Esri and Mapbox terms forbid caching. We use **NAIP aerial orthoimagery** (US public domain) plus USGS 3DEP, and call the component the "orthoimagery matcher".
- **F8 <1% drift** is feasible on EuRoC/TUM-VI (published OpenVINS 0.07–0.42% ATE/length). It does not transfer to large-depth UAV flight: OpenVINS drifts 0.6–1.4% on KITTI, and a published 2.8 km UAV flight diverged. The sim target is therefore separate.
- **F9 OpenVINS** is GPL-3.0 (this repo is MIT), and upstream is dormant (no commits since 2025-11-30). It runs as a separate process behind a generic protocol, with our own numbered patch series.
- **F10 Frontend wording.** "Selects keyframes; outputs residuals" doesn't match OpenVINS internals (clone window; residuals are internal). We accept OpenVINS's structure and expose its tracking health.
- **F11 "Leans on inertial + baro."** OpenVINS doesn't coast without frames, and consumer-IMU coasting drifts tens of metres per minute. The backend switches to IMU-preintegration outage mode, and the forest segment is judged on NEES and recovery, not absolute error.
- **F12 "No discontinuity"** holds only by design (§3). It is defined numerically in Phase 3 exit 4.
- **F13 Deterministic replay** comes in tiers:
  - A: bit-identical sensor streams;
  - B: bit-identical images for the same machine and manifest;
  - C: estimator outputs within tolerance across platforms.
  The matcher runs on CPU in replay. Native vs container is always compared with tolerances, because libc++ and libstdc++ RNG distributions differ.
- **F14 EuRoC** is "In Copyright – Non-Commercial". Portfolio use is fine; pitch numbers lead with TUM-VI (CC BY 4.0) and the sim.
- **F15 DoD timeline** (unspecified in the SPEC), fixed in `demo.yaml`: Flagstaff AZ, a **2.0 km** route at 100 m AGL and 9 m/s.

  | Time (s) | Event |
  |---|---|
  | 0–5 | Stationary (static init) |
  | 5–25 | Climb |
  | 55–60 | Jam ramp |
  | 60–120 | GPS outage |
  | 120 | GPS re-acquired by a spoofer (seamless capture) |
  | 125 | 1 m/s drag-off begins |
  | ≈150–195 | Forest (Mars Hill) |
  | ≥30 s after | Textured tail |
  | ≈250–275 | Landing (flow + lidar) |
- **F16 "PX4/ArduPilot could consume it without changes"** means no *code* changes. Autopilot *parameter* changes are required. PX4 uses only the covariance diagonal and ArduPilot uses sqrt(trace).
- **F17 "Map-aided error in the low single meters"** holds on textured segments where fixes exist. Forest, water and rooftop-canyon legs get few fixes.

## 3. Architecture
```
 dss.sim (render proc) | dss.datasets | dss.live      SensorSample stream (int64 ns, calibrated, time-synced)
        ▼
 dss.sensors ──► dss.vio.ovclient ══ stdin/stdout lockstep ══► ovserver (C++, GPL-3.0, OpenVINS)
        │            ▲ pose, vel, 6×6 cov, clone-pair Δpose+cov, track health, fast-propagate, reanchor
        ├─► dss.aiding: ortho matcher (worker proc), lidar, baro, mag, flow (≤20 m AGL)
        ├─► dss.integrity.gnss_gate ◄── GNSS            (the ONLY path into fusion; import-linter contract)
        ▼
 dss.fusion: ALL (REF + gated GPS) | REF (never sees GPS)     ES-EKF fallback behind the same interface
        ▼
 dss.integrity: health vector, χ²/CUSUM, FSM ──► published = REF ⊕ clamp(ALL−REF), bled; HPL(t)
        ├─► dss.output (MAVLink 2 VISION_POSITION_ESTIMATE, 30 Hz, NED)
        ├─► dss.dashboard (Rerun)            └─► dss.eval (metrics, NEES, reports)
```
- **REF stays independent of GPS.** Everything REF touches is GPS-free:
  - Initialization: the surveyed home pose, or the first accepted map fix from a global search.
  - Yaw: mag + map only. Mag hard/soft iron is calibrated pre-flight; only the residual is simulated.
  - Matcher search prior: from REF only.
  - **ovserver re-anchoring:** only from REF's state and marginals, never ALL or the published state. The VIO, and with it the Doppler-vs-VIO test, therefore stays GPS-free.
- **Published pose.**
  - clamp = K·√λmax(PSD(P_REF − P_ALL)) with **K = 5**, the same threshold as the ALL-vs-REF separation test that triggers SUSPECT.
  - Every absolute correction is bled at ≤0.5 m/s and ≤2°/s, applied as δp plus yaw δψ about the *current published position*.
  - The published covariance carries the un-bled offset.
  - On a spoof alarm, ALL is re-seeded from REF (never by removing factor indices).
  - `HPL(t) = 5.68·√λmax(horizontal P_pub)` is the per-epoch horizontal protection level: 1e-7 integrity risk for a 2-D Gaussian. It is shown on the dashboard.
- **Vision outage.** VIO increments are excluded, and the backend uses NavState + `CombinedImuFactor` over non-overlapping intervals. On recovery, ovserver is re-anchored from REF: a fresh VioManager with covariance from REF's marginals, not `initialize_with_gt`'s hard-coded static covariance.
- **Frames and time (ADR-0002).**
  - Frames: ENU world at the AOI origin, FLU body, RDF camera.
  - Quaternions: Hamilton `[w,x,y,z]` in Python. OpenVINS JPL is converted only at the protocol boundary, and NED/FRD only at the MAVLink boundary.
  - Time: int64 ns in Python; float64 seconds only inside ovserver.
- **Run modes (ADR-0003).**
  - `--lockstep` is deterministic; its `metrics.json` has no wall-clock fields.
  - `--realtime` writes `timing.json` (RTF, rates, latency).
  - The renderer runs in its own process behind a bounded queue.
- **Process boundary.** It keeps GPL out of the MIT package and isolates OpenVINS's `std::exit` calls and stdout printing. It lets Python use opencv-python while ovserver links OpenCV 4.14. The same argv runs natively or under `docker run -i`.

## 4. Library choices (primary → fallback if it won't build or behave on Apple Silicon)
| Component | Primary | Fallback |
|---|---|---|
| Env | uv 0.12.19 + uv-managed CPython 3.13.15 | python.org 3.13 + `SSL_CERT_FILE`=certifi |
| Sensor layer, integrity, GNSS gate, flow | In-house numpy/scipy; hypothesis for property tests | — |
| **VIO core** | **OpenVINS @`69488123ed9362dd44b6f28e7f4680abbff1442b`** (v2.7 can't build against Ceres 2.2). Native build: `eigen@3` 3.4.1, OpenCV 4.14 minimal (KleidiCV off) and Ceres 2.2 (miniglog) from source into `$HOME/.cache/dss/cpp`, Homebrew boost 1.92 | ① Same source in `ubuntu:24.04` **linux/arm64** (upstream-CI stack, zero patches), same pipe protocol. ② rpng/sqrtVINS (LGPL-3.0, same API, no Ceres) or rpng/MINS (GPL-3.0, macOS-arm64 CI, in-filter position updates). ③ Basalt 0.1.7 (BSD-3, prebuilt aarch64-apple-darwin); optimization-based, so a SPEC deviation |
| **Frontend (VIO)** | OpenVINS `ov_core` TrackKLT (FAST + pyramidal KLT) inside ovserver | Container TrackKLT; or cv2 KLT tracks fed through `feed_measurement_simulation` in the same ovserver |
| **Fusion backend** | **GTSAM 4.3.0** (cp313 universal2, includes `gtsam_unstable`): `IncrementalFixedLagSmoother`, lag 10–20 s, GaussNewton, **QR**. Built-in factors: `BetweenFactorPose3`, `GPSFactorArm`, `BarometricFactor`, `MagPoseFactorPose3`, `AttitudeFactor`, `PriorFactorPose3`, `CombinedImuFactor` (outages only). `CustomFactor` only for 1-D range-to-DSM and 2-D flow at ≤10 Hz | numpy **error-state EKF** (Sola 2017 + stochastic cloning), CI smoke-tested from Phase 2. It is independent of the single, fresh GTSAM wheel (4.2.2 needs numpy<2) |
| Ortho matcher | Tier A: orthorectify + OpenCV NCC/phase correlation (CPU, deterministic) | Tier B (stretch): kornia 0.8.3 ALIKED/DISK + LightGlue on MPS in a restartable worker, or XFeat on CPU. A CI license gate bans SuperPoint (non-commercial) and EfficientLoFTR/MatchAnything (PRL) |
| Geo | NAIP via Planetary Computer STAC (pystac-client 0.9 + planetary-computer 1.0; 45-min tokens, so extract once), rasterio 1.5.1 (GDAL 3.12.4), pyproj 3.8.0 (+ GEOID18 grid), 3DEP DSM/DEM, Microsoft building footprints (CDLA-2.0) | USGS NAIP ImageServer |
| Sim renderer | MuJoCo 3.14.0 render-only, `MUJOCO_GL=cgl` set before import (verified on this Mac: windowless CGL, 16384 px textures) | ModernGL 5.12 standalone CGL; wgpu + pygfx (Metal) as the escape hatch |
| Sim models | numpy PCG64 via `SeedSequence(root, spawn_key=(DOMAIN, stream_id))` (`root.spawn()` forbidden, it collides); pygeomag 1.1.0 (WMM2025); min-snap/flatness trajectories | ppigrf (IGRF-14) |
| Python OpenCV | `opencv-python-headless`, version pinned after the Phase 0 smoke test for the arm64-macOS KleidiCV resize crash (opencv#29794) | Grayscale-only paths |
| **Frontend (dashboard)** | **Rerun 0.38.1** pinned, behind `dashboard/rerun_sink.py`: native viewer, `rr.spawn(connect=False, memory_limit="3GB")` + `rr.set_sinks(GrpcSink, FileSink)` | FastAPI + WebSocket + three.js/r3f + uPlot (Node 25 and pnpm present), or Lichtblick + MCAP |
| Output | pymavlink 2.4.50 (LGPL import OK), `MAVLINK20=1` set before import | Hand-packed MAVLink 2 frames (crc_extra 158) |
| Eval | In-house metrics; posyaw alignment ported from MIT rpg_trajectory_evaluation | `evo` 1.37.1 CLI via `uv tool install` (GPL, isolated, never imported) |

## 5. Phases
**Metric definitions (frozen in `docs/metrics.md` in Phase 0; criteria below cite them):**
- **ATE:** posyaw-aligned on datasets; no alignment in sim, which shares the world frame. Association within 0.01 s.
- **RPE:** 8/16/24/32/40 m segments on datasets, 100–800 m in sim.
- **Drift %:** 100·ATE/path length. End-point drift is aligned on the first mocap segment and measured on the last.
- **Relocalization:** success = error ≤5 m / attempts; precision; false-accept (>20 m); time to first fix.
- **TTR** (time to recover after GPS loss) = t* − t_loss:
  - t* is the first time from which every 1 Hz sample in [t*, t*+10 s] has error ≤5 m and truth inside the 99% ellipse;
  - TTR = 0 if that already holds at t_loss;
  - a run is censored and fails if t* never occurs.
- **ANEES(t):** mean NEES over N independent seeds at 1 Hz. Pass = inside the two-sided 95% χ²(dN)/N band in ≥90% of epochs, with the time-average also inside. N is fixed per gate in advance. The *one-sided* variant used for raw VIO: ≤ upper bound in ≥90% of epochs, time-average ≥1.0.
- **NIS bands:** block bootstrap.
- **HMI epoch:** |p_pub − p_true|_h > HPL(t) with no active alert.
- **SID (spoof-induced deviation):** max_t |p_pub,spoofed(t) − p_pub,nominal(t)| on paired same-seed runs.
- **Spoof timing:**
  - t_onset = first epoch where the spoofed position departs from truth.
  - t_detectable = first time the true offset > MDB = 5.94·σ_innov (2-DoF, P_FA 1e-3, P_MD 1e-2).
  - Alarm = any entry into SUSPECT or REJECTED.
- **Divergence:** NaN/Inf, 0 MSCKF updates for >5 s, or error >1 m (datasets) / >5% of distance (sim).
- **RTF:** stack-only (ovserver + fusion + integrity + output) fed from recorded sensor logs.

### Phase 0: repo, sensor models, eval harness, dead-reckoning baseline
**SPEC covered:** sensor abstraction layer; sim sensor models (non-visual); ground-truth recorder; eval harness. Also retires the Phase 1 build risk and takes the first reading on R11.

| Task | Deliverable |
|---|---|
| 0.1 **OpenVINS build spike + large-depth probe** (1 day for the build, +½ day for the probe; §10) | `cpp/spike/*`, `cpp/patches/open_vins/*` (if needed), ADR-0001 |
| 0.2 Env bootstrap + `dss doctor` + budget ledger | `Brewfile`, `scripts/bootstrap_macos.sh`, `pyproject.toml`, `uv.lock`, `src/dss/doctor.py`, `docs/data-budget.md` |
| 0.3 Core: time, frames, rotations, RNG streams, config, run manifest | `src/dss/core/*`, ADR-0002, ADR-0003 |
| 0.4 Sensor layer: samples, Kalibr/EuRoC calibration + extrinsic chains, merged ordered stream, time offset/jitter/interpolation | `src/dss/sensors/*` |
| 0.5 Dataset readers (EuRoC ASL + TUM-VI EuRoC-format; 16→8-bit by range) and budgeted fetcher (stream-extract, remotezip) behind `dss fetch` | `src/dss/datasets/{asl,groundtruth,fetch}.py`, `configs/datasets/manifest.json` |
| 0.6 Sim models (see below) + recorder | `src/dss/sim/*`, `configs/sensors/*`, `configs/scenarios/*` |
| 0.7 Eval harness (all metrics above) + `docs/metrics.md` | `src/dss/eval/*` |
| 0.8 Dead-reckoning baseline: ES-EKF propagation only (reused as the Phase 2 fallback) + a baro-alt/mag-yaw variant | `src/dss/baselines/dead_reckoning.py`, `reports/phase0/*` |

The 0.6 sim models:
- **Trajectories:** flatness-based, with takeoff/landing.
- **IMU:** Kalibr/OpenVINS discretization; EuRoC and TUM-VI presets.
- **Baro:** ISA + Gauss-Markov bias + lag.
- **Mag:** WMM2025 + residual iron.
- **GNSS:** M9N-class position + velocity; coloured Gauss-Markov error; C/N0 jamming; step, drag-off and meaconing spoofs.
- **Lidar and flow:** OPTICAL_FLOW_RAD model at ≤20 m AGL.

**Exit criteria**
1. **Build + probe decision (ADR-0001).**
   - OpenVINS@69488123 plus the recorded patch series (possibly empty) builds natively with the §10 recipe.
   - `test_sim_repeat`, built without NDEBUG (`nm -u` shows the platform assert symbol: `___assert_rtn` on macOS, `__assert_fail` on glibc), prints `success! they all are the same!`.
   - `run_simulation` exits 0.
   - `sim_rmse` on `rpng_sim` gives a finite position RMSE ≤0.5 m and a finite NEES.
   - `klt_smoke` keeps ≥100 tracks across 20 synthetic frames with parallax, and `cv::imread` works.
   - The **large-depth probe** (flat 98–102 m and relief 80–120 m variants, 5 seeds each) records ATE/length, divergence, NEES and per-epoch update counts. It is informational. A divergence forces a Phase 1 re-plan before renderer work; zero updates in the first 10 s means the probe is misconfigured, not that VIO diverged.
   - Recipe errors are fixed, never grounds for switching to the container.
   - If native fails for platform reasons within 1 day, the linux/arm64 container passes the same checks (Tier C comparison). If both fail within 2 days, ADR-0001 selects sqrtVINS or MINS and its sim-repeat check passes.
2. **Environment.**
   - Gating: `uv run dss doctor` reports CPython 3.13.15, HTTPS OK, free disk ≥15 GB (warn <25 GB), and the ledger within budget.
   - Advisory (`dss doctor --full`, `machine` marker): group imports; `IncrementalFixedLagSmoother` constructs; versions match `uv.lock`; rasterio PROJ = pyproj PROJ; one CGL frame renders; cv2 CV_8UC3 `INTER_LINEAR` resize smoke passes.
3. **Tests.** `DSS_PHASE_GATE=1 uv run pytest` is green locally, with no gated test skipped. CI on `macos-26` is green for `-m "not data and not gpu and not slow and not machine"`, checked out under a path containing a space. `git ls-files --error-unmatch tests/environment/test_environment.py` passes.
4. **Sensor models:**
   - noise-free strapdown closes to <1 cm / <1e-4 rad after 60 s;
   - flatness rates match 10 kHz finite differences to <1e-6 rad/s, with no jumps at joints;
   - **Allan** (3 h static, 200 Hz, `slow` marker): a WLS fit of σ²(τ)=N²/τ+K²τ/3 over τ∈[0.05, 1080] s gives N̂ within ±5% and K̂ within ±20% of config, for every axis of both presets;
   - ISA round trip <1e-6 m; WMM2025 official test values (committed fixture) within 0.1 nT / 0.01°;
   - **GNSS** static, 20 seeds × 1 h at 5 Hz: pooled CEP 1.50 m ±5%; σ_v 0.050 m/s ±10%; fitted τ_pos/τ_vel within ±20%. Below C/N0_min (param, default 28 dB-Hz) there is no fix within 1 epoch; reacquisition only after the configured hold;
   - the spoof injector reproduces its profiles exactly;
   - flow at ≤20 m AGL recovers horizontal velocity within 5% RMS;
   - lidar σ = √(0.1²+(0.05r)²), with out-of-range returns marked invalid.
5. **Sensor layer.**
   - Injected clock offset/jitter is recovered within 0.1 ms.
   - The merged stream is time-ordered with a deterministic tie-break (hypothesis).
   - Extrinsic chains compose/invert to 1e-12.
   - Parsed EuRoC/TUM-VI calibrations equal OpenVINS's shipped `T_imu_cam` to 1e-9 (golden, `data` marker).
6. **Eval harness.**
   - posyaw recovers an injected yaw+translation to <1e-9 m.
   - SE3 ATE equals `evo_ape -a` within 1 mm on TUM-VI room1 GT vs a perturbed copy.
   - RPE handles segments longer than the path.
   - End-point drift and TTR self-tests pass.
   - DR error growth matches ½·b_a·t² within 1%.
7. **Covariance plumbing.** DR position ANEES (N=50, 60 s) is inside [2.36, 3.72] per the rule above.
8. **Determinism tier A.** Two fresh seed-42 runs give identical SHA-256 for every sensor stream and GT. Adding a stream ID changes no existing hash.
9. **Artifacts and data.**
   - `reports/phase0/` has DR reports for sim, TUM-VI room1 and EuRoC MH_01 (from the HF mirror if ETH returns 429).
   - The manifest records SHA-256 on first fetch (trust-on-first-use), and a re-fetch re-hashes identically.
   - One outer EuRoC zip, streamed from any byte-identical mirror, is hashed against ETH's published checksum. If ETH's checksum page stays unreachable after retries over 24 h, provenance is recorded as unverified.

### Phase 1: VIO core integrated, hitting the drift target
**SPEC covered:** VIO frontend and estimator core; synthetic pinhole rendering.
- **1.1 ovserver** (`cpp/ovserver/`, GPL-3.0; separate CMake project, never `add_subdirectory(ov_msckf)`; container via `cpp/docker/Dockerfile`):
  - I/O hygiene: `dup2` stdout→stderr; protocol on the saved fd; length-prefixed strict request/response.
  - Input validation before OpenVINS: sizes and masks; IMU fed first.
  - Emits: pose, velocity, 6×6 `[p,θ]` covariance, clone-pair Δpose + joint covariance (≤0.5 s apart), and frontend health (n_tracked, n_msckf_used, n_slam, ≤200 `(id,u,v)` tracks).
  - Adds `fast_state_propagate` with **PR #549 backported**, and `reanchor(t, x17, P15)`.
  - Builds: Release for timing; a no-NDEBUG build for tests.
  - Configs and determinism overrides live in `cpp/ovserver/config/`.
- **1.2** `src/dss/vio/{protocol,ovclient,backend,synthetic}.py`: codec, client (EOF → `VioFailure`), `VioBackend` interface, and a synthetic-increment generator so fusion never waits on C++.
- **1.3** Dataset runs: TUM-VI rooms 1–6 + corridor4, then all 11 EuRoC sequences, mono + stereo, GT init. Static-init success rate is reported separately.
- **1.4a Flagstaff world + route** (`src/dss/geo/{aoi,naip,dem,cache,crs}.py`, `src/dss/sim/world/tiles.py`):
  - `dss geo fetch --aoi flagstaff --epochs 2023` gets the 0.3 m render-epoch crop (≤0.5 GB) plus a 3DEP DSM crop;
  - ground tiles with DSM relief;
  - `configs/routes/flagstaff.yaml` (final 2.0 km waypoint route with takeoff/landing);
  - `configs/scenarios/denied_route.yaml` listing the textured, forest and urban segment IDs.
- **1.4 Renderer v1** (`src/dss/sim/render/{scene,mujoco_backend,camera,photometric}.py`):
  - Camera: a mocap body; centered principal point, fx=fy, Renderer size == `resolution`.
  - Scene settings: shadows off; linear colorspace; `inertia="shell"`; group culling, never alpha-0 (that breaks `mj_ray`).
  - Photometrics in numpy: exposure, Koschmieder haze, adaptive blur, shot/read noise, vignetting, illumination-step events.
- **1.5 Rig sweep and soak.**
  - Sweep at 5 seeds: {40, 80, 120 m AGL} × {nadir mono, 45° oblique mono, oblique stereo}, with camera-IMU offset/jitter injected; pick the rig.
  - Final gate at 25 seeds.
  - Soak: 5 seeds × 7 legs of alternating direction (≈14 km, ≈26 min).
  - A **GT-derived re-anchor** means x17 = GT ⊞ δ, δ ~ N(0, P15_proxy), with ovserver told P15_proxy (σ_h 2.4 m, σ_z 1 m, σ_roll/pitch 0.5°, σ_yaw 1°, σ_v 0.2 m/s, biases from the IMU preset).
- **1.6** Minimal Rerun debug adapter (`src/dss/dashboard/rerun_sink.py`).

**Exit criteria.** All ATE gates use posyaw alignment, ov_data GT (with the corrected V1_01) and the shipped configs plus determinism overrides.
1. **Protocol.**
   - 10,000 frames at OpenVINS verbosity ALL with 0 framing errors.
   - A malformed frame is rejected without exiting.
   - A killed ovserver surfaces as `VioFailure` within 1 frame.
   - Health/track fields are schema-tested, and n_tracked matches TrackKLT internals on TUM-VI room1.
2. **Determinism.** Two TUM-VI room1 runs give byte-identical pose+cov logs. Native-vs-container is reported only if both builds exist.
3. **EuRoC stereo.** Each sequence ≤1.5× ICRA-2020 (V1_01 0.061, V1_02 0.056, V1_03 0.057, V2_01 0.053, V2_02 0.048 m); average ≤0.08 m.
4. **EuRoC mono.** All 11 without divergence, each ≤1.5× reference (MH01–05: 0.10/0.17/0.12/0.25/0.41; V101–V203: 0.06/0.06/0.07/0.10/0.06/0.14 m).
5. **Drift (SPEC target).** Drift ≤1.0% on every EuRoC sequence and TUM-VI room1–6 (expected ≤0.5%); corridor4 end-point drift ≤1%; RPE reported.
6. **TUM-VI rooms 1–5.** Mono average ≤0.10 m, stereo ≤0.12 m (1.5× stock FEJ-AID 0.067/0.079 m).
7. **Timing.**
   - Release build, shipped threading, EuRoC V1_01: mono p99 ≤50 ms/frame and estimator RTF ≥1.0 at 20 Hz.
   - Stereo and single-thread deterministic timings are reported, not gated.
   - ovserver RSS <1 GB, with a least-squares slope ≤1 MB/min over minutes 5–25 of the soak.
8. **Renderer.**
   - Projection residual <0.2 px vs K; RGB/depth aligned.
   - ≥20 camera frames/s for the chosen rig *as configured* (cameras × blur subframes × RGB+depth, 640×480) over 1,000 route frames; below that, switch to ModernGL.
   - Tier B: 100 frames rendered twice under the same manifest give identical SHA-256.
9. **Sim VIO** (chosen rig, Flagstaff ground-tile scene, 25 seeds):
   - no divergence over the route with GT-derived re-anchors every ≤60 s;
   - unaided distance-to-divergence and drift reported (target ≤2%);
   - position ANEES meets the one-sided rule (≤4.03), both overall and in the 30 s after each re-anchor. Position/orientation NEES reported against ICRA's 1.88/2.20;
   - an injected 5 ms camera-IMU offset recovered to ≤0.5 ms, and 10 ms jitter causes no divergence.
10. **Budget ledger** within limits.

### Phase 2: orthoimagery aiding + fusion backend (bounded long-range error)
**SPEC covered:** aiding layer (orthoimagery matcher, flow + rangefinder, baro/mag priors); fusion backend.
- **2.1 Geo.**
  - Map-epoch candidates: 2017-06-27 (summer) and 2021-11-08 (long shadows, possible snow), matched against the 2023 render epoch. Pick one and record the season gap.
  - Co-register the epochs.
  - Pin datums: NAD83/UTM 12N + NAVD88, with GEOID18 via `pyproj sync`.
- **2.2 Full scene** (`src/dss/sim/world/{buildings,trees}.py`): buildings from footprints + DSM heights; trees from NAIP NDVI, merged per tile; a procedural high-rise block on one street for Phase 3's urban canyon. The route itself is unchanged from 1.4a.
- **2.3 Matcher Tier A** (`src/dss/aiding/{ortho,match_ncc,fix}.py`):
  - Covariance: correlation-peak curvature, floored by the map registration error.
  - Accept a fix only on peak-ratio pass, scale within ±15%, yaw within ±10° and χ²₂ ≤9.21 against REF's prediction. After an outage, require 2 consecutive consistent fixes.
  - Search window: footprint/2 + 3σ_REF, clipped to [50, 500] m; beyond that, tiled relocalization.
- **2.4 Fusion** (`src/dss/fusion/{backend,gtsam_backend,eskf_backend,conversions,keys}.py`):
  - ALL = REF + map fixes in Phase 2; GPS arrives only via the Phase 3 gate.
  - Graph hygiene: ENU graph; QR; every key timestamped; every covariance symmetrized and PD-checked.
  - VIO increments at 5–10 Hz, scaled by α tuned on **denied** segments.
  - The JPL→GTSAM covariance mapping is Monte-Carlo-tested (10⁴ samples, ≤5% Frobenius).
  - Outage mode as in §3.
- **2.5 Aiding models** (`src/dss/aiding/{rangefinder,baro,mag,flow}.py`): lidar range-to-DSM with canopy returns; baro bias random walk; mag field/inclination disturbance check; flow fused only at ≤20 m AGL.
- **2.6** ES-EKF fallback behind the same interface.
- **2.7** Real-data spot check (report-only): Tier A on an ALTO public subset (≤2 GB, BSD-3 repo, NAIP-referenced).

**Exit criteria.** Textured segments are the IDs in `denied_route.yaml`.
1. **Geo.** STAC returns ≥2 epochs. Co-registration residual <1 m RMS. Map-epoch choice and season gap recorded.
2. **Tier-A speed.** NCC ≤20 ms per 512² patch (fresh process).
3. **Fix quality** (sim, cross-epoch, 100 m AGL, textured segments):
   - median ≤3.0 m, p95 ≤8 m;
   - success ≥80% of attempts;
   - false-accept ≤1% of accepted fixes;
   - fix ANEES (2-DoF; 4 fixes/seed × 25 seeds, ≥500 m apart) in [1.63, 2.41];
   - same-epoch vs cross-epoch gap reported;
   - time to first fix after σ_REF >50 m ≤30 s (25 trials);
   - ALTO median/p95 reported.
4. **Bounded error** (no GPS, 25 seeds), on textured segments:
   - per-seed horizontal RMSE ≤5 m, pooled p99 ≤10 m, fused median ≤3 m;
   - 0 epochs above HPL(t), computed from the fused covariance (the published output only exists from Phase 3);
   - median over seeds of RMSE_VIO/RMSE_aided ≥3, and RMSE_aided ≤ RMSE_VIO on every seed;
   - REF ≤3.4 m RMS (the fitted σ_REF/τ_REF feed the Phase 3 Monte Carlo).

   Also:
   - forest leg reported separately;
   - vertical p95 ≤3 m, and ≤0.5 m at touchdown with lidar + flow (with vs without flow reported);
   - map-aided soak (5 seeds × 7 legs): textured p99 ≤10 m, 0 epochs above HPL(t), per-leg RMSE slope ≤0.1 m/leg.
5. **Honest covariance** (fused):
   - 3-DoF position ANEES (N=25) in [2.12, 4.03] and 6-DoF pose ANEES (N=25) in [4.72, 7.43], also as a function of time since the last fix;
   - NIS exceedance at the 95% gate inside bootstrap bounds;
   - clean-run yaw error ≤ WMM declination uncertainty + 1°.
6. **Robustness.**
   - A 30-min soak of VIO BetweenFactors only, plus a 120 s CombinedImuFactor outage, raises no `IndeterminateSystemException`.
   - p95 backend update ≤20 ms for 2 smoothers at 10 Hz, CustomFactors included.
   - The ES-EKF CI smoke test (synthetic increments) passes.
   - License gate green.
7. **Full-scene re-check.** Phase 1 exits 8 (fps as configured) and 9 (no divergence, 5 seeds) re-run on the 2.2 scene. fps <20 switches to ModernGL, and the rig is re-confirmed.
8. **Budget ledger** within limits.

### Phase 3: integrity monitor, GPS attack interface, output, dashboard, demo
**SPEC covered:** integrity monitor, GPS attack interface, output interface, dashboard, scenario suite, live camera mode.
- **3.1 Health vector** (`src/dss/integrity/{vision,health}.py`):
  - Sources: VIO, MAP, GPS, BARO, MAG, RANGE, FLOW and IMU (saturation, dropout, bias jump).
  - Vision detectors: feature collapse (<20% of 200), blur (Laplacian variance vs running median), illumination shock.
  - Hysteresis, noise inflation, outage mode, and re-anchor from REF on recovery.
- **3.2 GNSS gate** (`src/dss/integrity/{gnss_gate,cusum,fsm}.py`):
  - States: UNAVAILABLE → PROBATION → TRUSTED → SUSPECT → REJECTED.
  - A χ²₂ pre-gate exceedance (13.82) only excludes that sample. SUSPECT is entered only when:
    - a CUSUM (position gap Gauss-Markov-corrected; velocity gap Doppler vs VIO with a yaw-error term) crosses h;
    - the ALL−REF separation exceeds K·σ_ss (K = 5, §3); or
    - 3 consecutive pre-gate failures occur.
  - **PROBATION** starts at re-acquisition and lasts ≥ capture hold + p99 latency(1 m/s) + 5 s (≈35 s provisional; re-derived after the envelope is committed). It ends only when both CUSUMs < h/2 and σ_REF ≤5 m.
  - **Corpora:** real ovserver post-VIO logs from distinct rendered runs (open-sky GNSS, no jam, spoof or NLOS; route incl. forest/twilight; each run ≥ PROBATION + 300 s) in float32+zstd (≤2 GB).
    - Set T (≥10 h, seeds 1000–1499) is for tuning; thresholds are frozen in `configs/integrity/thresholds.yaml` (hash recorded in the manifest).
    - Set H (≥10 h, seeds 2000–2499) is held out and used for the false-alarm test **and** the spoof sweep.
- **3.3** Published-output logic as in §3 (`src/dss/integrity/separation.py`).
- **3.4 Output** (`src/dss/output/{mavlink_vpe,frames,timesync}.py`):
  - VISION_POSITION_ESTIMATE at 30 Hz, MAVLink 2, NED.
  - Covariance: conservative diagonal, explicit 21 floats.
  - `reset_counter` only on a true re-init; TIMESYNC responder; SET_GPS_GLOBAL_ORIGIN; never ODOMETRY on the same link.
  - Also delivers `docs/autopilot-integration.md` and `src/dss/eval/px4_ev_gate.py` (an EKF2-style 5-SD gate emulator driven by the sim IMU).
- **3.5 Scenario suite** (`configs/scenarios/{urban_canyon,forest,twilight,aggressive,gps_cutover,dragoff_spoof,demo}.yaml`, `src/dss/sim/gnss_geometry.py`):
  - `mj_ray` sky mask + NLOS biases on the high-rise block;
  - illumination-shock events;
  - thresholds frozen before the first suite run (commit SHA printed on the scoreboard).
- **3.6 Dashboard** (`src/dss/dashboard/{blueprint,rerun_sink,scoreboard}.py`), one fixed 3-column blueprint:
  - column 1: 3D view with drone, GT/estimate trajectories and 3σ ellipsoid;
  - column 2: camera + tracks, error series with HPL, StateTimeline attack timeline;
  - column 3: markdown scoreboard, health bars, event log.
  - `.rrd` only for demo/`--debug` runs.
  - `scripts/record_demo.sh` replays `run.rrd` at 1× and captures it with ffmpeg avfoundation (fallback: `screencapture -v`).
- **3.7 Live camera** (`src/dss/live/{webcam,mono_vo}.py`): tracks + up-to-scale VO. The SPU-IMU spike is optional and not gated.
- **3.8** PX4 SIH acceptance (native, else the multi-arch `px4io/px4-dev` image), then the demo run.

**Exit criteria**
1. **Definition of done.**
   - `uv run dss demo --seed 42 --lockstep` runs the F15 timeline end to end.
   - The one-screen dashboard shows continuous pose, detection flags, failover events, and final numbers taken after the post-forest recovery.
   - It emits `runs/demo-42/{run.rrd,metrics.json}`. A re-run with the same manifest (CPU matcher) gives a byte-identical `metrics.json`.
   - `scripts/record_demo.sh runs/demo-42` produces `demo.mp4` (≥30 fps, full blueprint visible).
2. **Spoof detection.**
   - First, re-run the latency Monte Carlo with the Phase 0 coloured GNSS model and the Phase 2 σ_REF/τ_REF, then **commit the envelope** (provisional: 20 m step ≤1.2 s; 2 m/s ≤8 s; 1 m/s ≤20 s; 0.5 m/s ≤35 s). Any committed value above its provisional one needs your sign-off.
   - Sweep on set H over {20 m step; 0.1, 0.3, 0.5, 1, 2, 3 m/s; 0.01, 0.05, 0.2 m/s²}: ≥100 distinct segments per point, ≥300 s after onset (undetected runs are censored and reported).
   - Pass:
     - p95 latency from t_onset ≤ the envelope at every point;
     - p95(t_flag − t_detectable) ≤5 s;
     - SID ≤15 m for every rate ≤0.5 m/s and ≤0.2 m/s².
   - Demo: flagged before PROBATION ends, and |p_pub − p_true|_h ≤ HPL(t) from 120 s to t_flag.
3. **False alarms.** 0 alarms on held-out set H with frozen thresholds, i.e. λ ≤0.3/h at 95%. Stretch: ≥30 h, i.e. ≤1 per 10 h.
4. **Failover continuity** (all scenarios, 5 seeds):
   - 0 `reset_counter` increments;
   - jump |p_pub(k) − (p_pub(k−1) + Δp_prop(k))| ≤ max(0.05 m, 3σ of the published increment) at every 30 Hz sample;
   - |Δp_pub − Δp_true| ≤0.05 m per sample on vision-healthy segments;
   - bleed ≤0.5 m/s and ≤2°/s;
   - 0 rejections and no rejection streak ≥1 s in the `px4_ev_gate` replay.
5. **Honest output.**
   - Published 3-DoF ANEES (N=5) in [1.25, 5.50], bleed intervals included; attitude covariance consistent by the same rule.
   - 0 HMI epochs in every scenario.
   - REF independence: in `--lockstep`, REF's output log is byte-identical with GNSS present vs absent (same seed), including across a forest outage while GPS is TRUSTED.
6. **Degradation.**
   - Forest/twilight: VIO FAILED within 1 s of collapse, announced (event, health bar, inflated VPE covariance).
   - Blur detector (≥1 px/exposure): flagged within 0.5 s, ≥95% detection, false alarms in ≤1% of 1 s nominal windows.
   - Illumination shock: flagged within 2 frames, ≤1 false alarm per 10 min.
   - After the forest, REF is back within B = 10 m ≤10 s after the next accepted fix. Published horizontal error is ≤ B by t_fix + 10 s + (e_fix − B)/0.5 m/s + 10 s.
7. **Per-source fault injection** (5 seeds each): each fault flagged in the health vector within 2 s, and honest GPS is never REJECTED. Faults:
   - false map fix 30 m;
   - baro step 10 m;
   - mag anomaly ≥5°;
   - lidar dropout + canopy bias;
   - flow scale error 30% on landing;
   - IMU saturation.
8. **Scenario suite** (6 × 5 seeds, frozen thresholds; scoreboard generated; TTR reported per scenario):
   - urban_canyon: NLOS-biased GPS never TRUSTED while its bias > HPL; 0 HMI.
   - aggressive (≤200°/s, ≤2 g): no divergence; blur true-positive rate ≥95%; RMSE ≤5 m.
   - twilight: map aiding disabled and announced below the illumination threshold.
   - forest: exit 6.
   - gps_cutover: exit 4 + TTR ≤10 s.
   - dragoff_spoof: exit 2.
9. **Output interface.**
   - Frames start with 0xFD and carry 21 finite covariance values.
   - Rate 30.0±0.5 Hz (`--realtime`); pymavlink round-trip passes.
   - p99 latency (VPE send − newest IMU delivered) <250 ms.
   - Import contract: only `gnss_gate` builds GPS factors.
   - PX4 SIH: "starting EV position fusion" within 5 s, and no "failing, resetting" or "stopping EV position fusion" over 300 s, with EV derived from SITL truth through `dss.output`. If neither native nor container SIH builds in 1 day, record it and rely on `px4_ev_gate` + golden tests.
10. **Dashboard, live mode, CPU.**
    - A 10-min `--realtime` loop: gRPC backlog never >1 s; viewer RSS <4 GB.
    - FaceTime camera: tracks ≥15 fps plus a live mono-VO trajectory labelled "scale: arbitrary" for ≥60 s with ≤1 re-init; a desk loop closes within 10% after Sim(3) alignment.
    - Stack-only RTF ≥1.0; E-core figures and a labelled target estimate published.
11. **Budget ledger** within limits.

**Stretch (not gating the DoD):**
- Tier B learned matchers + DSM PnP;
- NO-MAP smoother;
- ES-EKF route parity;
- native-vs-container parity;
- 30 h false-alarm proof;
- Pittsburgh AOI;
- ArduPilot SITL;
- SPU-IMU live VIO;
- a closed-loop "naive vs protected" clip.

## 6. Risks
**6a. Machine-specific** (sorted by likelihood, then impact)
| # | Risk | L/I | Tripwire | Mitigation |
|---|---|---|---|---|
| R1 | OpenVINS won't build. Causes: CMake 4 needs a policy override; ArUco on by default vs contrib-less OpenCV; custom-prefix Ceres; Eigen 5 / Ceres-Eigen lock; OpenCV 5 default; catkin-relative `sim_traj_path`; no exported CMake package | H/H | Task 0.1 | §10 recipe (verified line by line against pinned sources); numbered patches incl. PR #549; container → sqrtVINS/MINS |
| R2 | Disk exhaustion (75 GB free, Docker 28 GB) | H/M | `dss doctor` ledger | Ledger (§8); stream-extract; no persisted frames or depth; `.rrd` for demo/debug only; EuRoC images deletable after Phase 1 gates; ask before any Docker prune |
| R3 | Compute budget: ~30–35 wall-hours across gates; laptop sleep; thermal throttling; CGL verified only in a GUI session | H/M | measured RTF × planned sim-hours >48 h | Sweeps at 5 seeds; ≤2 workers; `caffeinate -dims`, plugged in, logged-in session; set H reused for FA and spoof sweeps |
| R4 | Broken Python HTTPS; polluted globals; pip `ninja` shadowing | H/L | `dss doctor` | uv-managed CPython; sanitized PATH + `-G Ninja -DCMAKE_MAKE_PROGRAM` in scripts |
| R5 | MuJoCo camera silently wrong (K coupled to `resolution`, principal-point signs, sRGB-as-linear, −Z/+Y), or CGL fails headless | M/H | projection test <0.2 px | Exact-K tests; `MUJOCO_GL=cgl` before import; GPU tests skipped in CI; ModernGL peer fallback |
| R6 | 16 GB RAM contention (≈1 GB per resident tile, MPS matcher, Rerun, Docker default 8 GB) | M/M | RSS per run | 2–4 tiles resident; Rerun `memory_limit=3GB`; restartable matcher; Docker ≤4 GB, build `-j2` |
| R7 | macOS TCC prompts (Screen Recording for `demo.mp4`, Camera for live mode) | M/M | `dss doctor --tcc` grabs 1 screen + 1 camera frame | Grant once; record interactively; fallback: ViewerClient screenshots → ffmpeg |
| R8 | Space in repo path breaks shell/CMake | M/M | CI checkout under a spaced path | Quote everything; C++ trees in `$HOME/.cache/dss/cpp` |
| R9 | Toolchain churn and upstream bugs: opencv@4 deprecation; **OpenCV KleidiCV arm64-macOS crash (opencv#29794)** in 4.14/5.0; Rerun 0.x breaking releases; kornia-rs 0.2.0; uv; Homebrew can't pin versions | M/M | `dss doctor` version drift + resize smoke | `uv.lock`; brew versions + tarball SHA-256s recorded; `-DWITH_KLEIDICV=OFF`; opencv-python pinned after the smoke test; adapters around Rerun/kornia |
| R10 | Dataset hosting (EuRoC ETH HTTP 429; unlabeled HF mirror) | M/M | fetch script | TUM-VI first; HF mirror + remotezip; outer-zip provenance check with an "unverified" fallback |

**6b. Project / algorithmic**
| # | Risk | L/I | Tripwire | Mitigation |
|---|---|---|---|---|
| R11 | VIO diverges at altitude or over long runs (large-depth failure; OpenVINS #481) | H/H | Task 0.1 probe; Phase 1 sweep + soak | Oblique/stereo rig option; lidar; re-anchor from REF; sqrtVINS/MINS |
| R12 | Dishonest covariance (correlated increments, IMU coasting, coloured GNSS, `initialize_with_gt` static covariance) | H/H | ANEES gates every phase | Clone-pair covariance; α tuned on denied segments; outage NavState; REF-marginal re-anchor |
| R13 | Inverse crime inflates matching accuracy | H/H | same- vs cross-epoch gap | Render 2023, match 2017/2021; 2.5D relief; photometrics; ALTO spot check |
| R14 | Spoof thresholds false-alarm under coloured noise; committed envelope worse than provisional | H/M | set H alarms; envelope commit | Tune on T, test on disjoint H; commit envelope after re-running the Monte Carlo with measured σ_REF; your sign-off if worse |
| R15 | GTSAM 4.3.0 is 9 days old: Cholesky pivot throw (#2782); untimestamped keys never marginalize | M/H | Phase 2 soak | QR; key/timestamp asserts; catch → re-seed; ES-EKF fallback |
| R16 | License contamination | M/H | CI license gate | GPL confined to `cpp/`; evo as an isolated uv tool; allantools dev-only; no bpy; no SuperPoint/PRL weights; EuRoC out of pitch material |
| R17 | MAVLink mis-accepted (v1 zero covariance, ODOMETRY double-feed, no TIMESYNC, PX4 `yaw_align`) or PX4 SIH won't build on macOS 27 | M/M | golden encode tests; SIH run | MAVLink 2 forced; one message type; TIMESYNC; container SIH; `px4_ev_gate` fallback |
| R18 | Datum errors (NAD83 vs WGS84 1–2 m; NAVD88 −23 m at Flagstaff; pyproj ships no grids) | M/M | round-trip tests | Sim in NAD83/UTM + NAVD88; GEOID18 grid pinned |

## 7. Exact Phase 0 file tree
```
denied-signal-stack/
├── .github/
│   └── workflows/
│       └── ci.yml                   # runs-on macos-26; uv sync --locked; ruff; mypy; pytest -m "not data and not gpu and not slow and not machine"; spaced checkout path
├── .gitignore                       # edit: + /data/ /runs/ *.rrd .DS_Store — existing lib/, env/, ENV/, *.manifest, *.log rules mean: no dirs named lib/env, no committed .manifest/.log files
├── .python-version                  # 3.13.15
├── Brewfile                         # cmake, ninja, eigen@3, boost, uv (bottles; versions recorded by doctor)
├── LICENSE                          # existing MIT
├── NOTICE                           # NAIP/FSA credit, TUM-VI CC BY 4.0, EuRoC non-commercial, WMM (NOAA), OpenVINS GPL boundary
├── PLAN.md
├── README.md                        # edit: quickstart
├── SPEC.md                          # existing
├── pyproject.toml                   # package dss; groups core/fusion/geo/render/vision/dash/mavlink/dev; markers data/gpu/slow/machine
├── uv.lock
├── configs/
│   ├── datasets/
│   │   └── manifest.json            # URL, bytes, SHA-256 (trust-on-first-use)
│   ├── rigs/
│   │   ├── sim_nadir_mono.yaml
│   │   └── tumvi_512.yaml
│   ├── scenarios/
│   │   ├── dr_route_2km.yaml
│   │   └── dr_static.yaml
│   └── sensors/
│       ├── baro_bmp390.yaml
│       ├── flow_px4flow.yaml
│       ├── gnss_m9n.yaml
│       ├── imu_adis16448.yaml
│       ├── imu_bmi160.yaml
│       ├── mag_default.yaml
│       └── rangefinder_sf11c.yaml
├── cpp/
│   ├── README.md                    # GPL boundary: everything under cpp/ that links OpenVINS is GPL-3.0
│   ├── patches/
│   │   └── open_vins/               # 000N-*.patch, only if Task 0.1 needs them
│   └── spike/
│       ├── CMakeLists.txt           # C++17; imported ov_msckf_lib target (no upstream CMake package) + OpenCV/Boost::filesystem/Eigen3
│       ├── LICENSE                  # GPL-3.0
│       ├── build_openvins_macos.sh
│       ├── klt_smoke.cpp
│       ├── make_traj.py             # stdlib-only: 2 km S-turn at 100 m AGL; "t x y z qx qy qz qw" space-separated, ≥20 Hz, z-up
│       ├── probe_large_depth.sh
│       └── sim_rmse.cpp
├── docs/
│   ├── adr/
│   │   ├── 0001-vio-build-and-process-boundary.md
│   │   ├── 0002-frames-units-time.md
│   │   └── 0003-determinism-and-run-modes.md
│   ├── data-budget.md               # ledger checked by dss doctor
│   └── metrics.md                   # frozen metric definitions (§5)
├── reports/
│   └── phase0/                      # generated and committed: DR sim / TUM-VI room1 / EuRoC MH_01 reports, NEES plot
├── scripts/
│   └── bootstrap_macos.sh           # sanitized PATH; brew bundle; uv python install 3.13.15; uv sync; uv tool install evo==1.37.1; dss doctor
├── src/
│   └── dss/
│       ├── __init__.py
│       ├── cli.py                   # dss doctor | fetch | sim | baseline | eval
│       ├── doctor.py
│       ├── py.typed
│       ├── baselines/
│       │   ├── __init__.py
│       │   └── dead_reckoning.py
│       ├── core/
│       │   ├── __init__.py
│       │   ├── config.py
│       │   ├── frames.py
│       │   ├── manifest.py
│       │   ├── rng.py
│       │   ├── rotations.py
│       │   └── time.py
│       ├── datasets/
│       │   ├── __init__.py
│       │   ├── asl.py
│       │   ├── fetch.py
│       │   └── groundtruth.py
│       ├── eval/
│       │   ├── __init__.py
│       │   ├── align.py
│       │   ├── associate.py
│       │   ├── consistency.py
│       │   ├── metrics.py
│       │   ├── plots.py
│       │   └── report.py
│       ├── sensors/
│       │   ├── __init__.py
│       │   ├── calibration.py
│       │   ├── rig.py
│       │   ├── samples.py
│       │   ├── stream.py
│       │   └── timesync.py
│       └── sim/
│           ├── __init__.py
│           ├── baro.py
│           ├── flow.py
│           ├── gnss.py
│           ├── imu.py
│           ├── mag.py
│           ├── rangefinder.py
│           ├── recorder.py
│           ├── runner.py
│           ├── scenario.py
│           └── trajectory.py
└── tests/
    ├── conftest.py
    ├── baselines/
    │   └── test_dead_reckoning.py
    ├── core/
    │   ├── test_config.py
    │   ├── test_frames.py
    │   ├── test_manifest.py
    │   ├── test_rng.py
    │   ├── test_rotations.py
    │   └── test_time.py
    ├── datasets/
    │   ├── test_asl_reader.py
    │   ├── test_fetch.py
    │   └── test_groundtruth.py
    ├── environment/
    │   └── test_environment.py      # `machine` marker
    ├── eval/
    │   ├── test_align.py
    │   ├── test_associate.py
    │   ├── test_consistency.py
    │   ├── test_cross_check_evo.py  # fails (never skips) under DSS_PHASE_GATE=1
    │   ├── test_metrics.py
    │   └── test_ttr.py
    ├── fixtures/
    │   ├── make_asl_fixture.py      # synthesizes a tiny EuRoC-format folder from the sim; no third-party data committed
    │   └── wmm2025_test_values.txt  # NOAA public domain; URL + SHA-256 in NOTICE
    ├── sensors/
    │   ├── test_calibration.py
    │   ├── test_rig.py
    │   ├── test_samples.py
    │   ├── test_stream.py
    │   └── test_timesync.py
    └── sim/
        ├── test_baro.py
        ├── test_determinism.py
        ├── test_flow.py
        ├── test_gnss.py
        ├── test_imu_allan.py        # `slow` marker
        ├── test_imu_closure.py
        ├── test_mag_wmm.py
        ├── test_rangefinder.py
        └── test_trajectory.py
```
Outside the repo: `$HOME/.cache/dss/cpp/` (source tarballs, the `ws/src/open_vins` workspace, builds and the install prefix; no spaces) and `data/` (gitignored).

## 8. Verification, budgets
- **Phase 0:**
  - `bash scripts/bootstrap_macos.sh`
  - `DSS_PHASE_GATE=1 uv run pytest`
  - `uv run dss fetch tumvi:room1 euroc:MH_01`
  - `uv run dss baseline dr --scenario configs/scenarios/dr_route_2km.yaml --seed 42`, run twice with the hashes compared
  - `uv run dss baseline dr --dataset tumvi:room1` and `uv run dss baseline dr --dataset euroc:MH_01`
  - `bash cpp/spike/build_openvins_macos.sh && bash cpp/spike/probe_large_depth.sh`
- **Phase 1:**
  - `uv run dss fetch tumvi:all euroc:all`
  - `uv run dss vio run --dataset tumvi:room1-6,corridor4 --dataset euroc:all --init gt` → `reports/phase1/vio_gates.md` (auto PASS/FAIL per criterion)
  - `uv run dss geo fetch --aoi flagstaff --epochs 2023`
  - `uv run dss sim sweep --route flagstaff --seeds 5`
  - `uv run dss sim soak --route flagstaff --seeds 5 --legs 7`
- **Phase 2:**
  - `uv run dss geo fetch --aoi flagstaff --epochs 2017,2021`
  - `uv run dss sim run --scenario denied_route --seeds 25`
  - `uv run dss eval alto`
- **Phase 3:**
  - `uv run dss corpus build --set T --hours 10 --jobs 2`, then the same with `--set H`
  - `uv run dss integrity tune --set T`
  - `uv run dss integrity replay --set H`
  - `uv run dss spoof sweep --set H`
  - `uv run dss suite --seeds 5`
  - `uv run dss demo --seed 42 --lockstep`
  - `bash scripts/record_demo.sh runs/demo-42`
  - `uv run pytest -m mavlink`
- **Reports:** every report embeds the run manifest: seed, git SHA, `uv.lock` hash, ovserver binary, OpenVINS commit + patch hashes, brew versions, GL strings.
- **Disk ledger** (`docs/data-budget.md`, enforced by `dss doctor` in every phase):

  | Item | Budget |
  |---|---|
  | Datasets (stream-extracted; EuRoC 10.7 + TUM-VI 11.0) | ≤22 GB |
  | ALTO subset | ≤2 GB |
  | Geo AOI crops | ≤2 GB |
  | Replay corpora T+H (float32+zstd) | ≤2 GB |
  | `runs/` | ≤6 GB |
  | C++ trees | ~3 GB |
  | venv + uv cache (torch) | ~4 GB |
  | Docker images (container fallback, PX4) | ≤6 GB, only if used |
  | PX4 native tree | ≤3 GB, only if built |
  | Free-disk floor | fail <15 GB, warn <25 GB |

  Peak planned ≈50 GB of the 75 GB free.
- **Compute estimate** (≤2 workers, RTF≈1):

  | Phase | Wall-clock |
  |---|---|
  | 0 | <1 h |
  | 1 | ~8 h (datasets, 45-run sweep, 25-seed gate, soak) |
  | 2 | ~7 h (includes the full-scene re-check) |
  | 3 | ~15 h (T + H corpora ≥20 sim-hours, suite, replay sweeps) |

  Batches run overnight under `caffeinate`.

## 9. Defaults I chose (override any at approval)
- **Simulator:** we build our own (render-only MuJoCo + kinematic flatness); no physics engine in v1.
- **World:**
  - Demo world: Flagstaff AZ (2.0 km route). Urban canyon: a procedural high-rise block in the same world (Pittsburgh is stretch).
  - Map epoch: picked in Phase 2 between 2017 and 2021.
- **Rigs and hardware:**
  - The sim rig starts as nadir mono + 120 m lidar (the $399-class story); the Phase 1 sweep may switch to an oblique or stereo VIO camera.
  - Native ovserver first; Docker only as fallback.
  - Autopilot proof: PX4 SIH in Phase 3; ArduPilot is optional.
- **Scope and tooling:**
  - Matcher: Tier A classical first; learned matchers are stretch.
  - Benchmarks: TUM-VI first, then EuRoC; EuRoC numbers kept out of commercial material.
  - Dashboard: Rerun native viewer; the web dashboard is fallback only.
  - Evaluation: the DoD is judged open-loop.

## 10. First Phase 0 task: OpenVINS native build spike + large-depth probe
**Why first:** it is the cheapest gate on Phase 1, and the same binary gives a renderer-free first reading on R11. That risk (VIO at 100 m AGL over 2 km) is the one most able to change the architecture: rig, estimator, or in-filter map updates. **Timebox:** 1 day for the build, +½ day for the probe. **Script:** `cpp/spike/build_openvins_macos.sh`.

**Shared environment:** `P="$HOME/.cache/dss/cpp"`, `PATH=/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin`, `E3="$(brew --prefix eigen@3)/share/eigen3/cmake"`, `GEN="-G Ninja -DCMAKE_MAKE_PROGRAM=/opt/homebrew/bin/ninja"`, and `-j4` for every build.

1. **Deps.**
   - `brew bundle` installs cmake, ninja, eigen@3, boost and uv.
   - Fetch the SHA-256-pinned tarballs:
     - OpenCV 4.14.0: `ee8fb9b30eb60850431b4656447080e3737b56e45719c92b67f245950609f86e`
     - Ceres 2.2.0: `48b2302a7986ece172898477c3bcd6deb8fb5cf19b3327bc49969aad4cede82d`
   - Clone OpenVINS @69488123 into `$P/ws/src/open_vins`.
2. **OpenCV 4.14:**
   ```
   cmake -S "$P/src/opencv-4.14.0" -B "$P/build/opencv" $GEN -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$P" \
     -DBUILD_LIST=core,imgproc,imgcodecs,highgui,features2d,flann,calib3d,video \
     -DWITH_KLEIDICV=OFF -DWITH_FFMPEG=OFF -DWITH_OPENCL=OFF -DWITH_EIGEN=OFF -DWITH_TBB=OFF -DWITH_OPENEXR=OFF -DWITH_AVIF=OFF \
     -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_opencv_apps=OFF -DBUILD_JAVA=OFF -DBUILD_opencv_python3=OFF
   ```
   Then build and `cmake --install`.
3. **Ceres 2.2:**
   ```
   cmake -S "$P/src/ceres-solver-2.2.0" -B "$P/build/ceres" $GEN -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$P" \
     -DEigen3_DIR="$E3" -DMINIGLOG=ON -DGFLAGS=OFF -DSUITESPARSE=OFF -DUSE_CUDA=OFF -DLAPACK=ON \
     -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARKS=OFF
   ```
   Then build and install.
4. **Stock OpenVINS:**
   ```
   cmake -S "$P/ws/src/open_vins/ov_msckf" -B "$P/build/ov_msckf" $GEN -DCMAKE_BUILD_TYPE= \
     -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DENABLE_ROS=OFF -DENABLE_ARUCO_TAGS=OFF -DEigen3_DIR="$E3" \
     -DOpenCV_DIR="$P/lib/cmake/opencv4" -DCeres_DIR="$P/lib/cmake/Ceres" -DCMAKE_PREFIX_PATH="$P" \
     -DCMAKE_INSTALL_PREFIX="$P" -DCMAKE_INSTALL_RPATH="$P/lib"
   ```
   - `CMAKE_BUILD_TYPE` is set empty explicitly, so asserts stay live.
   - `CMAKE_PREFIX_PATH` is load-bearing: OpenVINS probes OpenCV 3 first, which resets `OpenCV_DIR`.
   - Confirm `OPENCV: 4.14.0` in the configure log.
   - Build, install, then:
     ```
     cd "$P/ws"
     CFG="$P/ws/src/open_vins/config/rpng_sim/estimator_config.yaml"
     nm -u "$P/build/ov_msckf/test_sim_repeat" | grep -q ___assert_rtn
     "$P/build/ov_msckf/test_sim_repeat" "$CFG"    # prints the success line
     "$P/build/ov_msckf/run_simulation" "$CFG"     # exit 0
     ```
5. **Helpers** (`cpp/spike/CMakeLists.txt`, C++17, imported `ov_msckf_lib` target):
   - **`sim_rmse`:**
     1. Call `print_and_load` and then `print_and_load_simulation`.
     2. Set `num_opencv_threads=0` and turn multi-threading off.
     3. Construct the Simulator, then the VioManager.
     4. Initialize with GT, and hold each camera frame back by one, as `run_simulation` does.
     5. Evaluate `get_state(state->_timestamp + calib_camimu_dt)` against `state->_imu->pos()`, with NEES from `StateHelper::get_marginal_covariance`.
     6. Log per-epoch MSCKF/SLAM update counts.
   - **`klt_smoke`:** TrackKLT on CV_8UC1 frames with two independently shifted texture regions (parallax, so the F-matrix RANSAC isn't degenerate), zero masks and sensor id 0; plus `cv::imread` of a PNG it wrote.
6. **Probe** (`make_traj.py` + `probe_large_depth.sh`).
   - Per seed, write the three YAMLs into one directory with an absolute `sim_traj_path`.
   - Config keys: `max_cameras: 1`, `use_stereo: false`, `fi_max_dist: 200` (the default 60 m would reject every feature), intrinsics/IMU-intrinsics/g-sensitivity calibration off.
   - Camera and IMU: nadir `T_imu_cam` (image top toward travel) and BMI160 noise.
   - Run two variants, `sim_{min,max}_feature_gen_dist` 98/102 and 80/120, 5 seeds each via `sim_seed_measurements`.
7. **Record** the patches needed (or the failure class: recipe vs platform) and the probe results in ADR-0001.
