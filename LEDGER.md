# LEDGER — autonomous build loop

Loop started 2026-09-28 21:56 EDT. Hard stop 2026-09-29 07:56 EDT (10 h).
One entry per phase: verdict first, then numbers, files, cuts, and risks.

## Phase 0: repo, sensor models, eval harness, dead-reckoning baseline (2026-09-28 22:18)
**Verdict: DONE, with 2 logged cuts** (no EuRoC data; no evo cross-check). Every other Phase 0 exit criterion was met and verified by a test or artifact.

| Exit | Result | Evidence |
|---|---|---|
| 1 Build + probe (ADR-0001) | **PASS**. Stock OpenVINS @69488123 builds natively with an empty patch series. test_sim_repeat succeeds with asserts live, run_simulation exits 0, sim_rmse RMSE is 0.039 m with finite NEES, klt_smoke keeps ≥178 tracks. Probe (10 runs): 0 divergences, ATE/len 0.51% flat / 0.49% relief, **position NEES ≈9 (overconfident)** | `docs/adr/0001-*.md` |
| 2 Environment | **PASS**. `dss doctor` reports CPython 3.13.15, HTTPS 200, 70 GB free, ledger in budget. The `--full` advisories all pass, including a MuJoCo CGL frame, the IFLS constructor and cv2 5.0.0 resize | `uv run dss doctor --full` |
| 3 Tests | **PASS**. `DSS_PHASE_GATE=1 uv run pytest`: 38 passed, 0 skipped. The CI yml has a spaced checkout path, but CI has not been run (no push) | `tests/` |
| 4 Sensor models | **PASS**. Noise-free strapdown closes to 1.8 mm over 60 s. Flatness rates match 10 kHz finite differences to <1e-6. Allan fit (3 h, both presets, all axes) N̂ within ±5% and K̂ within ±20%. ISA round trip <1e-6 m. GNSS (20×1 h) CEP within ±5%, σ_v within ±10%, τ_pos within ±20%. Jam: no fix within 1 epoch, reacquisition after the hold. Spoof profiles exact. Flow <5% RMS. Lidar σ formula holds, with out-of-range marked invalid | `tests/sim/*` |
| 5 Sensor layer | **PASS**. Offset recovered <0.1 ms with 0.5 ms jitter. Merged stream is ordered and deterministic (hypothesis). SE3 chain round trip <1e-12. EuRoC sensor.yaml chain = OpenVINS `T_imu_cam` to 1e-9 | `tests/sensors/*` |
| 6 Eval harness | **PASS, one cut**. posyaw <1e-9 m; RPE handles segments longer than the path; end-point drift and TTR self-tests pass; DR growth = ½·b_a·t² within 1%. **Cut:** SE3 ATE vs `evo_ape` not run (evo not installed; needs no credentials, deferred) | `tests/eval/*` |
| 7 Covariance plumbing | **PASS**. DR position ANEES, N=50, 60 s = 2.42, inside [2.36, 3.72], 92% of epochs inside | `tests/baselines/test_dead_reckoning.py` |
| 8 Determinism tier A | **PASS**. Identical SHA-256 per stream; adding a stream changes no existing hash | `tests/sim/test_determinism.py` |
| 9 Artifacts + data | **PARTIAL**. DR reports for sim and TUM-VI room1 are in `reports/phase0/`. The TUM-VI SHA-256 is recorded TOFU in `configs/datasets/manifest.json`. **Cut:** EuRoC MH_01 (ETH returns 429 and times out; research-collection also 429) | `reports/phase0/dr_metrics.json` |

Numbers:
- DR, 2 km sim route, seed 42, BMI160, GT init: error 27 m at 60 s, final 4.2 km.
- With baro + mag: error 17.7 m at 60 s, final 0.9 km, vertical RMSE 0.84 m.
- TUM-VI room1 DR: 2.5 m at 10 s, 108 m at 60 s.

Cuts / deviations:
- EuRoC unavailable (HTTP 429). TUM-VI stands in for all dataset gates.
- Lidar σ_rel is 0.005, not the 0.05 in PLAN. At 5% of range, 100 m gives σ = 5 m, which is not an SF11/C.
- The WMM check is a sanity band at Flagstaff, not the NOAA test-value fixture (not fetched).
- `opencv-python-headless` resolved to 5.0.0; its resize smoke passes.

Risks carried forward: VIO position NEES ≈9 at 100 m depth (R12); EuRoC gates in Phase 1 cannot run.

## Phase 1: VIO core integrated, drift target (2026-09-28 23:35)
**Verdict: PARTIAL.**
- Pass: dataset gates on TUM-VI rooms 1–5, the renderer, and OpenVINS in the sim loop.
- Cut: EuRoC (exits 3/4), the rig sweep, the 25-seed sim-VIO gate (reduced), and the soak.
- Key finding: re-anchoring a fresh mono MSCKF at 100 m AGL from REF's post-outage state diverges.

| Exit | Result | Evidence |
|---|---|---|
| 1 Protocol | **PASS**: 10k frames at verbosity ALL, 0 framing errors; malformed frames rejected; kill → `VioFailure`; health schema | `tests/vio/test_protocol.py` |
| 2 Determinism | **PASS**: two room1 runs byte-identical (sha256 fb7af363…) | `reports/phase1/vio_gates.md` |
| 3/4 EuRoC | **CUT**: ETH returns HTTP 429 or times out; research-collection returns 429 | — |
| 5 Drift ≤1% | **PASS (partial scope)**: max 0.074% over TUM-VI rooms 1–5, mono and stereo. room6/corridor4 not fetched | `vio_gates.json` |
| 6 TUM-VI rooms 1–5 | **PASS**: mono average 0.064 m (gate 0.10), stereo 0.077 m (gate 0.12) | `vio_gates.md` |
| 7 Timing | **PASS (proxy)**: on TUM-VI in place of EuRoC V1_01: mono p99 ≤12.4 ms, RTF 4.6; RSS 32–51 MB. RSS-slope soak not run | `vio_gates.md` |
| 8 Renderer | **PASS**: projection residual <0.2 px; depth matches RGB to 5 cm; Tier B 100 frames give identical SHA-256; 254 fps raw (≥20 gate). In the pipeline, with photometrics, about 57 fps | `tests/sim/test_render.py` |
| 9 Sim VIO | **PARTIAL**: see below. The 25-seed ANEES gate and the time-offset recovery test were not run | `eval/results/*` |
| 10 Budget | **PASS**: data 9.9 GB (TUM-VI 5 rooms + geo 0.14 GB) | `dss doctor` |

What was built:
- **ovserver ↔ `dss.vio.ovclient`.** Protocol v1; JPL↔Hamilton checked by Monte Carlo.
- **Flagstaff AOI.**
  - NAIP render epoch 2023-06 at 0.3 m; map epochs 2021-11 and 2017-06 at 1 m.
  - 3DEP 1/3″ DEM on NAD83 / UTM 12N (local ENU, grid north).
  - Imagery via Planetary Computer.
- **MuJoCo terrain renderer.** Textured DEM mesh, exact K, CGL, and a photometric chain: vignetting, haze, motion blur, shot/read noise, and auto-exposure with lag.
- **Route.** 2.0 km, terrain-following at 100 m AGL, 9 m/s. It runs west from downtown and makes a U-turn inside the Mars Hill forest, so the tail flies back over town (F15's "textured tail").

Sim VIO (OpenVINS on rendered frames, nadir mono 640×480, f=400, BMI160):
- **GT init at 12 s.** Tracks the whole climb and cruise, with no divergence before the forest.
- **Velocity error** 0.05–0.13 m/s at 100 m AGL, i.e. ≈0.6–1.4% of distance.

**Finding (R11/R12): re-anchoring after a vision outage.**
- A fresh `VioManager` seeded from REF's post-outage state and marginals, as PLAN §3 prescribes, converges to a wrong velocity (0.4–0.8 m/s; 5–18 m/s after darkness).
- The same re-anchor from exact truth with a tight prior holds (≈0.05 m/s). With truth and REF's loose prior it drifts 0.35–0.8 m/s.
- Cause: at 100 m depth, monocular MSCKF velocity observability is weak, so the prior dominates.

Three distinct fix attempts, all failed (2 of 3 seeds diverged):
1. Consistency gating + re-anchor.
2. Clamped re-anchor covariance + a refresh re-anchor after 3 map fixes.
3. An 18-state ESKF with a VIO velocity-bias state, reset correlated with REF's velocity error (kept; it is principled).

**CUT (failure rule):**
- After a vision outage, ovserver is re-anchored from PLAN 1.5's **GT-derived anchor**, x = GT ⊞ δ with δ ~ N(0, P15_proxy): σ_h 2.4 m, σ_z 1 m, σ_v 0.2 m/s, 0.5/0.5/1°.
- This is the only place ground truth touches the estimator. REF and ALL never see GT.
- `fusion.reanchor_source: ref` restores the PLAN behaviour for the next attempt.
- **Next step for this cut:** stereo or oblique rig (the Phase 1.5 sweep), or lidar-scaled feature depth priors.

Files:
- `src/dss/vio/*`, `cpp/ovserver/*` (mask support), `src/dss/geo/{aoi,naip}.py`
- `src/dss/sim/{render/*,world/tiles.py}`, `src/dss/sim/trajectory.py` (terrain following, U-turn, yaw wobble)
- `reports/phase1/*`, `tests/{vio,sim/test_render.py}`

Risks: sim VIO position covariance is optimistic (spike NEES ≈9), so fusion uses a VIO-bias state instead of trusting ovserver's covariance.

## Phase 2: orthoimagery aiding + fusion backend (2026-09-29 00:45)
**Verdict: PARTIAL, core gates PASS.**
- Pass: fix quality, bounded error, VIO/aided ratio and honest covariance, all on 5 seeds.
- Cut: GTSAM backend, full-scene buildings/trees, ALTO real-data check, the 25-seed N, and the <1 m co-registration gate.

| Exit | Result | Evidence |
|---|---|---|
| 1 Geo | **PARTIAL**. STAC returns 7 NAIP epochs; map epoch = **2017-06-27** (same season as the 2023-06 render; 6-year gap). Raw epoch-to-epoch misregistration is 3–16 m, locally systematic: 2021 is off by (−7, +11) m over the downtown/U-turn legs. Phase-correlation residual 3.6 m (2017) / 5.1 m (2021), **above the <1 m gate**. The fix is a dense patch-NCC displacement field (40 m grid, robust median), used to warp 2017 onto the render epoch (PLAN 2.1 "co-register the epochs"). R13 note: map geometry is now tied to the render epoch; content is still 6 years apart | `reports/phase2/coregistration.json`, `src/dss/geo/coreg.py` |
| 2 Tier-A speed | **PASS**: 96² patch over a ≤400² window at 1 m takes <20 ms (tested). A 512² template over a 712² window is <0.5 s, i.e. the literal 512² gate is not met; the operational patch is 96² | `tests/aiding/test_matcher.py` |
| 3 Fix quality | **PASS** (textured segments, all suite scenarios, cross-epoch): 4,124 attempts, 3,744 accepted; **median 1.02 m, p95 3.15 m, success 89.9%, false-accept 0%**. Fix NIS mean 0.23: covariance conservative, by design, for spatially correlated registration error. ALTO not run (CUT). Time-to-first-fix after σ_REF>50 m: post-forest first accepted fix arrives 13–16 s after light returns | `reports/phase2/gates.json` |
| 4 Bounded error | **PASS** (denied_route, no GPS for the whole flight, 5 seeds, real OpenVINS): per-seed textured RMSE 1.07–1.54 m (≤5), pooled p99 4.27 m (≤10), median 0.98 m (≤3), **0 HMI epochs**, RMSE_VIO/RMSE_aided median **5.8** (≥3; every seed ≥3.3). REF RMSE 1.2–2.1 m (≤3.4). Vertical p95 0.78 m; touchdown vertical 0.08–1.5 m | same |
| 5 Honest covariance | **PASS at N=5** (PLAN: N=25): REF 3-DoF position ANEES 4.31, inside [1.25, 5.50] | same |
| 6 Robustness | **CUT/REPLACED**. GTSAM IFLS not used: an **18-state ES-EKF** (15 INS + 3 VIO velocity-bias Gauss-Markov) is the backend, behind the same `NavFilter` interface; outage mode is IMU + baro + lidar + mag coasting. No IndeterminateSystem risk. p95 update cost is far below 20 ms (fusion totals ≈7 s CPU per 278 s run) | `src/dss/fusion/*` |
| 7 Full scene | **CUT**: no buildings/trees meshes; the "forest" is photometric canopy darkness over NAIP forest texture | — |
| 8 Budget | **PASS** | — |

Design decisions made during this phase (all logged in code docstrings):
- Map fixes: χ²₂ ≤ 9.21 gate. After a >15 s gap they need 3 pairwise-consistent fixes with peak ratio ≥1.8. Consensus relocalization runs when ≥4 strong rejected fixes agree.
- Fix covariance = peak curvature + 1.5 m floor + 1 m registration + tilt term, ×2 for spatial correlation.
- Orthorectification is per-pixel onto the DEM.
- GPS is de-weighted ×20 at 1 Hz for its τ=60 s coloured error.
- VIO: horizontal velocity only, with a bias state reset on re-anchor.
- The 2021 epoch was tried first and abandoned for its ~10–16 m systematic misregistration; 2017 co-registered is used.

Forest leg (reported separately, as the PLAN requires): the 45 s dark segment coasts on IMU + baro + lidar + mag.
- Horizontal error reaches 25–66 m (median 2–12 m over the segment).
- VIO FAILED is announced 2.0 s after the canopy starts (ramp 2 s).
- After light returns, REF is within 0.7–2.8 m of truth 10 s after the first accepted fix.

## Phase 3: integrity monitor, GPS attack interface, output, dashboard, demo (2026-09-29 02:05)
**Verdict: DoD MET; Phase 3 PARTIAL.**
- **The demo** runs end to end. It is deterministic (`metrics.json` byte-identical on re-run) and `demo.mp4` is recorded. `eval/results/` holds trajectory/error/timeline plots + metrics for all 9 scenarios × 5 seeds, all on real OpenVINS.
- **Fail:** the 0-false-alarm gate (1.18/h on held-out set H), aggressive HMI, and part of the committed spoof envelope.
- **Cut:** PX4 SIH, the webcam run itself (TCC), full-length ≥10 h rendered corpora, and per-source fault injection.

Demo, `uv run dss demo --seed 42 --lockstep` → `runs/demo-42/{run.rrd,metrics.json,vpe.mavlink,*.png}`; `bash scripts/record_demo.sh runs/demo-42` → `demo.mp4`.

**Timeline** (F15): jam ramp 55–60 s, outage, spoofer capture at 120 s, 1 m/s drag-off at 125 s, Mars Hill forest canopy 150–195 s, textured tail, landing 248–273 s.

**Numbers:**
- **Accuracy:** published horizontal RMSE 5.68 m. Final error after the post-forest recovery (t=248 s) is 2.12 m against HPL 7.1 m; touchdown 2.13 m.
- **Integrity:** **0 HMI epochs**. Max 30 Hz jump 0.018 m; bleed ≤ 0.50 m/s; 0 `reset_counter` increments.
- **Output:** VPE at 30.00 Hz, 8,347 frames, all starting 0xFD with 21 finite covariance values.
- **Spoof:** flagged 6.0 s after drag-off onset, before PROBATION ended, and 4.8 s before the offset was even detectable (MDB). The published position stayed within HPL until the flag.
- **Vision:** VIO FAILED announced 2.0 s after the canopy starts (2 s darkness ramp). REF was within 1.4 m 10 s after the first post-forest fix.

**Determinism:** two runs give identical `metrics.json` (sha256 db5bf2fa…).

**The video** is 1920×1080, 30 fps, 278 s at 1×. It is rendered offline from the run logs, in the same 3-column layout as the Rerun blueprint, because screen capture needs a TCC grant.

| Exit | Result |
|---|---|
| 1 Definition of done | **PASS** (above) |
| 2 Spoof detection | **PARTIAL**. Offline sweep: frozen gate vs REF logs from 35 rendered runs, fresh GNSS seeds, onset at 100 s while TRUSTED; 140 segments per point. p50 latency 0–2 s for the 20 m step and for ≥1 m/s drags; 17 s at 0.5 m/s; 27 s at 0.3 m/s. p95 has a ≈15 s tail from segments whose onset fell inside a REF-degraded hold. **Envelope:** 1 m/s ≤20 s **PASS** (p95 15 s); 20 m step ≤1.2 s **FAIL** (p95 15 s); 2 m/s ≤8 s **FAIL** (p95 15 s); 0.5 m/s ≤35 s **FAIL** (p95 25.6 s but 2 of 140 censored). **Blind spot (documented):** with REF degraded (twilight, VIO-only, vision outages) the gate holds and cannot flag; 40/40 censored. SID stays small for fast drags; for slow drags it is 13–57 m at flag time. In-suite: dragoff_spoof flagged a median 2.8 s after onset; demo 4–6 s |
| 3 False alarms | **FAIL**. 32 alarms over 27.2 h of held-out set H: 45 rendered REF logs × 8 GNSS seeds 2000+; thresholds tuned on seeds 1000+ (set T) and frozen in `configs/integrity/thresholds.yaml`, sha256 646f0fe9…. Rate UB95 1.58/h. **Root cause:** REF overconfidence in two flight phases, climb/accel (~33 s: 3.5 m error at σ 1.2 m) and landing deceleration (~247 s: 1 m/s velocity error at σ 0.17 m/s), not GNSS noise. Deviation: set H reuses the same REF logs as T with disjoint GNSS seeds (PLAN wants disjoint ≥10 h rendered corpora) |
| 4 Failover continuity | **PASS**: 0 `reset_counter` increments; max jump 0.018 m (≤0.05); bleed ≤0.5 m/s; `px4_ev_gate` replay shows **0 rejections** in every scenario |
| 5 Honest output | **PARTIAL**. HMI = **0 in 8/9 scenarios**; aggressive 50 epochs, in descent after the yaw-oscillation leg. Published NEES time-average (N=5): 3.5–4.6 on nominal / forest / demo, inside [1.25, 5.50]; 14–26 on aggressive / VIO-only / twilight (overconfident). REF independence holds: demo ≡ forest and denied_route ≡ gps_cutover ≡ urban ≡ dragoff give byte-identical REF trajectories, and GPS never reaches REF (tested import contract) |
| 6 Degradation | **PARTIAL**. Forest: VIO FAILED 2.0 s after onset (the ≤1 s gate is missed because the darkness ramps over 2 s); announced via event, health bar and inflated VPE covariance. REF back within 1.4 m (≤10 m) 10 s after the first fix. The blur and illumination-shock detectors are implemented (`integrity/vision.py`) but their ≥95%-detection gates were not run |
| 7 Fault injection | **CUT** (hooks exist in the sim runner: baro step, mag anomaly, lidar dropout/bias, flow scale; not swept) |
| 8 Scenario suite | 9 scenarios × 5 seeds, all on real OpenVINS; `eval/results/scoreboard.md`. **urban_canyon:** NLOS GPS REJECTED 0.2 s after the bias appears, 0 HMI. **aggressive** (≤≈115°/s yaw, 12 m/s): no divergence, RMSE median 4.5 m, HMI 50 (FAIL). **twilight:** map disabled and announced in 5/5; RMSE 18 m. **forest:** as demo. **gps_cutover:** TTR 0.005 s (published was already REF-consistent at loss). **dragoff_spoof:** as exit 2 |
| 9 Output interface | **PASS**: 0xFD frames, 21 finite covariance values, 30.0 Hz, pymavlink round-trip, import contract. **CUT:** PX4 SIH (no time; the `px4_ev_gate` emulator stands in); latency p99 not measured separately (lockstep) |
| 10 Dashboard / live / CPU | **PARTIAL**. Rerun `.rrd` with the fixed 3-column blueprint (0.38 API). Live mode (`dss.live.webcam`, KLT + up-to-scale VO, "scale: arbitrary") is tested on a rendered loop: Sim(3) error <10%. FaceTime run not attempted (Camera TCC prompt would block an unattended session). Stack RTF ≈1.2–1.9 including rendering on M1 Pro; E-core runs not done |
| 11 Budget | **PASS**: data 7.9 GB, C++ 1.5 GB, eval/results 115 MB (logs gitignored), 60+ GB free |
