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
