# LEDGER — autonomous build loop

Loop started 2026-09-28 21:56 EDT. Hard stop 2026-09-29 07:56 EDT (10 h).
One entry per phase: verdict first, then numbers, files, cuts, and risks.

## Phase 0: repo, sensor models, eval harness, dead-reckoning baseline (2026-09-28 22:55)
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
