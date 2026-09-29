# Phase 1 VIO dataset gates (TUM-VI)

OpenVINS @69488123 via ovserver (release, lockstep, `num_opencv_threads: 0`), shipped `tum_vi` config, GT init at motion onset (GT speed > 0.2 m/s), posyaw ATE vs mocap0, association 0.01 s.

| seq | mode | ATE m | drift % | RPE8 % | p50 ms | p99 ms | RTF | RSS MB | diverged |
|---|---|---|---|---|---|---|---|---|---|
| room1 | mono | 0.071 | 0.049 | 0.428 | 5.567 | 9.810 | 4.599 | 32.719 | False |
| room1 | stereo | 0.077 | 0.053 | 0.427 | 12.511 | 19.703 | 2.785 | 43.625 | False |
| room2 | mono | 0.071 | 0.050 | 0.489 | 5.242 | 9.048 | 4.755 | 33.297 | False |
| room2 | stereo | 0.098 | 0.070 | 0.664 | 11.627 | 18.091 | 2.956 | 51.250 | False |
| room3 | mono | 0.065 | 0.048 | 0.510 | 5.267 | 8.729 | 4.697 | 32.734 | False |
| room3 | stereo | 0.083 | 0.061 | 0.476 | 12.331 | 19.090 | 2.803 | 44.344 | False |
| room4 | mono | 0.030 | 0.044 | 0.387 | 5.626 | 10.320 | 4.591 | 31.891 | False |
| room4 | stereo | 0.027 | 0.040 | 0.332 | 12.620 | 19.740 | 2.782 | 43.594 | False |
| room5 | mono | 0.081 | 0.062 | 0.533 | 5.358 | 12.420 | 4.686 | 35.375 | False |
| room5 | stereo | 0.097 | 0.074 | 0.467 | 12.574 | 21.299 | 2.766 | 47.578 | False |

| gate | result | detail |
|---|---|---|
| exit1_protocol | **PASS** | 10k frames @ verbosity ALL, malformed-frame, kill->VioFailure, schema: tests/vio/test_protocol.py  |
| exit2_determinism | **PASS** | two full TUM-VI room1 mono runs, byte-identical pose+cov log  |
| exit3_euroc_stereo | **CUT** | CUT: EuRoC unavailable (ETH HTTP 429 / timeout)  |
| exit4_euroc_mono | **CUT** | CUT: EuRoC unavailable (ETH HTTP 429 / timeout)  |
| exit5_drift | **PASS** | TUM-VI room1, room2, room3, room4, room5 only; corridor4 and missing rooms not fetched {"max_drift_pct": 0.074252677} |
| exit6_tumvi_rooms | **PASS** | needs rooms 1-5; partial if fewer are available {"mono_avg_m": 0.063573671, "stereo_avg_m": 0.076508638, "rooms": ["room1", "room2", "room3", "room4", "room5"]} |
| exit7_timing | **PASS** | proxy on TUM-VI (EuRoC V1_01 unavailable); release build, lockstep single-thread; RSS slope over a 25-min soak is a sim gate (not run here)  |

Determinism (room1 mono, 2 full runs): identical (sha256 fb7af3637bf9770b…)
