# Metric definitions (frozen, Phase 0)

Implemented in `src/dss/eval/`. Criteria in PLAN.md cite these names.

- **ATE**: RMSE of position error after alignment. Datasets: posyaw alignment (`align_posyaw`). Sim: no alignment (shared world frame). Association within 0.01 s (`associate`).
- **RPE**: translation error of sub-trajectories of fixed path length (8/16/24/32/40 m datasets, 100–800 m sim). Segments longer than the path yield `count: 0`.
- **Drift %**: 100 · ATE / path length. End-point drift: SE3-aligned on the first segment, measured on the last.
- **Relocalization**: success = accepted fix with error ≤ 5 m / attempts; precision = successes / accepted; false-accept = accepted with error > 20 m; time to first fix.
- **TTR** (time to recover after GPS loss) = t* − t_loss; t* is the first time from which every sample in [t*, t*+10 s] has error ≤ 5 m and truth inside the 99% ellipse; 0 if that already holds at t_loss; censored (fail) if never.
- **ANEES(t)**: mean NEES over N independent seeds at 1 Hz. Pass = inside the two-sided 95% χ²(dN)/N band in ≥ 90% of epochs and the time-average inside. One-sided variant (raw VIO): ≤ upper bound in ≥ 90% of epochs, time-average ≥ 1.0.
- **HPL(t)** = 5.68 · √λmax(horizontal P_pub). **HMI epoch**: |p_pub − p_true|_h > HPL(t) with no active alert.
- **SID** (spoof-induced deviation): max_t |p_pub,spoofed(t) − p_pub,nominal(t)| on paired same-seed runs.
- **Spoof timing**: t_onset = first epoch the spoofed position departs from truth; t_detectable = first time the true offset > MDB = 5.94 · σ_innov; alarm = any entry into SUSPECT or REJECTED.
- **Divergence**: NaN/Inf, 0 MSCKF updates for > 5 s, or error > 1 m (datasets) / > 5% of distance (sim).
- **RTF**: stack-only (VIO + fusion + integrity + output) fed from recorded sensor logs.
