# Autopilot integration (PX4 / ArduPilot)

`dss.output.mavlink_vpe` publishes **MAVLink 2 `VISION_POSITION_ESTIMATE` (#102) at 30 Hz** in the local NED frame (origin = AOI origin; send `SET_GPS_GLOBAL_ORIGIN` once so the autopilot can geo-reference it). One message type only; never send `ODOMETRY` on the same link.

- `usec`: sim/sensor time of the estimate (µs). Pair with the TIMESYNC responder for real hardware.
- `x, y, z`: published position (NED). This is REF ⊕ clamp(ALL − REF), bled at ≤ 0.5 m/s, so absolute corrections never appear as jumps.
- `roll, pitch, yaw`: FRD body in NED, with yaw bled at ≤ 2°/s.
- `covariance[21]`: upper triangle of the 6×6 (x, y, z, roll, pitch, yaw) matrix. It is a **conservative diagonal**: both horizontal axes carry λmax of the horizontal covariance, including the un-bled offset. All 21 values are finite (MAVLink 2; a v1 frame would be zero-filled).
- `reset_counter`: incremented only on a true re-initialization. There were 0 increments in every suite run.

**No code changes, but parameter changes are required** (PLAN F16):
- **PX4 (EKF2).**
  - `EKF2_EV_CTRL` = horizontal + vertical position (bit 0 + bit 1). Leave yaw off unless mag is unavailable.
  - Set `EKF2_HGT_REF` as desired: baro or vision.
  - `EKF2_EV_DELAY` ≈ measured VPE latency.
  - Keep `EKF2_EVP_GATE` = 5, which the `dss.eval.px4_ev_gate` emulator reproduces.
  - `EKF2_EV_NOISE_MD` = 0, so EKF2 uses our variances.
  - PX4 uses only the covariance **diagonal**.
- **ArduPilot.**
  - `VISO_TYPE` = 1 (MAVLink).
  - `EK3_SRC1_POSXY`/`VELXY`/`POSZ` = 6 (ExternalNav).
  - ArduPilot collapses the covariance to √trace.

Verification in this repo:
- `tests/output/test_mavlink.py` round-trips frames through pymavlink: 0xFD magic, NED mapping, yaw convention, 21 finite covariance values.
- `reports/phase3/gates.json → exit4_ev_gate_replay` replays each scenario's VPE stream through the EKF2-style 5-SD gate.
- PX4 SIH acceptance was not run: it is CUT in the LEDGER, and the gate emulator plus golden tests stand in.
