# ovserver configs

Copies of OpenVINS @69488123 `config/tum_vi` with determinism overrides only:
- `verbosity: WARNING` (protocol tests override to ALL);
- `num_opencv_threads: 0` (serial OpenCV; ovserver already forces `use_multi_threading_*` off);
- `tum_vi_mono`: `max_cameras: 1`, `use_stereo: false`.
Everything else (noise, calibration flags, `num_pts`, masks) is as shipped.
- `sim_nadir_mono`: the Task 0.1 large-depth probe config (nadir mono 640×480, f=400 px, `fi_max_dist: 200`,
  `fi_max_baseline: 1000`, intrinsics/IMU-intrinsics/g-sensitivity calibration off, `num_opencv_threads: 0`,
  `use_mask: false`, BMI160-class noise), with the `sim_*` keys removed and `init_dyn_use: false` (GT init).
  `T_imu_cam` equals `dss.sensors.rig.R_BODY_NADIR_CAM` with the camera 5 cm below the IMU.
