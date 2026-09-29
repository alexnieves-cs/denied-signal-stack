#!/usr/bin/env bash
# Task 0.1 large-depth probe (PLAN.md §10 step 6): OpenVINS MSCKF on its own simulator with a nadir
# mono camera at ~100 m depth over a 2 km S-turn. Informational; results go into ADR-0001.
#
# Variants: flat (feature depth 98-102 m) and relief (80-120 m); seeds via sim_seed_measurements.
# Usage: probe_large_depth.sh [n_seeds=5]      (GPL-3.0 context: drives OpenVINS)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
P="$HOME/.cache/dss/cpp"
export PATH=/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin
NSEEDS="${1:-5}"
OUT="$P/probe"
TEMPLATE="$P/ws/src/open_vins/config/rpng_sim/estimator_config.yaml"
BIN="$P/build/spike/sim_rmse"
mkdir -p "$OUT"
python3 "$HERE/make_traj.py" "$OUT/traj_2km_100m.txt"

write_cfg() { # dir, min_dist, max_dist, seed
  local d="$1" mn="$2" mx="$3" seed="$4"
  mkdir -p "$d"
  sed -e 's|^verbosity:.*|verbosity: "WARNING"|' \
      -e 's|^use_stereo:.*|use_stereo: false|' \
      -e 's|^max_cameras:.*|max_cameras: 1|' \
      -e 's|^calib_cam_intrinsics:.*|calib_cam_intrinsics: false|' \
      -e 's|^calib_imu_intrinsics:.*|calib_imu_intrinsics: false|' \
      -e 's|^calib_imu_g_sensitivity:.*|calib_imu_g_sensitivity: false|' \
      -e 's|^num_pts:.*|num_pts: 200|' \
      -e 's|^num_opencv_threads:.*|num_opencv_threads: 0|' \
      -e "s|^sim_seed_measurements:.*|sim_seed_measurements: $seed|" \
      -e "s|^sim_traj_path:.*|sim_traj_path: \"$OUT/traj_2km_100m.txt\"|" \
      -e 's|^sim_freq_cam:.*|sim_freq_cam: 20|' \
      -e 's|^sim_freq_imu:.*|sim_freq_imu: 200|' \
      -e "s|^sim_min_feature_gen_dist:.*|sim_min_feature_gen_dist: $mn|" \
      -e "s|^sim_max_feature_gen_dist:.*|sim_max_feature_gen_dist: $mx|" \
      "$TEMPLATE" > "$d/estimator_config.yaml"
  # the default fi_max_dist (60 m) would reject every feature at 100 m depth
  printf '\nfi_max_dist: 200\nfi_min_dist: 0.25\nfi_max_baseline: 1000\n' >> "$d/estimator_config.yaml"
  # nadir camera, image top toward travel: cam z = -body z, cam y = -body x, cam x = -body y
  cat > "$d/kalibr_imucam_chain.yaml" <<'YAML'
%YAML:1.0
cam0:
  T_imu_cam:
    - [0.0, -1.0, 0.0, 0.0]
    - [-1.0, 0.0, 0.0, 0.0]
    - [0.0, 0.0, -1.0, -0.05]
    - [0.0, 0.0, 0.0, 1.0]
  cam_overlaps: []
  camera_model: pinhole
  distortion_coeffs: [0.0, 0.0, 0.0, 0.0]
  distortion_model: radtan
  intrinsics: [400.0, 400.0, 320.0, 240.0]
  resolution: [640, 480]
  rostopic: /cam0/image_raw
YAML
  # BMI160-class consumer IMU (same preset as configs/sensors/imu_bmi160.yaml)
  cat > "$d/kalibr_imu_chain.yaml" <<'YAML'
%YAML:1.0
imu0:
  T_i_b:
    - [1.0, 0.0, 0.0, 0.0]
    - [0.0, 1.0, 0.0, 0.0]
    - [0.0, 0.0, 1.0, 0.0]
    - [0.0, 0.0, 0.0, 1.0]
  accelerometer_noise_density: 2.0e-3
  accelerometer_random_walk: 5.0e-4
  gyroscope_noise_density: 2.4e-4
  gyroscope_random_walk: 2.0e-5
  rostopic: /imu0
  time_offset: 0.0
  update_rate: 200.0
  model: "kalibr"
  Tw: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
  R_IMUtoGYRO: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
  Ta: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
  R_IMUtoACC: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
  Tg: [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
YAML
}

SUMMARY="$OUT/summary.tsv"
printf 'variant\tseed\tresult\n' > "$SUMMARY"
for variant in flat:98:102 relief:80:120; do
  IFS=: read -r name mn mx <<<"$variant"
  for ((s = 1; s <= NSEEDS; s++)); do
    d="$OUT/$name/seed$s"
    write_cfg "$d" "$mn" "$mx" "$s"
    printf '[%s] %s seed %d ... ' "$(date +%H:%M:%S)" "$name" "$s"
    if "$BIN" "$d/estimator_config.yaml" "$d/per_epoch.csv" >"$d/log.txt" 2>&1; then :; else echo "(nonzero exit)"; fi
    line="$(grep -o 'RESULT.*' "$d/log.txt" || echo 'RESULT crashed=1')"
    echo "$line"
    printf '%s\t%d\t%s\n' "$name" "$s" "$line" >> "$SUMMARY"
  done
done
echo "summary: $SUMMARY"
