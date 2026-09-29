#!/usr/bin/env bash
# Task 0.1 — native OpenVINS build spike for macOS arm64 (see PLAN.md §10, ADR-0001).
#
# Builds a hermetic toolchain into $HOME/.cache/dss/cpp (no spaces in the path):
#   minimal OpenCV 4.14.0 + Ceres 2.2.0 (miniglog) against Homebrew eigen@3 3.4.1 and boost 1.92,
# then stock OpenVINS @69488123 (ROS-free) and the spike helpers (sim_rmse, klt_smoke).
#
# Usage: build_openvins_macos.sh [all|deps|openvins|tests|helpers]   (default: all)
# This file drives GPL-3.0 code (OpenVINS); see cpp/spike/LICENSE.
set -euo pipefail

STAGE="${1:-all}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
P="$HOME/.cache/dss/cpp"
export PATH=/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin
unset CMAKE_BUILD_TYPE   # CMake >= 3.22 would read it from the environment
JOBS="${DSS_JOBS:-4}"

OPENCV_VER=4.14.0
OPENCV_SHA=ee8fb9b30eb60850431b4656447080e3737b56e45719c92b67f245950609f86e
CERES_VER=2.2.0
CERES_SHA=48b2302a7986ece172898477c3bcd6deb8fb5cf19b3327bc49969aad4cede82d
OV_SHA=69488123ed9362dd44b6f28e7f4680abbff1442b

E3="$(brew --prefix eigen@3)/share/eigen3/cmake"
GEN=(-G Ninja -DCMAKE_MAKE_PROGRAM=/opt/homebrew/bin/ninja)
LOGS="$P/logs"
mkdir -p "$P/src" "$P/build" "$LOGS" "$P/ws/src"

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
run_logged() { # name, command...
  local name="$1"; shift
  log "$name → $LOGS/$name.txt"
  if ! "$@" >"$LOGS/$name.txt" 2>&1; then
    log "FAILED: $name (tail follows)"; tail -40 "$LOGS/$name.txt"; exit 1
  fi
}

fetch_and_verify() { # url, file, sha256
  local url="$1" file="$2" sha="$3"
  [[ -f "$P/src/$file" ]] || curl -fsSL -o "$P/src/$file" "$url"
  echo "$sha  $P/src/$file" | shasum -a 256 -c - >/dev/null || { log "SHA-256 mismatch: $file"; exit 1; }
}

stage_deps() {
  log "toolchain: $(cmake --version | head -1), ninja $(ninja --version), $(brew list --versions eigen@3 boost | tr '\n' ' ')"
  fetch_and_verify "https://github.com/opencv/opencv/archive/refs/tags/${OPENCV_VER}.tar.gz" "opencv-${OPENCV_VER}.tar.gz" "$OPENCV_SHA"
  fetch_and_verify "http://ceres-solver.org/ceres-solver-${CERES_VER}.tar.gz" "ceres-solver-${CERES_VER}.tar.gz" "$CERES_SHA"
  [[ -d "$P/src/opencv-${OPENCV_VER}" ]] || tar -xzf "$P/src/opencv-${OPENCV_VER}.tar.gz" -C "$P/src"
  [[ -d "$P/src/ceres-solver-${CERES_VER}" ]] || tar -xzf "$P/src/ceres-solver-${CERES_VER}.tar.gz" -C "$P/src"

  if [[ ! -f "$P/lib/cmake/opencv4/OpenCVConfig.cmake" ]]; then
    run_logged opencv-configure cmake -S "$P/src/opencv-${OPENCV_VER}" -B "$P/build/opencv" "${GEN[@]}" \
      -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$P" \
      -DBUILD_LIST=core,imgproc,imgcodecs,highgui,features2d,flann,calib3d,video \
      -DWITH_KLEIDICV=OFF -DWITH_FFMPEG=OFF -DWITH_OPENCL=OFF -DWITH_EIGEN=OFF -DWITH_TBB=OFF \
      -DWITH_OPENEXR=OFF -DWITH_AVIF=OFF \
      -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_opencv_apps=OFF \
      -DBUILD_JAVA=OFF -DBUILD_opencv_python3=OFF
    run_logged opencv-build cmake --build "$P/build/opencv" -j "$JOBS"
    run_logged opencv-install cmake --install "$P/build/opencv"
  fi

  if [[ ! -f "$P/lib/cmake/Ceres/CeresConfig.cmake" ]]; then
    run_logged ceres-configure cmake -S "$P/src/ceres-solver-${CERES_VER}" -B "$P/build/ceres" "${GEN[@]}" \
      -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$P" -DEigen3_DIR="$E3" \
      -DMINIGLOG=ON -DGFLAGS=OFF -DSUITESPARSE=OFF -DUSE_CUDA=OFF -DLAPACK=ON \
      -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARKS=OFF
    run_logged ceres-build cmake --build "$P/build/ceres" -j "$JOBS"
    run_logged ceres-install cmake --install "$P/build/ceres"
  fi
}

stage_openvins() {
  local src="$P/ws/src/open_vins"
  if [[ ! -d "$src/.git" ]]; then
    git init -q "$src"
    git -C "$src" remote add origin https://github.com/rpng/open_vins.git
  fi
  if [[ "$(git -C "$src" rev-parse HEAD 2>/dev/null || true)" != "$OV_SHA" ]]; then
    git -C "$src" fetch -q --depth 1 origin "$OV_SHA"
    git -C "$src" checkout -q FETCH_HEAD
  fi
  # Numbered patch series (only if the spike needed any), applied idempotently.
  shopt -s nullglob
  for p in "$HERE/../patches/open_vins/"*.patch; do
    if git -C "$src" apply --check "$p" 2>/dev/null; then git -C "$src" apply "$p"; log "applied $(basename "$p")"; fi
  done
  shopt -u nullglob

  # CMAKE_BUILD_TYPE is deliberately empty: OpenVINS adds -O3 -g3 itself, and NDEBUG stays
  # undefined so test_sim_repeat's assert()-based checks are live.
  run_logged ov-configure cmake -S "$src/ov_msckf" -B "$P/build/ov_msckf" "${GEN[@]}" -DCMAKE_BUILD_TYPE= \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DENABLE_ROS=OFF -DENABLE_ARUCO_TAGS=OFF -DEigen3_DIR="$E3" \
    -DOpenCV_DIR="$P/lib/cmake/opencv4" -DCeres_DIR="$P/lib/cmake/Ceres" -DCMAKE_PREFIX_PATH="$P" \
    -DCMAKE_INSTALL_PREFIX="$P" -DCMAKE_INSTALL_RPATH="$P/lib"
  grep -q "OPENCV: ${OPENCV_VER}" "$LOGS/ov-configure.txt" || { log "OpenVINS did not pick up OpenCV ${OPENCV_VER}"; exit 1; }
  run_logged ov-build cmake --build "$P/build/ov_msckf" -j "$JOBS"
  run_logged ov-install cmake --install "$P/build/ov_msckf"
}

stage_tests() {
  local b="$P/build/ov_msckf"
  local cfg="$P/ws/src/open_vins/config/rpng_sim/estimator_config.yaml"
  cd "$P/ws"   # rpng_sim's sim_traj_path is relative to a catkin-style workspace root
  nm -u "$b/test_sim_repeat" | grep -q ___assert_rtn || { log "test_sim_repeat was built with NDEBUG"; exit 1; }
  run_logged test_sim_repeat "$b/test_sim_repeat" "$cfg"
  grep -F -q 'success! they all are the same!' "$LOGS/test_sim_repeat.txt" || { log "test_sim_repeat: no success line"; exit 1; }
  run_logged run_simulation "$b/run_simulation" "$cfg"
  log "stock OpenVINS tests passed"
}

stage_helpers() {
  run_logged spike-configure cmake -S "$HERE" -B "$P/build/spike" "${GEN[@]}" -DCMAKE_BUILD_TYPE= \
    -DOV_PREFIX="$P" -DOV_SRC="$P/ws/src/open_vins" -DEigen3_DIR="$E3" -DCMAKE_PREFIX_PATH="$P"
  run_logged spike-build cmake --build "$P/build/spike" -j "$JOBS"
  cd "$P/ws"
  run_logged sim_rmse "$P/build/spike/sim_rmse" "$P/ws/src/open_vins/config/rpng_sim/estimator_config.yaml"
  run_logged klt_smoke "$P/build/spike/klt_smoke" "$P/build/spike"
  tail -5 "$LOGS/sim_rmse.txt"; tail -3 "$LOGS/klt_smoke.txt"
}

case "$STAGE" in
  deps) stage_deps ;;
  openvins) stage_openvins ;;
  tests) stage_tests ;;
  helpers) stage_helpers ;;
  all) stage_deps; stage_openvins; stage_tests; stage_helpers ;;
  *) echo "usage: $0 [all|deps|openvins|tests|helpers]" >&2; exit 2 ;;
esac
log "stage '$STAGE' done"
