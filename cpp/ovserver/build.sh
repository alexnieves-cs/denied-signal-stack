#!/usr/bin/env bash
# Build ovserver against the Task 0.1 toolchain in $HOME/.cache/dss/cpp (run cpp/spike/build_openvins_macos.sh first).
# Usage: build.sh [test|release]   test (default): asserts live; release: -O3 -DNDEBUG for timing.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
P="$HOME/.cache/dss/cpp"
export PATH=/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin
MODE="${1:-test}"
E3="$(brew --prefix eigen@3)/share/eigen3/cmake"
BT=""; [[ "$MODE" == release ]] && BT=Release
cmake -S "$HERE" -B "$P/build/ovserver-$MODE" -G Ninja -DCMAKE_MAKE_PROGRAM=/opt/homebrew/bin/ninja \
  -DCMAKE_BUILD_TYPE="$BT" -DEigen3_DIR="$E3" -DCMAKE_PREFIX_PATH="$P" -DOV_PREFIX="$P" >/dev/null
cmake --build "$P/build/ovserver-$MODE" -j "${DSS_JOBS:-4}"
mkdir -p "$P/bin"
cp "$P/build/ovserver-$MODE/ovserver" "$P/bin/ovserver-$MODE"
[[ "$MODE" == test ]] && cp "$P/build/ovserver-$MODE/ovserver" "$P/bin/ovserver"
echo "built $P/bin/ovserver-$MODE"
