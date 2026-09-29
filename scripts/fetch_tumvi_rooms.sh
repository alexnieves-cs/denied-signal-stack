#!/usr/bin/env bash
# Stream-extract TUM-VI rooms (EuRoC export, 512_16) into data/tumvi, recording SHA-256 per tar.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../data/tumvi"
for r in "$@"; do
  d="dataset-${r}_512_16"
  [[ -d "$d/mav0/mocap0" ]] && continue
  curl -sfL "https://cdn3.vision.in.tum.de/tumvi/exported/euroc/512_16/${d}.tar" | tee >(shasum -a 256 > "${r}.tar.sha256") | tar -x -f -
  echo "done $r $(cat ${r}.tar.sha256)"
done
