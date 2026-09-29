#!/usr/bin/env bash
# Produce demo.mp4 (1920x1080, 30 fps, 1x real time) for a finished run directory, e.g. runs/demo-42.
# The one-screen dashboard (same 3-column layout as the Rerun blueprint in run.rrd) is rendered offline from
# the run's logs, so no Screen Recording (TCC) permission is needed. Interactive alternative:
#   uv run rerun runs/demo-42/run.rrd   (then record the viewer window yourself)
set -euo pipefail
RUN="${1:?usage: record_demo.sh runs/demo-42}"
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PATH=/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin
uv run python -m dss.dashboard.video "$RUN"
ffprobe -v error -show_entries stream=width,height,r_frame_rate -show_entries format=duration "$RUN/demo.mp4"
