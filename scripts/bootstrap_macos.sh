#!/usr/bin/env bash
# Bootstrap this Mac: Homebrew bottles, uv-managed CPython 3.13.15, venv, doctor.
set -euo pipefail
export PATH=/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin   # pip ninja from other envs must not shadow
cd "$(dirname "${BASH_SOURCE[0]}")/.."
brew bundle --file=Brewfile
uv python install 3.13.15
uv sync
uv tool install evo==1.37.1 || echo "evo optional (GPL, isolated tool) not installed"
uv run dss doctor
