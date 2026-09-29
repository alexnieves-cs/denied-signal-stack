"""Run manifest: everything needed to reproduce a run. No wall-clock fields in lockstep mode."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from dss.core.config import REPO_ROOT


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: str | Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_array(a: np.ndarray) -> str:
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def git_dirty() -> bool:
    try:
        out = subprocess.check_output(["git", "status", "--porcelain", "--", "src", "configs"], cwd=REPO_ROOT, text=True)
        return bool(out.strip())
    except Exception:
        return True


def build_manifest(seed: int, config: dict, extra: dict | None = None) -> dict:
    lock = REPO_ROOT / "uv.lock"
    m = {
        "seed": int(seed),
        "config_sha256": sha256_bytes(json.dumps(config, sort_keys=True, default=str).encode()),
        "uv_lock_sha256": sha256_file(lock) if lock.exists() else None,
    }
    if extra:
        m.update(extra)
    return m


def dumps_canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, indent=2, default=_json_default)


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))
