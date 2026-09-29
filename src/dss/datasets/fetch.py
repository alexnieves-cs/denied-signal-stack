"""Budgeted dataset fetcher (stream-extract) with a trust-on-first-use SHA-256 manifest."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from dss.core.config import CONFIG_DIR, REPO_ROOT

DATA = REPO_ROOT / "data"
MANIFEST = CONFIG_DIR / "datasets" / "manifest.json"

SOURCES = {
    "tumvi:room1": {"url": "https://cdn3.vision.in.tum.de/tumvi/exported/euroc/512_16/dataset-room1_512_16.tar",
                    "dir": "tumvi/dataset-room1_512_16", "kind": "tar"},
    "euroc:MH_01": {"url": "http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/machine_hall/MH_01_easy/MH_01_easy.zip",
                    "dir": "euroc/MH_01_easy", "kind": "zip"},
}


def load_manifest() -> dict:
    return json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}


def save_manifest(m: dict) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(m, indent=2, sort_keys=True) + "\n")


def record(key: str, sha256: str, nbytes: int) -> dict:
    """Trust on first use: the first hash is recorded; later fetches must match it."""
    m = load_manifest()
    entry = m.get(key)
    if entry and entry["sha256"] != sha256:
        raise RuntimeError(f"{key}: SHA-256 changed ({entry['sha256']} -> {sha256})")
    m[key] = {"url": SOURCES.get(key, {}).get("url"), "sha256": sha256, "bytes": nbytes,
              "provenance": entry.get("provenance", "tofu") if entry else "tofu"}
    save_manifest(m)
    return m[key]


def fetch(key: str) -> Path:
    src = SOURCES[key]
    dst = DATA / src["dir"]
    if dst.exists():
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src["kind"] != "tar":
        raise RuntimeError(f"{key}: only streamed tar sources are supported ({src['url']})")
    h = hashlib.sha256()
    n = 0
    tar = subprocess.Popen(["tar", "-x", "-f", "-", "-C", str(dst.parent)], stdin=subprocess.PIPE)
    curl = subprocess.Popen(["curl", "-sfL", src["url"]], stdout=subprocess.PIPE)
    assert curl.stdout and tar.stdin
    for chunk in iter(lambda: curl.stdout.read(1 << 20), b""):
        h.update(chunk)
        n += len(chunk)
        tar.stdin.write(chunk)
    tar.stdin.close()
    if curl.wait() or tar.wait():
        raise RuntimeError(f"{key}: fetch failed")
    record(key, h.hexdigest(), n)
    return dst


def path(key: str) -> Path:
    return DATA / SOURCES[key]["dir"]
