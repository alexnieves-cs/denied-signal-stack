"""Deterministic RNG streams (ADR-0003).

Every stream is ``SeedSequence(root, spawn_key=(DOMAIN, stream_id))``. ``root.spawn()`` is
forbidden because its counter makes the stream depend on call order. Adding a stream id never
changes an existing stream.
"""

from __future__ import annotations

import zlib

import numpy as np

DOMAINS = {
    "imu": 1,
    "baro": 2,
    "mag": 3,
    "gnss": 4,
    "lidar": 5,
    "flow": 6,
    "camera": 7,
    "matcher": 8,
    "reanchor": 9,
    "scenario": 10,
    "vio": 11,
    "fusion": 12,
}


def stream_id(name: str) -> int:
    return zlib.crc32(name.encode()) & 0x7FFFFFFF


def rng(root_seed: int, domain: str, name: str = "default") -> np.random.Generator:
    ss = np.random.SeedSequence(int(root_seed), spawn_key=(DOMAINS[domain], stream_id(name)))
    return np.random.Generator(np.random.PCG64(ss))
