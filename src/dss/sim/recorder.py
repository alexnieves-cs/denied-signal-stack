"""Ground-truth + sensor recorder: one .npz per stream plus a hash manifest."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from dss.sensors.samples import Stream
from dss.sim.runner import SimOutput


def save(out: SimOutput, run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    for name, s in list(out.streams.items()) + [("gt", out.gt)]:
        np.savez(run_dir / f"{name}.npz", t_ns=s.t_ns, **s.fields)
    hashes = out.hashes()
    (run_dir / "streams.json").write_text(json.dumps({"seed": out.seed, "sha256": hashes}, indent=2, sort_keys=True) + "\n")
    return hashes


def load(run_dir: Path) -> dict[str, Stream]:
    out = {}
    for f in sorted(Path(run_dir).glob("*.npz")):
        d = dict(np.load(f))
        kind = "gt" if f.stem == "gt" else "".join(c for c in f.stem if not c.isdigit())
        out[f.stem] = Stream(f.stem, kind, d.pop("t_ns"), d)
    return out
