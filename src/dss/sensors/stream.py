"""Merged, time-ordered sensor stream with a deterministic tie-break.

Order key: (t_ns, kind priority, stream name, index). The result does not depend on the order
in which streams are passed in.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from dss.sensors.samples import PRIORITY, Stream


def merge_order(streams: list[Stream]) -> np.ndarray:
    """Return a structured array (t_ns, prio, name_rank, sid, idx) sorted by the order key."""
    names = sorted(s.name for s in streams)
    rank = {n: i for i, n in enumerate(names)}
    parts = []
    for s in streams:
        n = len(s)
        a = np.empty(n, dtype=[("t", "i8"), ("prio", "i4"), ("name", "i4"), ("idx", "i8")])
        a["t"] = s.t_ns
        a["prio"] = PRIORITY.get(s.kind, 8)
        a["name"] = rank[s.name]
        a["idx"] = np.arange(n)
        parts.append(a)
    allv = np.concatenate(parts) if parts else np.empty(0, dtype=[("t", "i8"), ("prio", "i4"), ("name", "i4"), ("idx", "i8")])
    order = np.lexsort((allv["idx"], allv["name"], allv["prio"], allv["t"]))
    out = allv[order]
    return out


def iterate(streams: list[Stream]) -> Iterator[tuple[int, Stream, int]]:
    by_rank = {i: s for i, s in enumerate(sorted(streams, key=lambda s: s.name))}
    for rec in merge_order(streams):
        yield int(rec["t"]), by_rank[int(rec["name"])], int(rec["idx"])
