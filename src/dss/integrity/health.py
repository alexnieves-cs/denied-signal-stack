"""Per-source health vector."""

from __future__ import annotations

SOURCES = ("IMU", "VIO", "MAP", "GPS", "BARO", "MAG", "RANGE", "FLOW")
LEVEL = {"OK": 0, "DEGRADED": 1, "SUSPECT": 2, "FAILED": 3, "REJECTED": 3, "UNAVAILABLE": 1, "DISABLED": 1, "PROBATION": 1,
         "TRUSTED": 0}


class Health:
    def __init__(self):
        self.state = {s: "OK" for s in SOURCES}
        self.events: list[tuple[float, str, str, str]] = []

    def set(self, t: float, src: str, st: str, why: str = ""):
        if self.state.get(src) != st:
            self.events.append((t, src, st, why))
            self.state[src] = st

    def vector(self) -> dict:
        return dict(self.state)
