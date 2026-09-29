"""Report writing: metrics.json (canonical, no wall-clock in lockstep) + markdown summary."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from dss.core.manifest import dumps_canonical


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items() if not (isinstance(v, np.ndarray) and v.size > 64)}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if not np.isfinite(f) else round(f, 9)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def write_metrics(path: Path, metrics: dict) -> str:
    txt = dumps_canonical(_clean(metrics)) + "\n"
    Path(path).write_text(txt)
    return txt


def md_table(rows: list[dict], cols: list[str]) -> str:
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c, "")
            cells.append(f"{v:.3f}" if isinstance(v, float) else str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)
