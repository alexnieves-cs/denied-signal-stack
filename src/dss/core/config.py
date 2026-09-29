"""YAML config loading with ``extends:`` inheritance and dotted overrides."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / "configs"


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def resolve(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute() or p.exists():
        return p
    q = CONFIG_DIR / p
    if q.exists():
        return q
    if q.with_suffix(".yaml").exists():
        return q.with_suffix(".yaml")
    raise FileNotFoundError(path)


def load(path: str | Path) -> dict[str, Any]:
    p = resolve(path)
    data = yaml.safe_load(p.read_text()) or {}
    parent = data.pop("extends", None)
    if parent:
        data = deep_merge(load(parent), data)
    return data


def set_dotted(cfg: dict, key: str, value: Any) -> None:
    parts = key.split(".")
    d = cfg
    for p in parts[:-1]:
        d = d.setdefault(p, {})
    d[parts[-1]] = value
