"""Matplotlib plots for reports (Agg backend; no display)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

COLORS = {"gt": "#222222", "est": "#1f77b4", "ref": "#2ca02c", "gps": "#d62728", "pub": "#9467bd", "dr": "#ff7f0e"}


def trajectory_plot(path: Path, series: dict[str, np.ndarray], title: str, events: list[tuple[str, np.ndarray]] | None = None):
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, p in series.items():
        ax.plot(p[:, 0], p[:, 1], label=name, color=COLORS.get(name.split()[0].lower()), lw=1.2)
    for label, xy in events or []:
        ax.plot(xy[0], xy[1], "x", ms=8)
        ax.annotate(label, xy[:2], fontsize=7)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("East [m]")
    ax.set_ylabel("North [m]")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def error_plot(path: Path, t: np.ndarray, series: dict[str, np.ndarray], title: str, bounds: dict[str, np.ndarray] | None = None,
               spans: list[tuple[float, float, str]] | None = None, vlines: list[tuple[float, str]] | None = None,
               ylabel: str = "horizontal error [m]", logy: bool = False):
    fig, ax = plt.subplots(figsize=(10, 4))
    for (a, b, lab) in spans or []:
        ax.axvspan(a, b, alpha=0.12, label=lab)
    for name, e in series.items():
        ax.plot(t, e, label=name, lw=1.0, color=COLORS.get(name.split()[0].lower()))
    for name, b in (bounds or {}).items():
        ax.plot(t, b, "--", label=name, lw=0.9)
    for x, lab in vlines or []:
        ax.axvline(x, color="k", ls=":", lw=0.8)
        ax.text(x, ax.get_ylim()[1] * 0.95, lab, rotation=90, fontsize=7, va="top")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("t [s]")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def anees_plot(path: Path, t: np.ndarray, anees_t: np.ndarray, lo, hi, title: str):
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(t, anees_t, lw=1.0, label="ANEES(t)")
    if lo is not None:
        ax.axhline(lo, color="r", ls="--", lw=0.8, label=f"band [{lo:.2f}, {hi:.2f}]")
    ax.axhline(hi, color="r", ls="--", lw=0.8)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("ANEES")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
