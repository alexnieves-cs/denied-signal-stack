"""dss command line: doctor | fetch | baseline | sim | eval | demo | suite ..."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="dss")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("doctor")
    d.add_argument("--full", action="store_true")
    f = sub.add_parser("fetch")
    f.add_argument("keys", nargs="+")
    b = sub.add_parser("baseline")
    b.add_argument("kind", choices=["dr"])
    b.add_argument("--scenario")
    b.add_argument("--dataset")
    b.add_argument("--seed", type=int, default=42)
    b.add_argument("--out", default=None)
    s = sub.add_parser("sim")
    s.add_argument("action", choices=["record"])
    s.add_argument("--scenario", required=True)
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--out", required=True)
    for name in ("demo", "run"):
        r = sub.add_parser(name)
        r.add_argument("--scenario", default="configs/scenarios/demo.yaml")
        r.add_argument("--seed", type=int, default=42)
        r.add_argument("--lockstep", action="store_true")
        r.add_argument("--realtime", action="store_true")
        r.add_argument("--out", default=None)
        r.add_argument("--rrd", action="store_true", help="also write run.rrd")
        r.add_argument("--vio", default="auto", choices=["auto", "ovserver", "synthetic"])
    su = sub.add_parser("suite")
    su.add_argument("--seeds", type=int, default=5)
    su.add_argument("--out", default="eval/results")
    su.add_argument("--vio", default="auto", choices=["auto", "ovserver", "synthetic"])
    su.add_argument("--only", nargs="*")
    a = ap.parse_args(argv)

    if a.cmd == "doctor":
        from dss import doctor

        return doctor.main(a.full)
    if a.cmd == "fetch":
        from dss.datasets import fetch

        for k in a.keys:
            print(k, "->", fetch.fetch(k))
        return 0
    if a.cmd == "baseline":
        from dss.baselines import reports

        return reports.main(a)
    if a.cmd == "sim":
        from dss.sim import recorder, runner

        out = runner.simulate(runner.load_scenario(a.scenario), a.seed)
        h = recorder.save(out, a.out)
        for k, v in h.items():
            print(f"{k:8s} {v}")
        return 0
    if a.cmd in ("demo", "run"):
        from dss.app import run_cli

        return run_cli(a, demo=(a.cmd == "demo"))
    if a.cmd == "suite":
        from dss.app import suite_cli

        return suite_cli(a)
    return 2


if __name__ == "__main__":
    sys.exit(main())
