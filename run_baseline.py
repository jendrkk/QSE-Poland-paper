#!/usr/bin/env python3
"""
run_baseline.py — orchestrator (front-end) for the MRRH Poland baseline.

Reads the central config, calibrates and solves the model for each requested
year, and writes one self-describing RunResult per year to runs/<run_id>/.

Years with observed flows (2011, 2021) are solved directly; 2026 (no flows)
borrows the estimated phi from its configured source year, which is therefore
solved first automatically.

Usage
-----
  python run_baseline.py --config config/baseline.yaml
  python run_baseline.py --config config/baseline.yaml --years 2021 2026
  python run_baseline.py --config config/baseline.yaml --repo-root /path/to/repo

The runs directory layout:
  runs/<run_id>/run.pkl               # the single complete artefact
  runs/<run_id>/manifest.json         # human-readable scalars/params/provenance
  runs/<run_id>/config.resolved.yaml  # exact config used
  runs/<run_id>/run.log
"""
from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

import yaml

from qse_poland_paper.config import RunConfig
from qse_poland_paper.solve import calibrate_year
from qse_poland_paper import validate


LOG = logging.getLogger("mrrh.run")


def _order_years(cfg: RunConfig, years):
    """Solve source years (those others borrow from) before their dependents."""
    years = list(years)
    ordered, seen = [], set()

    def add(y):
        if y in seen:
            return
        src = cfg.years[y].borrow_params_from
        if src is not None and src in years:
            add(src)
        ordered.append(y)
        seen.add(y)

    for y in years:
        add(y)
    return ordered


def main(argv=None):
    ap = argparse.ArgumentParser(description="MRRH Poland baseline orchestrator")
    ap.add_argument("--config", default="config/baseline.yaml")
    ap.add_argument("--repo-root", default=None,
                    help="repository root (default: parent of the config file)")
    ap.add_argument("--years", nargs="*", type=int, default=None,
                    help="subset of years to solve (default: all configured)")
    ap.add_argument("--tag", default=None, help="override the run tag")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg = RunConfig.from_yaml(args.config, repo_root=args.repo_root)
    if args.tag:
        cfg.tag = args.tag
    years = args.years or sorted(cfg.years.keys())
    order = _order_years(cfg, years)

    runs_dir = cfg.paths.abspath(cfg.paths.runs_dir)
    phi_by_year: dict[int, float] = {}
    written = []

    for year in order:
        spec = cfg.years[year]
        borrowed = None
        if spec.flows is None and spec.borrow_params_from is not None:
            borrowed = phi_by_year.get(spec.borrow_params_from)
            if borrowed is None:
                # source year not solved this session; solve it silently first
                src_rr = calibrate_year(cfg, spec.borrow_params_from)
                phi_by_year[spec.borrow_params_from] = src_rr.params["phi"]
                borrowed = src_rr.params["phi"]

        print(f"[{year}] solving {cfg.run_id(year)} ...", flush=True)
        rr = calibrate_year(cfg, year, borrowed_phi=borrowed)
        phi_by_year[year] = rr.params["phi"]

        hard_ok = validate.all_hard_passed(rr.validation)
        print(validate.summary(rr.validation))
        print(f"    phi={rr.params['phi']:.4f}  prod_iters={rr.diagnostics.get('prod_iters')}"
              f"  gap={rr.diagnostics.get('prod_gap'):.2e}"
              f"  flows={rr.meta['flows_source']}")

        if not hard_ok and cfg.solver.strict_invariants:
            print(f"    REFUSING to save {cfg.run_id(year)}: hard invariant failed.",
                  file=sys.stderr)
            continue

        run_dir = runs_dir / cfg.run_id(year)
        pkl = rr.save(run_dir)
        (run_dir / "config.resolved.yaml").write_text(
            yaml.safe_dump(cfg.to_dict(), sort_keys=False, allow_unicode=True))
        written.append(pkl)
        print(f"    saved -> {pkl}")

    print("\nDone. Runs written:")
    for p in written:
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
