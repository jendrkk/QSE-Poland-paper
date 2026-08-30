#!/usr/bin/env python3
"""
run_partitions.py — historical-partition analysis & counterfactuals runner.

Loads one or more solved baseline runs, attaches the Prussian/Russian/Austrian
partition label from communes_2021_partitions.gpkg, and runs any of three
experiments (see qse_poland_paper/partitions.py):

  gaps      partition means & gaps in recovered fundamentals (+ optional controls)
  border    border-imposition counterfactual (welfare cost of re-drawing the seam)
  removal   gap-removal counterfactual (equalise a partition mean, GE effect)

Outputs LaTeX tables + a JSON summary to runs/partitions__<run_id>/ (or --outdir).

Examples
--------
  # everything, all three years, default border cost 1.5x on commuting
  python run_partitions.py runs/2011_garmin_baseline runs/2021_osm_baseline \
      runs/2026_osm_baseline --experiment all

  # gaps controlling for voivodeship FE and distance to Warsaw
  python run_partitions.py runs/2021_osm_baseline --experiment gaps \
      --controls woj log_tt_warsaw

  # border on commuting AND trade, 2x
  python run_partitions.py runs/2021_osm_baseline --experiment border \
      --channels both --commute-cost 2.0 --trade-cost 1.5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from qse_poland_paper.result import RunResult
from qse_poland_paper import partitions as P


def _infer_gpkg(run_path: Path, override):
    if override:
        return Path(override)
    for up in (1, 2, 3):
        cand = (run_path.resolve().parents[up]
                / "data/processed/shapefiles/communes_2021_partitions.gpkg")
        if cand.exists():
            return cand
    return None


def _fmt_gap(t):
    b, se = t
    star = "***" if abs(b) > 2.58 * se else "**" if abs(b) > 1.96 * se \
        else "*" if abs(b) > 1.64 * se else ""
    return f"{b:+.3f}{star} ({se:.3f})"


def _write_gaps_tex(res, path, year):
    lines = [r"\begin{table}[htbp]\centering",
             rf"\caption{{Partition gaps in recovered fundamentals, {year} "
             rf"(base = Russian; weight = {res['weight']}"
             + (f"; controls = {', '.join(res['controls'])}" if res['controls'] else "")
             + ")}",
             r"\label{tab:partition_gaps_%d}" % year,
             r"\begin{tabular}{lrrr}\hline\hline",
             r"Object & Prussian $-$R & Austrian $-$R & mean (P/R/A) \\ \hline"]
    for key, o in res["objects"].items():
        g = o["gaps_vs_base"]
        m = o["means"]
        lines.append(
            rf"{o['label']} & {_fmt_gap(g['P'])} & {_fmt_gap(g['A'])} & "
            rf"{m['P']:+.2f}/{m['R']:+.2f}/{m['A']:+.2f} \\")
    lines += [r"\hline\hline\end{tabular}", r"\end{table}"]
    Path(path).write_text("\n".join(lines) + "\n")


def _write_scalar_tex(rows, path, caption, label):
    lines = [r"\begin{table}[htbp]\centering", rf"\caption{{{caption}}}",
             rf"\label{{{label}}}", r"\begin{tabular}{lr}\hline\hline",
             r"Quantity & Value \\ \hline"]
    for k, v in rows:
        lines.append(rf"{k} & {v} \\")
    lines += [r"\hline\hline\end{tabular}", r"\end{table}"]
    Path(path).write_text("\n".join(lines) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description="MRRH Poland partition analysis runner")
    ap.add_argument("runs", nargs="+", help="one or more run.pkl (or run dir) paths")
    ap.add_argument("--gpkg", default=None,
                    help="partition GeoPackage (auto-inferred: communes_2021_partitions.gpkg)")
    ap.add_argument("--experiment", default="all",
                    choices=["gaps", "border", "removal", "all"])
    ap.add_argument("--outdir", default=None)
    # gaps
    ap.add_argument("--weight", default="R_n", choices=["R_n", "L_n", "none"])
    ap.add_argument("--controls", nargs="*", default=[],
                    help="woj | logpop | log_tt_warsaw")
    ap.add_argument("--objects", nargs="*", default=None,
                    help="subset of A_n b_n CMA real_v")
    # border
    ap.add_argument("--commute-cost", type=float, default=1.5,
                    help="multiplicative commuting-cost factor on cross-seam pairs")
    ap.add_argument("--trade-cost", type=float, default=1.0,
                    help="multiplicative trade-cost factor on cross-seam pairs")
    ap.add_argument("--channels", default="commute",
                    choices=["commute", "trade", "both"])
    # removal
    ap.add_argument("--target", default="A_n", choices=["A_n", "b_n"],
                    help="fundamental whose partition mean is equalised")
    args = ap.parse_args(argv)

    run_paths = [Path(r) for r in args.runs]
    gpkg = _infer_gpkg(run_paths[0], args.gpkg)
    if gpkg is None:
        raise SystemExit("no partition GeoPackage found; pass --gpkg")
    print(f"partition gpkg: {gpkg}")

    do = ({"gaps", "border", "removal"} if args.experiment == "all"
          else {args.experiment})

    for rp in run_paths:
        run = RunResult.load(rp)
        part = P.load_partition(gpkg, run.codes)
        outdir = Path(args.outdir) if args.outdir else (
            rp.resolve().parents[1] / f"partitions__{run.run_id}")
        tabdir = outdir / "tables"; tabdir.mkdir(parents=True, exist_ok=True)
        summary = {"run_id": run.run_id, "year": run.year,
                   "partition_sizes": {p: int((part == p).sum()) for p in P.PARTITIONS}}
        print(f"\n[{run.run_id}]  sizes P/R/A = "
              f"{summary['partition_sizes']['P']}/{summary['partition_sizes']['R']}/"
              f"{summary['partition_sizes']['A']}  -> {outdir}")

        if "gaps" in do:
            g = P.partition_gaps(run, part, objects=args.objects, weight=args.weight,
                                 controls=args.controls, gpkg_path=gpkg)
            _write_gaps_tex(g, tabdir / "partition_gaps.tex", run.year)
            summary["gaps"] = g
            for key, o in g["objects"].items():
                gp, ga = o["gaps_vs_base"]["P"], o["gaps_vs_base"]["A"]
                print(f"    gap {key:7s}: P-R {gp[0]:+.3f} ({gp[1]:.3f})  "
                      f"A-R {ga[0]:+.3f} ({ga[1]:.3f})")

        if "border" in do:
            b = P.impose_border(run, part, commute_cost=args.commute_cost,
                                trade_cost=args.trade_cost, channels=args.channels)
            _write_scalar_tex(
                [("Channels", b["channels"]),
                 ("Commuting-cost factor", f"{b['commute_cost']:.2f}"),
                 ("Trade-cost factor", f"{b['trade_cost']:.2f}"),
                 ("Welfare change (\\%)", f"{b['welfare_pct']:.3f}"),
                 ("s.d.\\ $\\log\\hat r_n$", f"{float(np.std(np.log(b['r']))):.4f}"),
                 ("s.d.\\ $\\log\\hat l_n$", f"{float(np.std(np.log(b['l']))):.4f}"),
                 ("converged", b["converged"])],
                tabdir / "border_ge.tex",
                f"Border-imposition GE, {run.year} (partition seams)",
                f"tab:border_ge_{run.year}")
            summary["border"] = {k: (float(v) if isinstance(v, (int, float, np.floating))
                                     else v)
                                 for k, v in b.items()
                                 if k in ("welfare_pct", "channels", "commute_cost",
                                          "trade_cost", "converged", "iters")}
            print(f"    border [{b['channels']}, x{b['commute_cost']}]: "
                  f"welfare {b['welfare_pct']:+.3f}%  conv={b['converged']}")

        if "removal" in do:
            rm = P.remove_gap(run, part, target=args.target, weight=args.weight)
            _write_scalar_tex(
                [("Target fundamental", rm["target"]),
                 ("Welfare change (\\%)", f"{rm['welfare_pct']:.3f}"),
                 ("s.d.\\ $\\log\\hat l_n$", f"{float(np.std(np.log(rm['l']))):.4f}"),
                 ("converged", rm["converged"])],
                tabdir / "removal_ge.tex",
                f"Gap-removal GE, {run.year} (equalise {rm['target']} partition mean)",
                f"tab:removal_ge_{run.year}")
            summary["removal"] = {"target": rm["target"],
                                  "welfare_pct": float(rm["welfare_pct"]),
                                  "converged": bool(rm["converged"])}
            print(f"    removal [{rm['target']}]: welfare {rm['welfare_pct']:+.3f}%")

        (outdir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
        print(f"    wrote tables + summary.json -> {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
