#!/usr/bin/env python3
"""
run_partitions.py — historical-partition analysis & counterfactuals runner.

Loads one or more solved baseline runs, attaches the Prussian/Russian/Austrian
partition label from communes_2021_partitions.gpkg, and runs any of three
experiments (see qse_poland_paper/partitions.py):

  gaps      partition means & gaps in recovered fundamentals (+ optional controls)
  border    border-imposition counterfactual: sensitivity sweep over the
            commuting-/trade-cost factor applied to cross-seam pairs
  removal   gap-removal counterfactual: equalise a partition mean, GE effect --
            productivity (A_n) only, amenity (b_n) only, and both jointly

Outputs LaTeX tables + a JSON summary to runs/partitions__<run_id>/ (or --outdir).
`removal_ge.tex` has one row per removal variant (default: all three); `border_ge.tex`
has one row per (channel, cost-factor) combination swept (default factors 1.0-3.0).

Examples
--------
  # everything, all three years, default removal variants + border sweep
  python run_partitions.py runs/2011_garmin_baseline runs/2021_osm_baseline \
      runs/2026_osm_baseline --experiment all

  # gaps controlling for voivodeship FE and distance to Warsaw
  python run_partitions.py runs/2021_osm_baseline --experiment gaps \
      --controls woj log_tt_warsaw

  # border sensitivity with a coarser sweep
  python run_partitions.py runs/2021_osm_baseline --experiment border \
      --border-sweep 1.0 1.5 2.0

  # removal table with only the productivity and joint variants
  python run_partitions.py runs/2021_osm_baseline --experiment removal \
      --removal-targets prod both
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from qse_poland_paper.result import RunResult
from qse_poland_paper import partitions as P

# Gap-removal variants -> the fundamental(s) equalised across P/R/A (mirrors
# run_partition_viz.py's REMOVAL_TARGET so the table rows and the six-hat maps
# describe the same three counterfactuals).
REMOVAL_TARGET = {"prod": "A_n", "qol": "b_n", "both": ("A_n", "b_n")}
REMOVAL_LABEL = {"prod": r"$A_n$ only", "qol": r"$b_n$ only",
                 "both": r"$A_n$ \& $b_n$ (joint)"}


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


def _write_removal_tex(results, path, year):
    """`results` = list of (row_label, remove_gap()-result dict), one row per
    removal variant (default: A_n only, b_n only, both jointly) -- all for the
    same year, in one table."""
    lines = [r"\begin{table}[htbp]\centering",
             rf"\caption{{Gap-removal GE, {year} (equalise partition mean(s))}}",
             r"\label{tab:removal_ge_%d}" % year,
             r"\begin{tabular}{lrrrl}\hline\hline",
             r"Target & Welfare (\%) & s.d.\ $\log\hat r_n$ & "
             r"s.d.\ $\log\hat l_n$ & converged \\ \hline"]
    for label, rm in results:
        lines.append(
            rf"{label} & {rm['welfare_pct']:+.3f} & "
            rf"{float(np.std(np.log(rm['r']))):.4f} & "
            rf"{float(np.std(np.log(rm['l']))):.4f} & {rm['converged']} \\")
    lines += [r"\hline\hline\end{tabular}", r"\end{table}"]
    Path(path).write_text("\n".join(lines) + "\n")


def _write_border_tex(rows, path, year, sweep):
    """`rows` = list of dicts (channel, commute_cost, trade_cost, welfare_pct,
    sd_r, sd_l, converged), one per swept (channel, factor) combination -- the
    border-imposition sensitivity table (replaces the old single-point table)."""
    lines = [r"\begin{table}[htbp]\centering",
             rf"\caption{{Border-imposition GE sensitivity, {year} (partition "
             rf"seams; factor sweep = {', '.join(f'{s:g}' for s in sweep)})}}",
             r"\label{tab:border_ge_%d}" % year,
             r"\begin{tabular}{lrrrrrl}\hline\hline",
             r"Channel & Commute$\times$ & Trade$\times$ & Welfare (\%) & "
             r"s.d.\ $\log\hat r_n$ & s.d.\ $\log\hat l_n$ & converged \\ \hline"]
    for r in rows:
        lines.append(
            rf"{r['channel']} & {r['commute_cost']:.2f} & {r['trade_cost']:.2f} & "
            rf"{r['welfare_pct']:+.3f} & {r['sd_r']:.4f} & {r['sd_l']:.4f} & "
            rf"{r['converged']} \\")
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
    # border -- sensitivity sweep over the cost factor, three channel rows per
    # factor: commuting only, trade only, and both channels scaled together
    ap.add_argument("--border-sweep", nargs="+", type=float,
                    default=[1.0, 1.25, 1.5, 2.0, 3.0],
                    help="cost factors to sweep for the border_ge.tex sensitivity "
                         "table (applied to commute-only, trade-only, and "
                         "both-channels-jointly rows)")
    # removal
    ap.add_argument("--removal-targets", nargs="+", default=["prod", "qol", "both"],
                    choices=["prod", "qol", "both"],
                    help="gap-removal variants to include in removal_ge.tex: prod "
                         "(equalise A_n means), qol (equalise b_n means), both "
                         "(A_n & b_n jointly) -- default: all three, one table per year")
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
            sweep = args.border_sweep
            border_rows = []

            def _run_border(channel_label, commute_cost, trade_cost, channels):
                b = P.impose_border(run, part, commute_cost=commute_cost,
                                    trade_cost=trade_cost, channels=channels)
                row = dict(channel=channel_label, commute_cost=commute_cost,
                          trade_cost=trade_cost, welfare_pct=b["welfare_pct"],
                          sd_r=float(np.std(np.log(b["r"]))),
                          sd_l=float(np.std(np.log(b["l"]))),
                          converged=b["converged"])
                border_rows.append(row)
                print(f"    border [{channel_label:16s} x{commute_cost:.2f}/"
                      f"{trade_cost:.2f}]: welfare {b['welfare_pct']:+.3f}%  "
                      f"conv={b['converged']}")

            for c in sweep:
                _run_border("commute only", c, 1.0, "commute")
            for t in sweep:
                _run_border("trade only", 1.0, t, "trade")
            for f in sweep:
                _run_border("both (matched)", f, f, "both")

            _write_border_tex(border_rows, tabdir / "border_ge.tex", run.year, sweep)
            summary["border"] = {"sweep": sweep, "rows": border_rows}

        if "removal" in do:
            removal_results = []
            summary["removal"] = {}
            for variant in args.removal_targets:
                rm = P.remove_gap(run, part, target=REMOVAL_TARGET[variant],
                                  weight=args.weight)
                removal_results.append((REMOVAL_LABEL[variant], rm))
                summary["removal"][variant] = {
                    "target": rm["target"], "welfare_pct": float(rm["welfare_pct"]),
                    "converged": bool(rm["converged"])}
                print(f"    removal [{variant:4s} -> {rm['target']}]: "
                      f"welfare {rm['welfare_pct']:+.3f}%  conv={rm['converged']}")
            _write_removal_tex(removal_results, tabdir / "removal_ge.tex", run.year)

        (outdir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
        print(f"    wrote tables + summary.json -> {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
