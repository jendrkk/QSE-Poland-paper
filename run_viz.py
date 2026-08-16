#!/usr/bin/env python3
"""
run_viz.py — visualization runner for MRRH Poland run bundles.

Takes one or more RunResult pickles and produces LaTeX-styled figures, gmina
maps and LaTeX tables. With a single run it draws that run's diagnostics and
fundamentals; with two or more it additionally produces a comparison set,
including the general-equilibrium implication of the road-network change between
the two vintages (a pure commuting-cost counterfactual on the earlier
calibration using the later travel-time matrix).

Outputs go to <run_dir>/figures and <run_dir>/tables for a single run, and to
runs/compare__<id1>__<id2>/ for a comparison, unless --outdir is given.

Usage
-----
  python run_viz.py runs/2021_osm_baseline/run.pkl
  python run_viz.py runs/2011_garmin_baseline/run.pkl runs/2021_osm_baseline/run.pkl
  python run_viz.py RUN.pkl --dpi 400 --format pdf --no-transparent --gpkg path/to.gpkg
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from qse_poland_paper.result import RunResult
from qse_poland_paper.viz import style, maps, figures, tables
from qse_poland_paper import counterfac as cf


def _infer_gpkg(run_path: Path, override):
    if override:
        return Path(override)
    # runs/<id>/run.pkl -> repo root is parents[2]
    for up in (2, 3):
        cand = run_path.resolve().parents[up] / "data/processed/shapefiles/communes_2021.gpkg"
        if cand.exists():
            return cand
    return None


# level maps to draw for a single run: (calibrated/inputs key, section, label, log)
_LEVEL_MAPS = [
    ("A_n", "calibrated", r"$\log A_n$ (productivity)", True),
    ("b_n", "calibrated", r"$\log b_n$ (amenity)", True),
    ("CMA", "calibrated", r"$\log$ CMA$_n$", True),
    ("P_n", "calibrated", r"$\log P_n$ (price index)", True),
    ("real_v", "calibrated", r"$\log v_n/$CPI (real income)", True),
    ("w_n", "calibrated", r"$\log w_n$ (wage)", True),
    ("Q_n", "inputs", r"$\log Q_n$ (floor price)", True),
]


def visualize_single(run: RunResult, gpkg, figdir, tabdir, *, dpi, transparent, fmt):
    figdir = Path(figdir); tabdir = Path(tabdir)
    codes = run.codes
    made = []

    # --- maps ---------------------------------------------------------------
    if gpkg is not None:
        for key, sec, label, log in _LEVEL_MAPS:
            try:
                vals = np.asarray(getattr(run, sec)[key], float)
                vals = np.log(vals) if log else vals
                made.append(maps.choropleth(
                    gpkg, codes, vals, title=f"{label}, {run.year}", label=label,
                    dpi=dpi, transparent=transparent, fmt=fmt,
                    outpath=figdir / f"map_{key}_{run.year}"))
            except Exception as e:
                print(f"  [map {key}] skipped: {e}")
        # own-commute share (per residence gmina), from the OBSERVED matrix
        try:
            com = run.inputs.get("comMat_observed", run.inputs["comMat"])
            own_share = np.diag(com) / com.sum(axis=1)
            made.append(maps.choropleth(
                gpkg, codes, own_share, title=f"Own-commute share, {run.year}",
                label="share", dpi=dpi, transparent=transparent, fmt=fmt,
                outpath=figdir / f"map_own_share_{run.year}"))
        except Exception as e:
            print(f"  [map own_share] skipped: {e}")

    # --- figures ------------------------------------------------------------
    try:
        made.append(figures.gravity_fit(
            run.inputs.get("comMat_observed", run.inputs["comMat"]),
            run.inputs["tau"], run.params["phi"],
            title=f"Commuting gravity, {run.year}", dpi=dpi, transparent=transparent,
            outpath=figdir / f"gravity_{run.year}"))
    except Exception as e:
        print(f"  [gravity] skipped: {e}")
    for key, label in [("A_n", r"$A_n$"), ("b_n", r"$b_n$")]:
        try:
            made.append(figures.distribution(
                run.calibrated[key], label, title=f"Distribution of {label}, {run.year}",
                dpi=dpi, transparent=transparent, outpath=figdir / f"dist_{key}_{run.year}"))
        except Exception as e:
            print(f"  [dist {key}] skipped: {e}")
    try:
        made.append(figures.lorenz(run.inputs["L_n"], label="workplace employment",
                    title=f"Employment concentration, {run.year}", dpi=dpi,
                    transparent=transparent, outpath=figdir / f"lorenz_L_{run.year}"))
    except Exception as e:
        print(f"  [lorenz] skipped: {e}")
    for (xk, xs), (yk, ys), xl, yl, tag in [
            (("w_n", "calibrated"), ("A_n", "calibrated"), r"$w_n$", r"$A_n$", "A_vs_w"),
            (("Q_n", "inputs"), ("b_n", "calibrated"), r"$Q_n$", r"$b_n$", "b_vs_Q"),
            (("CMA", "calibrated"), ("A_n", "calibrated"), r"CMA$_n$", r"$A_n$", "A_vs_CMA")]:
        try:
            made.append(figures.scatter(
                getattr(run, xs)[xk], getattr(run, ys)[yk], xl, yl,
                title=f"{yl} vs {xl}, {run.year}", dpi=dpi, transparent=transparent,
                outpath=figdir / f"scatter_{tag}_{run.year}"))
        except Exception as e:
            print(f"  [scatter {tag}] skipped: {e}")

    # --- tables -------------------------------------------------------------
    for fn, name in [(tables.gravity_table, "gravity"),
                     (tables.calibration_table, "calibration"),
                     (tables.moments_table, "diagnostics")]:
        try:
            made.append(fn(run, tabdir / f"{name}_{run.year}.tex"))
        except Exception as e:
            print(f"  [table {name}] skipped: {e}")
    return made


def visualize_comparison(runs, gpkg, outdir, *, dpi, transparent, fmt):
    outdir = Path(outdir)
    figdir, tabdir = outdir / "figures", outdir / "tables"
    made = []
    a, b = runs[0], runs[-1]                 # earliest, latest by input order
    codes = a.codes

    # cross-run comparison table
    try:
        made.append(tables.comparison_table(runs, tabdir / "comparison.tex"))
    except Exception as e:
        print(f"  [compare table] skipped: {e}")

    # fundamentals across the two runs (45-degree)
    for key, lab in [("A_n", "A_n"), ("b_n", "b_n")]:
        try:
            made.append(figures.compare_scatter(
                a.calibrated[key], b.calibrated[key],
                rf"$\log {lab}$ ({a.year})", rf"$\log {lab}$ ({b.year})",
                title=rf"${lab}$: {a.year} vs {b.year}", dpi=dpi, transparent=transparent,
                outpath=figdir / f"compare_{key}"))
        except Exception as e:
            print(f"  [compare {key}] skipped: {e}")

    # Delta-log maps of fundamentals and market access
    if gpkg is not None:
        for key, sec, lab in [("A_n", "calibrated", "A_n"), ("b_n", "calibrated", "b_n"),
                              ("CMA", "calibrated", "CMA_n"), ("Q_n", "inputs", "Q_n")]:
            try:
                dv = np.log(np.asarray(getattr(b, sec)[key], float)) \
                    - np.log(np.asarray(getattr(a, sec)[key], float))
                made.append(maps.choropleth(
                    gpkg, codes, dv, diverging=True,
                    title=rf"$\Delta\log {lab}$, {a.year}$\to${b.year}",
                    label=rf"$\Delta\log {lab}$", dpi=dpi, transparent=transparent,
                    fmt=fmt, outpath=figdir / f"delta_{key}"))
            except Exception as e:
                print(f"  [delta {key}] skipped: {e}")

    # --- road-network GE counterfactual: earlier calibration, later network ---
    try:
        res = cf.network_counterfactual(a, b.inputs["tau"])
        welf_pct = (res["welf"] - 1) * 100
        (tabdir).mkdir(parents=True, exist_ok=True)
        (tabdir / "network_ge.tex").write_text(
            "% Road-network GE counterfactual: {a}->{b} network on {a} fundamentals\n"
            "\\begin{{table}}[htbp]\\centering\n"
            "\\caption{{Road-network general equilibrium, {a}$\\to${b} network}}\n"
            "\\label{{tab:network_ge}}\n\\begin{{tabular}}{{lr}}\\hline\\hline\n"
            "Quantity & Value \\\\ \\hline\n"
            "Aggregate welfare change (\\%) & {w:.3f} \\\\\n"
            "s.d.\\ $\\log \\hat r_n$ (residence reallocation) & {r:.4f} \\\\\n"
            "s.d.\\ $\\log \\hat l_n$ (employment reallocation) & {l:.4f} \\\\\n"
            "converged & {c} \\\\\n\\hline\\hline\\end{{tabular}}\\end{{table}}\n".format(
                a=a.year, b=b.year, w=welf_pct,
                r=float(np.std(np.log(res["r"]))), l=float(np.std(np.log(res["l"]))),
                c=res["converged"]))
        made.append(tabdir / "network_ge.tex")
        if gpkg is not None:
            made.append(maps.choropleth(
                gpkg, codes, np.log(res["r"]), diverging=True,
                title=rf"Residence reallocation $\log\hat r_n$, {a.year}$\to${b.year} network",
                label=r"$\log\hat r_n$", dpi=dpi, transparent=transparent, fmt=fmt,
                outpath=figdir / "ge_residence_reallocation"))
        print(f"  road-network GE welfare change {a.year}->{b.year}: {welf_pct:+.3f}%")
    except Exception as e:
        print(f"  [network GE] skipped: {e}")
    return made


def main(argv=None):
    ap = argparse.ArgumentParser(description="MRRH Poland visualization runner")
    ap.add_argument("runs", nargs="+", help="one or more run.pkl (or run dir) paths")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--gpkg", default=None, help="commune GeoPackage (auto-inferred if omitted)")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--format", dest="fmt", default="png", help="png|pdf|svg")
    tr = ap.add_mutually_exclusive_group()
    tr.add_argument("--transparent", dest="transparent", action="store_true", default=True)
    tr.add_argument("--no-transparent", dest="transparent", action="store_false")
    ut = ap.add_mutually_exclusive_group()
    ut.add_argument("--usetex", dest="usetex", action="store_true", default=None)
    ut.add_argument("--no-usetex", dest="usetex", action="store_false")
    args = ap.parse_args(argv)

    eff = style.use_style(usetex=args.usetex)
    print(f"style: usetex={'on' if eff else 'off (fallback)'}, dpi={args.dpi}, "
          f"format={args.fmt}, transparent={args.transparent}")

    run_paths = [Path(r) for r in args.runs]
    runs = [RunResult.load(p) for p in run_paths]
    runs_sorted = sorted(runs, key=lambda r: r.year)
    gpkg = _infer_gpkg(run_paths[0], args.gpkg)
    if gpkg is None:
        print("  (no commune GeoPackage found; maps will be skipped — pass --gpkg)")

    all_made = []
    for run, rp in zip(runs, run_paths):
        base = Path(args.outdir) if (args.outdir and len(runs) == 1) else (
            rp if rp.is_dir() else rp.parent)
        print(f"[{run.run_id}] figures -> {base}")
        all_made += visualize_single(run, gpkg, base / "figures", base / "tables",
                                     dpi=args.dpi, transparent=args.transparent, fmt=args.fmt)

    if len(runs) >= 2:
        a, b = runs_sorted[0], runs_sorted[-1]
        cmp_dir = Path(args.outdir) if args.outdir else (
            run_paths[0].resolve().parents[1] / f"compare__{a.run_id}__{b.run_id}")
        print(f"[comparison {a.year} vs {b.year}] -> {cmp_dir}")
        all_made += visualize_comparison(runs_sorted, gpkg, cmp_dir,
                                         dpi=args.dpi, transparent=args.transparent, fmt=args.fmt)

    print(f"\nDone. {len(all_made)} artefacts written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
