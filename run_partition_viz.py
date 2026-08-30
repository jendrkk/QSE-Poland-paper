#!/usr/bin/env python3
"""
run_partition_viz.py — figures for the historical-partition story (Story 1).

Parallel to run_viz.py, for the partition analysis. Loads one or more solved
runs, attaches the P/R/A label from communes_2021_partitions.gpkg, recomputes the
partition gaps (three control specs) and the border-imposition / gap-removal
counterfactuals, and draws the figure set in the exact house style (viz.style),
with every legend BELOW the map/axes and clean (unfragmented) partition seams.

Maps (per run, with partition seams overlaid + the raw partition gaps in the title):
  map_partitions                     P/R/A reference choropleth (once)
  map_logCMA_<year>_seams            log commuter market access
  map_logA_<year>_seams              log productivity  (Topic-11 (a) analogue)
  map_logb_<year>_seams              log amenity       (Topic-11 (b) analogue)
  map_realv_<year>_seams             log real income
Cross-year / counterfactual maps:
  map_relaccess_<a>_<b>_seams        demeaned relative access gain (clean OSM pair)
  map_border_rhat_<year>_seams       residence reallocation under the seam
  map_removal_lhat_<year>_seams      employment reallocation, equalise A_n means (Topic-11 (c))
Non-map figures:
  fig_gaps_<year>, fig_warsaw_flip_<year>, fig_cma_vs_warsaw_<year>,
  fig_box_cma_resid_<year>, fig_border_welfare, fig_gap_trajectory_CMA

Usage
-----
  python run_partition_viz.py runs/2011_garmin_baseline runs/2021_osm_baseline \
      runs/2026_osm_baseline
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from qse_poland_paper.result import RunResult
from qse_poland_paper import partitions as P
from qse_poland_paper.viz import style, maps, partition_figs as pf


def _infer_gpkg(run_path, override):
    if override:
        return Path(override)
    for up in (1, 2, 3):
        cand = (run_path.resolve().parents[up]
                / "data/processed/shapefiles/communes_2021_partitions.gpkg")
        if cand.exists():
            return cand
    return None


def _gaps_struct(run, part, gpkg):
    specs = {"raw": [], "ttw": ["log_tt_warsaw"], "woj_ttw": ["woj", "log_tt_warsaw"]}
    out = {}
    for s, ctrl in specs.items():
        g = P.partition_gaps(run, part, objects=["A_n", "b_n", "CMA", "real_v"],
                             weight="R_n", controls=ctrl, gpkg_path=gpkg)
        out[s] = {k: {"P": list(o["gaps_vs_base"]["P"]),
                      "A": list(o["gaps_vs_base"]["A"]), "means": o["means"]}
                  for k, o in g["objects"].items()}
    return out


def _panel_row(run, part):
    return pd.DataFrame(dict(
        teryt=[str(c).zfill(7) for c in run.codes], year=run.year, partition=part,
        logCMA=np.log(np.asarray(run.calibrated["CMA"], float)),
        tt_warsaw=P.warsaw_traveltime(run)))


def main(argv=None):
    ap = argparse.ArgumentParser(description="MRRH Poland partition figures")
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--gpkg", default=None)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--commute-costs", nargs="*", type=float, default=[1.5, 2.0])
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--format", dest="fmt", default="png")
    ap.add_argument("--no-usetex", dest="usetex", action="store_false", default=None)
    args = ap.parse_args(argv)

    style.use_style(usetex=args.usetex)
    run_paths = [Path(r) for r in args.runs]
    runs = sorted((RunResult.load(p) for p in run_paths), key=lambda r: r.year)
    gpkg = _infer_gpkg(run_paths[0], args.gpkg)
    if gpkg is None:
        raise SystemExit("no partition GeoPackage found; pass --gpkg")
    outdir = Path(args.outdir) if args.outdir else (
        run_paths[0].resolve().parents[1] / "runs" / "partition_figures")
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"gpkg={gpkg}\nout={outdir}")

    parts = {r.year: P.load_partition(gpkg, r.codes) for r in runs}
    codes0 = runs[0].codes
    part0 = parts[runs[0].year]
    # one clean seam overlay (the frame is identical across years); cached in maps
    seams = maps.dissolve_boundaries(gpkg, codes0, part0)
    panel = pd.concat([_panel_row(r, parts[r.year]) for r in runs], ignore_index=True)
    gaps_all = {str(r.year): _gaps_struct(r, parts[r.year], gpkg) for r in runs}
    sv = dict(dpi=args.dpi, transparent=True, fmt=args.fmt)
    fkw = dict(dpi=args.dpi, transparent=True)          # figures.* take no fmt

    def gap_note(y, obj):
        g = gaps_all[str(y)]["raw"][obj]
        return rf"  (P$-$R $={g['P'][0]:+.2f}$, A$-$R $={g['A'][0]:+.2f}$)"

    maps.categorical(gpkg, codes0, part0, colors=pf.PART_COLOR, order=pf.PART_ORDER,
                     labels=pf.PART_LABEL, legend_title="partition",
                     title="Historical partitions (P/R/A)",
                     outpath=outdir / "map_partitions", **sv)

    LEVELS = [("CMA", "CMA", r"$\log$ CMA$_n$"),
              ("A_n", "A_n", r"$\log A_n$ (productivity)"),
              ("b_n", "b_n", r"$\log b_n$ (amenity)"),
              ("real_v", "real_v", r"$\log v_n/$CPI (real income)")]
    for r in runs:
        y = r.year; part = parts[y]
        df = panel[panel.year == y].reset_index(drop=True)
        for key, obj, lab in LEVELS:
            maps.choropleth(gpkg, r.codes,
                            np.log(np.asarray(r.calibrated[key], float)),
                            title=f"{lab}, {y}" + gap_note(y, obj),
                            label=lab.split(" (")[0], seams=seams,
                            outpath=outdir / f"map_{key}_{y}_seams", **sv)
        pf.gap_dotwhisker(gaps_all[str(y)]["raw"], objects=("A_n", "CMA", "real_v"),
                          title=f"Partition gaps in recovered fundamentals, {y} (raw)",
                          outpath=outdir / f"fig_gaps_{y}", **fkw)
        pf.warsaw_flip(gaps_all, y, objects=("CMA", "real_v"),
                       title=f"The Warsaw flip: partition gap as controls are added, {y}",
                       outpath=outdir / f"fig_warsaw_flip_{y}", **fkw)
        pf.cma_vs_warsaw(df, title=f"Market access vs distance to Warsaw, by partition ({y})",
                         outpath=outdir / f"fig_cma_vs_warsaw_{y}", **fkw)
        pf.partition_box(df, col="logCMA", resid_on="tt_warsaw",
                         ylabel=r"$\log$ CMA$_n$ (net of dist.\ to Warsaw)",
                         title=f"Conditional market-access distribution by partition, {y}",
                         outpath=outdir / f"fig_box_cma_resid_{y}", **fkw)

    bw = {}
    for r in runs:
        bw[str(r.year)] = {f"commute_{c}": P.impose_border(
            r, parts[r.year], commute_cost=c, channels="commute")["welfare_pct"]
            for c in args.commute_costs}
    pf.border_bars(bw, scenarios=[f"commute_{c}" for c in args.commute_costs],
                   scen_label={f"commute_{c}": rf"commute ${c}\times$" for c in args.commute_costs},
                   title="Welfare cost of re-imposing the seam (commuting channel)",
                   outpath=outdir / "fig_border_welfare", **fkw)

    if len(runs) >= 2:
        pf.gap_trajectory(gaps_all, obj="CMA", spec="woj_ttw",
                          title="Conditional market-access gap across network vintages",
                          outpath=outdir / "fig_gap_trajectory_CMA", **fkw)
        a, b = runs[-2], runs[-1]
        piv = panel.pivot_table(index="teryt", columns="year", values="logCMA")
        d = (piv[b.year] - piv[a.year]).reindex([str(c).zfill(7) for c in codes0]).values
        d = d - np.nanmean(d)
        maps.choropleth(gpkg, codes0, d, diverging=True,
                        title=rf"Relative market-access gain, {a.year}$\to${b.year} (demeaned)",
                        label=r"$\Delta\log$ CMA$_n$ $-$ mean", seams=seams,
                        outpath=outdir / f"map_relaccess_{a.year}_{b.year}_seams", **sv)
        obs = [r for r in runs if r.meta.get("flows_source") == "observed"]
        rr = obs[-1] if obs else runs[-1]
        prr = parts[rr.year]
        bres = P.impose_border(rr, prr, commute_cost=1.5, channels="commute")
        maps.choropleth(gpkg, rr.codes, np.log(bres["r"]), diverging=True,
                        title=rf"Residence reallocation $\log\hat r_n$, partition border ({rr.year})",
                        label=r"$\log\hat r_n$", seams=seams,
                        outpath=outdir / f"map_border_rhat_{rr.year}_seams", **sv)
        rres = P.remove_gap(rr, prr, target="A_n")
        maps.choropleth(gpkg, rr.codes, np.log(rres["l"]), diverging=True,
                        title=rf"Employment reallocation $\log\hat\ell_n$, equalise $A_n$ means ({rr.year})",
                        label=r"$\log\hat\ell_n$", seams=seams,
                        outpath=outdir / f"map_removal_lhat_{rr.year}_seams", **sv)
    print("done ->", outdir)


if __name__ == "__main__":
    raise SystemExit(main())
