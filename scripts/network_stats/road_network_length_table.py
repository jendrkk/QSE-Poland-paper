#!/usr/bin/env python3
"""
road_network_length_table.py
=============================

Length of the Polish road network by class -- A (motorway/autostrada),
S (trunk/droga ekspresowa), primary (national road, droga krajowa), and the
total routable network -- for the three MRRH model vintages (2011 Garmin,
2021 OSM, 2026 OSM), with year-over-year percentage changes. Prints a table
to the terminal and writes a ready-to-use LaTeX table for the paper's
appendix.

Data source
-----------
This reads the ``class_km`` / ``total_km`` fields already computed by
``scripts/tt_matrix/road_travel_time_matrix.py`` and stored in its
``.meta.json`` sidecars under ``data/processed/tt_matrix/``. Those numbers
are measured directly off the raw network files for each vintage:

    2011  data/raw/gpmap/out/poland_roads_2011_full.gpkg   (Garmin GPMapa TOPO)
    2021  data/raw/osm_pbf/poland_roads_2021-04-01_optimal.osm.pbf
    2026  data/raw/osm_pbf/poland_roads_2026-08-01_optimal.osm.pbf

using that script's classification logic: OSM ``highway=*`` tags directly
for 2021/2026, and for the 2011 Garmin GPKG, its GPMapa TOPO classes
remapped to their OSM-equivalent Polish-usage names with the A#/S# road
number promotion that recovers the motorway/trunk split inside Garmin's
mixed "motorway" bucket (see the docstring of road_travel_time_matrix.py,
section "GPKG parsing", for the full rationale). Re-deriving these lengths
independently would just reimplement that same, already-validated pipeline
on network files up to several GB in size, so this script reuses its output
directly rather than re-parsing the .pbf/.gpkg files.

If a required .meta.json is missing, run road_travel_time_matrix.py for
that year first (see scripts/tt_matrix/README.md).

Usage
-----
    python3 road_network_length_table.py

Output
------
    - a formatted table on stdout
    - road_network_length_table.tex, written next to this script
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]          # scripts/network_stats -> scripts -> repo root
TT_MATRIX_DIR = REPO_ROOT / "data" / "processed" / "tt_matrix"

# (year label, meta.json path, network source description)
YEAR_SOURCES = [
    ("2011", TT_MATRIX_DIR / "ttm_road_garmin_2011_final.meta.json", "Garmin GPMapa TOPO"),
    ("2021", TT_MATRIX_DIR / "ttm_road_osm_2021_final.meta.json", "OpenStreetMap"),
    ("2026", TT_MATRIX_DIR / "ttm_road_osm_2026_final.meta.json", "OpenStreetMap"),
]

# Row label -> OSM highway=* tags summed into it. ``None`` -> use total_km directly.
ROAD_CLASSES = [
    ("A roads -- motorway (autostrada)", ["motorway", "motorway_link"]),
    ("S roads -- trunk (droga ekspresowa)", ["trunk", "trunk_link"]),
    ("Primary -- national road (DK)", ["primary", "primary_link"]),
    ("Total network (all drivable classes)", None),
]


def load_year(path: Path, label: str, source: str) -> dict:
    if not path.exists():
        sys.exit(
            f"Missing {label} network stats file:\n  {path}\n"
            "Run scripts/tt_matrix/road_travel_time_matrix.py for this year "
            "first (see scripts/tt_matrix/README.md)."
        )
    meta = json.loads(path.read_text())
    class_km = meta["class_km"]
    total_km = meta["stats"]["total_km"]
    values = {}
    for row_label, tags in ROAD_CLASSES:
        values[row_label] = total_km if tags is None else sum(class_km.get(t, 0.0) for t in tags)
    return {
        "label": label,
        "source": source,
        "network_file": meta.get("network", "?"),
        "verdict": meta.get("verdict", "?"),
        "values": values,
    }


def pct_change(a: float, b: float) -> float:
    return (b - a) / a * 100.0 if a else float("nan")


def fmt_km(x: float) -> str:
    return f"{x:,.1f}"


def fmt_pct(x: float, latex: bool) -> str:
    sign = "+" if x >= 0 else ""
    return f"{sign}{x:.1f}\\%" if latex else f"{sign}{x:.1f}%"


def main() -> None:
    years = [load_year(p, lbl, src) for lbl, p, src in YEAR_SOURCES]
    row_labels = [r[0] for r in ROAD_CLASSES]

    # ---------------------------------------------------------------- #
    # Terminal table
    # ---------------------------------------------------------------- #
    print()
    print("Polish road network length by class -- 2011 / 2021 / 2026")
    for y in years:
        print(f"  {y['label']}: {y['source']} ({Path(y['network_file']).name}, "
              f"verdict: {y['verdict']})")
    print()

    label_w = 38
    col_w = 14
    pct_w = 11
    header = (f"{'Road class':<{label_w}}"
              + "".join(f"{y['label'] + ' km':>{col_w}}" for y in years)
              + f"{'d11-21':>{pct_w}}{'d21-26':>{pct_w}}{'d11-26':>{pct_w}}")
    print(header)
    print("-" * len(header))
    for label in row_labels:
        v11, v21, v26 = (y["values"][label] for y in years)
        d1121, d2126, d1126 = pct_change(v11, v21), pct_change(v21, v26), pct_change(v11, v26)
        print(f"{label:<{label_w}}"
              f"{fmt_km(v11):>{col_w}}{fmt_km(v21):>{col_w}}{fmt_km(v26):>{col_w}}"
              f"{fmt_pct(d1121, False):>{pct_w}}{fmt_pct(d2126, False):>{pct_w}}"
              f"{fmt_pct(d1126, False):>{pct_w}}")
    print()

    # ---------------------------------------------------------------- #
    # LaTeX table
    # ---------------------------------------------------------------- #
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Length of the Polish road network by class, 2011--2026}",
        r"\label{tab:road_network_lengths}",
        r"\begin{tabular}{lrrrrrr}",
        r"\hline\hline",
        r"Road class & 2011 (km) & 2021 (km) & 2026 (km) & "
        r"$\Delta$11--21 & $\Delta$21--26 & $\Delta$11--26 \\",
        r"\hline",
    ]
    for label in row_labels:
        v11, v21, v26 = (y["values"][label] for y in years)
        d1121, d2126, d1126 = pct_change(v11, v21), pct_change(v21, v26), pct_change(v11, v26)
        lines.append(
            f"{label} & {fmt_km(v11)} & {fmt_km(v21)} & {fmt_km(v26)} & "
            f"{fmt_pct(d1121, True)} & {fmt_pct(d2126, True)} & {fmt_pct(d1126, True)} \\\\"
        )
    lines += [
        r"\hline\hline",
        r"\end{tabular}",
        r"\begin{minipage}{0.95\linewidth}",
        r"\vspace{4pt}",
        r"{\footnotesize Notes: A = OSM \texttt{highway=motorway} (Polish autostrada); "
        r"S = OSM \texttt{highway=trunk} (Polish droga ekspresowa); "
        r"Primary = OSM \texttt{highway=primary} (Polish droga krajowa, DK). "
        r"Ramp/link segments (\texttt{*\_link}) are included in their parent class. "
        r"Total network is the full routable graph (all drivable classes). "
        r"The 2011 network is derived from a Garmin GPMapa TOPO extract; 2021 and 2026 "
        r"are OpenStreetMap extracts, so the 2011$\to$2021 change conflates real "
        r"infrastructure growth with a measurement/coverage change in the underlying map "
        r"source. The 2021$\to$2026 change is the only like-for-like (OSM$\to$OSM) "
        r"comparison. Class lengths are taken from the validated network builds in "
        r"\texttt{scripts/tt\_matrix/road\_travel\_time\_matrix.py}.}",
        r"\end{minipage}",
        r"\end{table}",
    ]

    out_path = SCRIPT_DIR / "road_network_length_table.tex"
    out_path.write_text("\n".join(lines) + "\n")
    print(f"LaTeX table written to: {out_path}")


if __name__ == "__main__":
    main()
