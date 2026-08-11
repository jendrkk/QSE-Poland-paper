#!/usr/bin/env python3
"""
commune_centroids.py
====================

Compute geometric and population-weighted centroids of Polish communes
(gminy), attach grid-based population, and write a GeoPackage.

Input
-----
* Commune polygons: PRG ``A03_Granice_gmin.shp`` (reprojected to EPSG:3035).
* Population grid Parquet from ``extract_population_grid.py``
  (``population``, ``lon``, ``lat``).

What the script produces
------------------------
A GeoPackage with all original commune attribute columns plus:

    centroid            geometric centroid as WKT Point (EPSG:3035)
    weighted_centroid   population-weighted centroid as WKT Point (EPSG:3035)
    pop                 sum of population-grid cells inside the commune

Secondary point geometries are stored as WKT so they survive writing
(GPKG/pyogrio only persists the active polygon geometry column).

The GeoPackage is assembled in a local scratch directory then moved to the
output path. Writing SQLite/GPKG directly on NFS fails with
``Failed to start transaction`` (no POSIX fcntl locks).

Usage
-----
    python commune_centroids.py                      # all defaults
    python commune_centroids.py --workers 32
    python commune_centroids.py --help

All paths default to the QSE_Poland_paper repository layout.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from joblib import Parallel, delayed

try:
    from tqdm import tqdm
except ImportError:  # graceful fallback if tqdm is not installed
    def tqdm(iterable=None, total=None, **_):
        return iterable if iterable is not None else range(total or 0)


# --------------------------------------------------------------------------- #
# Repository-relative default paths
# --------------------------------------------------------------------------- #
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POP_GRID = (
    REPO_ROOT / "data" / "raw" / "pop" / "poland_bbox_pop_100m.parquet"
)
DEFAULT_COMMUNES = (
    REPO_ROOT
    / "data"
    / "raw"
    / "shapefiles"
    / "PRG_jednostki_administracyjne_2021"
    / "A03_Granice_gmin.shp"
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "data" / "processed" / "shapefiles" / "communes_2021.gpkg"
)
DEFAULT_ID_COL = "JPT_KOD_JE"

LOGGER = logging.getLogger("commune_centroids")


# --------------------------------------------------------------------------- #
# Core steps
# --------------------------------------------------------------------------- #
def filter_pop_grid_by_communes(
    pop_grid_gdf: gpd.GeoDataFrame,
    communes_border,
    workers: int,
) -> gpd.GeoDataFrame:
    """Keep pop>0 cells whose centres lie inside the national communes union."""
    n0 = len(pop_grid_gdf)
    pop_grid_gdf = pop_grid_gdf.loc[pop_grid_gdf["population"] > 0].copy()
    LOGGER.info(
        "Dropped zero-pop cells: %s -> %s",
        f"{n0:,}",
        f"{len(pop_grid_gdf):,}",
    )

    minx, miny, maxx, maxy = communes_border.bounds
    pop_grid_gdf = pop_grid_gdf.cx[minx:maxx, miny:maxy]
    LOGGER.info("After bbox prefilter: %s cells", f"{len(pop_grid_gdf):,}")

    border_wkb = shapely.to_wkb(communes_border)
    coords = np.column_stack(
        (pop_grid_gdf.geometry.x.to_numpy(), pop_grid_gdf.geometry.y.to_numpy())
    )

    def _contains_chunk(xy: np.ndarray, border_wkb: bytes) -> np.ndarray:
        border = shapely.from_wkb(border_wkb)
        shapely.prepare(border)
        return shapely.contains(border, shapely.points(xy[:, 0], xy[:, 1]))

    n_jobs = workers if workers > 0 else (os.cpu_count() or 4)
    n_chunks = max(n_jobs, 4) * 4
    chunks = [c for c in np.array_split(coords, n_chunks) if len(c)]
    LOGGER.info(
        "Point-in-polygon: workers=%d  chunks=%d  cells=%s",
        n_jobs,
        len(chunks),
        f"{len(coords):,}",
    )

    t0 = time.time()
    mask_parts = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_contains_chunk)(chunk, border_wkb)
        for chunk in tqdm(chunks, desc="pip chunks", unit="chunk")
    )
    mask = np.concatenate(mask_parts)
    pop_grid_gdf = pop_grid_gdf.iloc[mask].copy()
    LOGGER.info(
        "Inside communes: %s cells  pop_sum=%s  (%.1fs)",
        f"{len(pop_grid_gdf):,}",
        f"{int(pop_grid_gdf['population'].sum()):,}",
        time.time() - t0,
    )
    return pop_grid_gdf


def compute_weighted_centroid(
    pop_grid_gdf: gpd.GeoDataFrame,
    communes: gpd.GeoDataFrame,
    id_col: str,
) -> gpd.GeoDataFrame:
    """Population-weighted centroids of each commune.

    Formula: (sum(pop * x) / sum(pop), sum(pop * y) / sum(pop)).
    Zero-pop cells contribute nothing and should already be omitted.

    Returns a GeoDataFrame with ``id_col``, ``pop``, and point ``geometry``.
    Communes with no intersecting pop cells fall back to the geometric
    centroid and ``pop == 0``.
    """
    LOGGER.info("Spatial join of %s pop cells to %d communes...",
                f"{len(pop_grid_gdf):,}", len(communes))
    t0 = time.time()
    joined = gpd.sjoin(
        pop_grid_gdf[["population", "geometry"]],
        communes[[id_col, "geometry"]],
        how="inner",
        predicate="within",
    )
    LOGGER.info(
        "sjoin matched %s cells in %.1fs",
        f"{len(joined):,}",
        time.time() - t0,
    )

    x = joined.geometry.x.to_numpy()
    y = joined.geometry.y.to_numpy()
    pop = joined["population"].to_numpy(dtype=np.float64)

    joined = joined[[id_col]].copy()
    joined["wx"] = pop * x
    joined["wy"] = pop * y
    joined["population"] = pop

    agg = joined.groupby(id_col, sort=False).sum(numeric_only=True)
    agg["x_w"] = agg["wx"] / agg["population"]
    agg["y_w"] = agg["wy"] / agg["population"]

    centroids = gpd.GeoDataFrame(
        {
            id_col: agg.index,
            "pop": agg["population"].to_numpy(),
        },
        geometry=gpd.points_from_xy(agg["x_w"], agg["y_w"]),
        crs=communes.crs,
    )

    out = pd.DataFrame(
        {
            id_col: centroids[id_col],
            "pop": centroids["pop"],
            "geometry": centroids.geometry.values,
        }
    )
    out = communes[[id_col]].merge(out, on=id_col, how="left")
    out = gpd.GeoDataFrame(out, geometry="geometry", crs=communes.crs)

    missing = out.geometry.isna()
    n_missing = int(missing.sum())
    if n_missing:
        LOGGER.warning(
            "%d communes had no pop cells; falling back to geometric centroid",
            n_missing,
        )
        geom_cent = communes.set_index(id_col).geometry.centroid
        out.loc[missing, "geometry"] = out.loc[missing, id_col].map(geom_cent).values
        out.loc[missing, "pop"] = 0

    return out


def write_gpkg_nfs_safe(
    gdf: gpd.GeoDataFrame,
    output: Path,
    *,
    layer: str = "communes",
    tmp_dir: Path | None = None,
) -> None:
    """Write a GeoPackage via local scratch, then move to ``output``.

    GPKG is SQLite and needs POSIX fcntl locks that NFS typically lacks,
    which surfaces as ``Failed to start transaction``.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    scratch = Path(
        tempfile.mkdtemp(
            prefix="commune_centroids_",
            dir=str(tmp_dir) if tmp_dir is not None else None,
        )
    )
    work_gpkg = scratch / output.name
    LOGGER.info("Assembling GPKG in local scratch %s", scratch)
    try:
        gdf.to_file(work_gpkg, driver="GPKG", layer=layer)
        if output.exists():
            output.unlink()
        shutil.move(str(work_gpkg), str(output))
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute commune centroids and population from the 100 m grid."
    )
    p.add_argument(
        "--pop-grid",
        type=Path,
        default=DEFAULT_POP_GRID,
        help="Input population-grid Parquet (lon/lat/population).",
    )
    p.add_argument(
        "--communes",
        type=Path,
        default=DEFAULT_COMMUNES,
        help="Input commune polygons shapefile.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output GeoPackage path.",
    )
    p.add_argument(
        "--id-col",
        default=DEFAULT_ID_COL,
        help="Commune ID column name.",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 4,
        help="Number of joblib workers for point-in-polygon.",
    )
    p.add_argument(
        "--tmp-dir",
        type=Path,
        default=None,
        help="LOCAL scratch dir for building the GeoPackage before moving "
             "to --output (needed when --output is on NFS). Default: system temp.",
    )
    p.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Optional log file (in addition to stderr).",
    )
    return p.parse_args(argv)


def setup_logging(log_file: Path | None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
    )


def main(argv=None) -> int:
    args = parse_args(argv)
    setup_logging(args.log_file)

    if not args.pop_grid.exists():
        LOGGER.error("Population grid not found: %s", args.pop_grid)
        return 1
    if not args.communes.exists():
        LOGGER.error("Communes shapefile not found: %s", args.communes)
        return 1

    t0 = time.time()

    LOGGER.info("Loading pop grid: %s", args.pop_grid)
    pop_grid = pd.read_parquet(args.pop_grid)
    LOGGER.info("  rows=%s  columns=%s", f"{len(pop_grid):,}", list(pop_grid.columns))
    pop_grid_gdf = gpd.GeoDataFrame(
        pop_grid,
        geometry=gpd.points_from_xy(pop_grid.lon, pop_grid.lat),
        crs="EPSG:4326",
    )
    del pop_grid
    LOGGER.info("Reprojecting pop grid -> EPSG:3035")
    pop_grid_gdf = pop_grid_gdf.to_crs("EPSG:3035")

    LOGGER.info("Loading communes: %s", args.communes)
    communes = gpd.read_file(args.communes)
    LOGGER.info("  features=%d  CRS=%s", len(communes), communes.crs)
    communes = communes.to_crs("EPSG:3035")
    communes_border = communes.geometry.union_all()
    LOGGER.info("Built national communes union (EPSG:3035)")

    pop_grid_gdf = filter_pop_grid_by_communes(
        pop_grid_gdf, communes_border, workers=args.workers
    )

    LOGGER.info("Computing geometric centroids")
    communes["centroid"] = communes.geometry.centroid

    weighted = compute_weighted_centroid(pop_grid_gdf, communes, args.id_col)
    communes = communes.merge(
        pd.DataFrame(
            {
                args.id_col: weighted[args.id_col],
                "pop": weighted["pop"].astype(np.int64),
                "weighted_centroid": weighted.geometry.values,
            }
        ),
        on=args.id_col,
        how="left",
    )

    # GPKG only writes the active geometry column; persist points as WKT.
    out = communes.copy()
    out["centroid"] = gpd.GeoSeries(out["centroid"], crs=out.crs).to_wkt()
    out["weighted_centroid"] = gpd.GeoSeries(
        out["weighted_centroid"], crs=out.crs
    ).to_wkt()

    LOGGER.info("Writing %d communes -> %s", len(out), args.output)
    write_gpkg_nfs_safe(out, args.output, tmp_dir=args.tmp_dir)

    size_mb = args.output.stat().st_size / 1e6
    LOGGER.info(
        "Done: %d communes, pop_sum=%s, %.1f MB, %.1fs total.",
        len(out),
        f"{int(out['pop'].sum()):,}",
        size_mb,
        time.time() - t0,
    )
    print(str(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
