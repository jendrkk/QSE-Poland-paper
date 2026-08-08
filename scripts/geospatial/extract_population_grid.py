#!/usr/bin/env python3
"""
extract_population_grid.py
==========================

Extract the JRC/ESTAT 2021 100 m census population grid for the Poland bounding
box and write it to a Parquet file.

Input raster
------------
GeoTIFF, EPSG:3035 (ETRS89-LAEA), 100 m resolution, dtype int16, nodata = -1.
Cell value = population count in the 100 m x 100 m cell.

What the script produces
------------------------
One row per 100 m cell whose *centre* falls inside the lon/lat bounding box of
Poland (data/raw/shapefiles/poland_bbox.geojson, EPSG:4326). Columns:

    population : int32    population count (nodata -1 is filled with 0)
    lon        : float64  cell-centre longitude (EPSG:4326)
    lat        : float64  cell-centre latitude  (EPSG:4326)

If run with ``--with-geometry`` the output is a GeoParquet with an additional
POINT geometry column (EPSG:4326) built from (lon, lat). By default the geometry
is omitted for speed and size; the lon/lat float columns fully identify each
cell centre.

Nodata policy
-------------
``--nodata-fill`` (default 0) is written wherever the raster is nodata (-1), so
every cell inside the bounding box is emitted, including genuine 0-population
cells and no-data cells (sea, outside census). Pass ``--drop-nodata`` to instead
drop nodata cells and keep only real census cells (0 population included).

Method (why it is fast)
-----------------------
* The lon/lat bbox is reprojected to EPSG:3035 with a densified boundary and
  snapped to the raster grid, giving a single read window (~58 M cells here).
* The window is split into horizontal row-tiles processed in parallel across all
  CPU cores. Each worker reads its tile, fills nodata, computes cell-centre
  coordinates in 3035, reprojects them to 4326 (vectorised pyproj), filters by
  the lon/lat bbox, and streams its result to a temporary Parquet part.
* Parts are then concatenated into the final file (streaming for the lon/lat
  output; in-memory assembly when geometry is requested).

Usage
-----
    python extract_population_grid.py                      # all defaults
    python extract_population_grid.py --workers 32 --with-geometry
    python extract_population_grid.py --help

All paths default to the QSE_Poland_paper repository layout.
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import os
import shutil
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import rasterio
from pyproj import Transformer
from rasterio.warp import transform_bounds
from rasterio.windows import Window, from_bounds

try:
    from tqdm import tqdm
except ImportError:  # graceful fallback if tqdm is not installed
    def tqdm(iterable=None, total=None, **_):
        return iterable if iterable is not None else range(total or 0)


# --------------------------------------------------------------------------- #
# Repository-relative default paths
# --------------------------------------------------------------------------- #
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RASTER = REPO_ROOT / "data" / "raw" / "pop" / \
    "JRC-ESTAT_Census_Population_2021_100m_rev0726.tif"
DEFAULT_BBOX = REPO_ROOT / "data" / "raw" / "shapefiles" / "poland_bbox.geojson"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "raw" / "pop" / \
    "poland_bbox_pop_100m.parquet"

LOGGER = logging.getLogger("extract_population_grid")


# --------------------------------------------------------------------------- #
# Worker configuration (picklable, sent to each process)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TileJob:
    raster_path: str
    part_path: str
    row_off: int          # global raster row offset of the tile
    col_off: int          # global raster col offset of the window
    n_rows: int           # tile height in pixels
    n_cols: int           # window width in pixels
    left: float           # raster transform: x of upper-left corner
    top: float            # raster transform: y of upper-left corner
    res: float            # pixel size (metres)
    src_crs: str
    dst_crs: str
    nodata: float
    nodata_fill: int
    drop_nodata: bool
    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float
    lonlat_dtype: str
    compression: str


def _process_tile(job: TileJob) -> tuple[str, int]:
    """Read one row-tile, reproject centres, filter, write a Parquet part.

    Returns (part_path, n_rows_written). Runs in a worker process.
    """
    with rasterio.open(job.raster_path) as ds:
        arr = ds.read(
            1,
            window=Window(job.col_off, job.row_off, job.n_cols, job.n_rows),
        )

    # Cell-centre coordinates in the source CRS (EPSG:3035).
    # x depends on column, y depends on row; build with broadcasting.
    cols = np.arange(job.n_cols, dtype=np.float64)
    rows = np.arange(job.n_rows, dtype=np.float64)
    x_centres = job.left + (job.col_off + cols + 0.5) * job.res            # (n_cols,)
    y_centres = job.top - (job.row_off + rows + 0.5) * job.res             # (n_rows,)

    xx = np.broadcast_to(x_centres, (job.n_rows, job.n_cols)).ravel()
    yy = np.repeat(y_centres, job.n_cols)
    pop = arr.ravel()

    nodata_mask = pop == job.nodata
    if job.drop_nodata:
        keep = ~nodata_mask
        xx = xx[keep]
        yy = yy[keep]
        pop = pop[keep]
    else:
        pop = pop.astype(np.int32, copy=True)
        pop[nodata_mask] = job.nodata_fill

    if xx.size == 0:
        # still write an empty, correctly-typed part for uniform concatenation
        table = _empty_table(job.lonlat_dtype)
        pq.write_table(table, job.part_path, compression=job.compression)
        return job.part_path, 0

    # Reproject cell centres 3035 -> 4326 (vectorised).
    transformer = Transformer.from_crs(job.src_crs, job.dst_crs, always_xy=True)
    lon, lat = transformer.transform(xx, yy)

    # Filter to the lon/lat bounding box (cell centre inside the rectangle).
    in_bbox = (
        (lon >= job.lon_min) & (lon <= job.lon_max) &
        (lat >= job.lat_min) & (lat <= job.lat_max)
    )
    lon = lon[in_bbox]
    lat = lat[in_bbox]
    pop = pop[in_bbox].astype(np.int32, copy=False)

    lonlat_np = np.float32 if job.lonlat_dtype == "float32" else np.float64
    table = pa.table(
        {
            "population": pa.array(pop, type=pa.int32()),
            "lon": pa.array(lon.astype(lonlat_np, copy=False)),
            "lat": pa.array(lat.astype(lonlat_np, copy=False)),
        }
    )
    pq.write_table(table, job.part_path, compression=job.compression)
    return job.part_path, int(pop.size)


def _empty_table(lonlat_dtype: str) -> pa.Table:
    lon_type = pa.float32() if lonlat_dtype == "float32" else pa.float64()
    return pa.table(
        {
            "population": pa.array([], type=pa.int32()),
            "lon": pa.array([], type=lon_type),
            "lat": pa.array([], type=lon_type),
        }
    )


# --------------------------------------------------------------------------- #
# Combination of parts
# --------------------------------------------------------------------------- #
def _combine_lonlat(parts: list[str], output: Path, compression: str) -> None:
    """Stream all parts into a single Parquet file with bounded memory."""
    writer = None
    try:
        for part in parts:
            pf = pq.ParquetFile(part)
            for batch in pf.iter_batches(batch_size=1_000_000):
                if writer is None:
                    writer = pq.ParquetWriter(
                        output, batch.schema, compression=compression
                    )
                writer.write_batch(batch)
    finally:
        if writer is not None:
            writer.close()


def _combine_with_geometry(parts: list[str], output: Path, compression: str) -> None:
    """Assemble a GeoParquet with a POINT geometry column (EPSG:4326)."""
    import geopandas as gpd  # imported lazily; only needed for this path

    tables = [pq.read_table(p) for p in parts]
    table = pa.concat_tables(tables) if len(tables) > 1 else tables[0]
    df = table.to_pandas()
    del tables, table

    geometry = gpd.points_from_xy(df["lon"].values, df["lat"].values, crs="EPSG:4326")
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    del df
    gdf.to_parquet(output, compression=compression, index=False)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def _read_bbox_lonlat(bbox_path: Path) -> tuple[float, float, float, float]:
    """Return (lon_min, lat_min, lon_max, lat_max) of the bbox geojson.

    Uses the total bounds of all features. The geojson is assumed EPSG:4326
    (WGS84 lon/lat), the standard for OSM/overpass exports.
    """
    import json

    with open(bbox_path) as fh:
        gj = json.load(fh)

    xs: list[float] = []
    ys: list[float] = []

    def _walk(coords):
        if isinstance(coords[0], (int, float)):
            xs.append(coords[0])
            ys.append(coords[1])
        else:
            for c in coords:
                _walk(c)

    feats = gj.get("features", [gj]) if gj.get("type") == "FeatureCollection" else [gj]
    for feat in feats:
        geom = feat.get("geometry", feat)
        _walk(geom["coordinates"])

    return min(xs), min(ys), max(xs), max(ys)


def _plan_window(ds, bbox_lonlat, densify_pts):
    """Compute the raster read window covering the lon/lat bbox."""
    lon_min, lat_min, lon_max, lat_max = bbox_lonlat
    b3035 = transform_bounds(
        "EPSG:4326", ds.crs, lon_min, lat_min, lon_max, lat_max,
        densify_pts=densify_pts,
    )
    win = from_bounds(*b3035, transform=ds.transform).round_offsets().round_lengths()

    # Clip window to the raster extent.
    col_off = max(0, int(win.col_off))
    row_off = max(0, int(win.row_off))
    col_end = min(ds.width, int(win.col_off) + int(win.width))
    row_end = min(ds.height, int(win.row_off) + int(win.height))
    return Window(col_off, row_off, col_end - col_off, row_end - row_off)


def _build_jobs(args, ds, window, tmp_dir):
    transform = ds.transform
    left = transform.c
    top = transform.f
    res = transform.a

    lon_min, lat_min, lon_max, lat_max = _read_bbox_lonlat(Path(args.bbox))
    nodata = ds.nodata if ds.nodata is not None else -1.0

    n_rows_total = int(window.height)
    n_cols = int(window.width)

    # Choose tile height so we get roughly workers * tiles_per_worker tiles.
    if args.tile_rows > 0:
        tile_rows = args.tile_rows
    else:
        target_tiles = max(args.workers * args.tiles_per_worker, 1)
        tile_rows = max(args.min_tile_rows, -(-n_rows_total // target_tiles))

    jobs: list[TileJob] = []
    idx = 0
    for r0 in range(0, n_rows_total, tile_rows):
        h = min(tile_rows, n_rows_total - r0)
        part_path = str(Path(tmp_dir) / f"part_{idx:05d}.parquet")
        jobs.append(
            TileJob(
                raster_path=str(args.raster),
                part_path=part_path,
                row_off=int(window.row_off) + r0,
                col_off=int(window.col_off),
                n_rows=h,
                n_cols=n_cols,
                left=left,
                top=top,
                res=res,
                src_crs=str(ds.crs),
                dst_crs="EPSG:4326",
                nodata=nodata,
                nodata_fill=args.nodata_fill,
                drop_nodata=args.drop_nodata,
                lon_min=lon_min,
                lon_max=lon_max,
                lat_min=lat_min,
                lat_max=lat_max,
                lonlat_dtype=args.lonlat_dtype,
                compression=args.compression,
            )
        )
        idx += 1
    return jobs, (lon_min, lat_min, lon_max, lat_max)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Extract JRC 100 m population grid for the Poland bbox to Parquet.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--raster", type=Path, default=DEFAULT_RASTER,
                   help="Input GeoTIFF population grid.")
    p.add_argument("--bbox", type=Path, default=DEFAULT_BBOX,
                   help="Bounding-box geojson (EPSG:4326).")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                   help="Output Parquet path.")
    p.add_argument("--workers", type=int, default=os.cpu_count(),
                   help="Number of worker processes.")
    p.add_argument("--tile-rows", type=int, default=0,
                   help="Rows per tile. 0 = auto from --tiles-per-worker.")
    p.add_argument("--tiles-per-worker", type=int, default=4,
                   help="Auto-tiling: target tiles per worker (load balancing).")
    p.add_argument("--min-tile-rows", type=int, default=64,
                   help="Auto-tiling: minimum rows per tile.")
    p.add_argument("--nodata-fill", type=int, default=0,
                   help="Value written where the raster is nodata (-1).")
    p.add_argument("--drop-nodata", action="store_true",
                   help="Drop nodata cells instead of filling them.")
    p.add_argument("--densify", type=int, default=101,
                   help="Points per edge when reprojecting bbox to 3035.")
    p.add_argument("--with-geometry", action="store_true",
                   help="Write a GeoParquet with a POINT geometry column.")
    p.add_argument("--lonlat-dtype", choices=["float32", "float64"],
                   default="float64", help="Precision of lon/lat columns.")
    p.add_argument("--compression", default="zstd",
                   help="Parquet compression codec.")
    p.add_argument("--tmp-dir", type=Path, default=None,
                   help="Directory for intermediate parts (default: system temp).")
    p.add_argument("--keep-parts", action="store_true",
                   help="Do not delete intermediate part files.")
    p.add_argument("--log-file", type=Path, default=None,
                   help="Optional log file (in addition to stderr).")
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

    if not args.raster.exists():
        LOGGER.error("Raster not found: %s", args.raster)
        return 1
    if not args.bbox.exists():
        LOGGER.error("Bbox geojson not found: %s", args.bbox)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    with rasterio.open(args.raster) as ds:
        LOGGER.info("Raster: %s", args.raster.name)
        LOGGER.info("  CRS=%s  size=%dx%d  res=%s  dtype=%s  nodata=%s",
                    ds.crs, ds.width, ds.height, ds.res, ds.dtypes[0], ds.nodata)
        window = _plan_window(ds, _read_bbox_lonlat(Path(args.bbox)), args.densify)
        LOGGER.info("  read window: col_off=%d row_off=%d w=%d h=%d (%.1f M cells)",
                    window.col_off, window.row_off, window.width, window.height,
                    window.width * window.height / 1e6)

        tmp_dir = Path(tempfile.mkdtemp(prefix="popgrid_", dir=args.tmp_dir))
        jobs, bbox = _build_jobs(args, ds, window, tmp_dir)

    LOGGER.info("bbox lon/lat: [%.6f, %.6f] x [%.6f, %.6f]",
                bbox[0], bbox[2], bbox[1], bbox[3])
    LOGGER.info("workers=%d  tiles=%d  tile_rows~%d  nodata=%s  geometry=%s",
                args.workers, len(jobs),
                jobs[0].n_rows if jobs else 0,
                "drop" if args.drop_nodata else f"fill={args.nodata_fill}",
                args.with_geometry)

    parts: list[str] = []
    total_rows = 0
    ctx = mp.get_context("spawn")
    try:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as ex:
            futures = {ex.submit(_process_tile, job): job for job in jobs}
            for fut in tqdm(as_completed(futures), total=len(futures),
                            desc="tiles", unit="tile"):
                part_path, n = fut.result()
                parts.append(part_path)
                total_rows += n

        LOGGER.info("Extracted %s cells across %d tiles in %.1fs.",
                    f"{total_rows:,}", len(parts), time.time() - t0)

        LOGGER.info("Combining parts -> %s", args.output)
        parts_sorted = sorted(parts)
        if args.with_geometry:
            _combine_with_geometry(parts_sorted, args.output, args.compression)
        else:
            _combine_lonlat(parts_sorted, args.output, args.compression)
    finally:
        if not args.keep_parts:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        else:
            LOGGER.info("Kept intermediate parts in %s", tmp_dir)

    size_mb = args.output.stat().st_size / 1e6
    LOGGER.info("Done: %s rows, %.1f MB, %.1fs total.",
                f"{total_rows:,}", size_mb, time.time() - t0)
    print(str(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
