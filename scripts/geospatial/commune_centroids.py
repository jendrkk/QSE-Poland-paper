#!/usr/bin/env python3
"""
commune_centroids.py
====================

Compute geometric and population-weighted centroids of Polish communes
(gminy), plus a ranked set of *population-cluster candidate points* that are
guaranteed to lie INSIDE each commune, attach grid-based population, and write
a GeoPackage.

Motivation for the candidates
------------------------------
For donut-shaped or strongly concave communes, both the geometric and the
population-weighted centroid can fall *outside* the polygon, which makes them
useless as origin/destination nodes for a travel-time matrix. To have a robust
interior representative for *every* commune, we additionally locate the centres
of the commune's largest population clusters and snap them inside the polygon.

Method
------
Working in EPSG:3035 (metric, locally isotropic), each commune's populated
100 m cells are clustered with a population-weighted k-means++ (self-contained
NumPy implementation, no scikit-learn dependency). Each cluster contributes one
candidate: its population-weighted barycenter. Candidates are ranked by cluster
population (descending). Insideness is guaranteed by a snap step -- if a
barycenter falls outside the polygon, it is replaced by the assigned populated
cell nearest to it (a cell is inside by construction of the ``within`` join).
Communes with no populated cells fall back to ``representative_point()`` (a
point shapely guarantees to lie inside the geometry).

The clustering is embarrassingly parallel across communes and is dispatched
over all workers with joblib/loky; inner BLAS threads are pinned to 1 to avoid
oversubscription.

Input
-----
* Commune polygons: PRG ``A03_Granice_gmin.shp`` (reprojected to EPSG:3035).
* Population grid Parquet from ``extract_population_grid.py``
  (``population``, ``lon``, ``lat``).

Output
------
A GeoPackage with two layers:

``communes`` (polygons) -- all original attribute columns plus:
    centroid            geometric centroid as WKT Point (EPSG:3035)
    weighted_centroid   population-weighted centroid as WKT Point (EPSG:3035)
    pop                 sum of population-grid cells inside the commune
    pop_anchor          rank-1 population-cluster candidate as WKT Point;
                        an interior-guaranteed representative node
    pop_candidates      all candidates as an ordered WKT MultiPoint
                        (rank 1..K, K = --n-candidates)
    n_pop_candidates    number of candidates found for the commune

``commune_candidates`` (points) -- long format, one row per candidate:
    <id_col>  cand_rank  cand_pop  inside_raw  geometry(Point EPSG:3035)
    ``inside_raw`` is 1 if the raw cluster barycenter was already inside the
    commune, 0 if it had to be snapped to the nearest populated cell.

Secondary point geometries in the ``communes`` layer are stored as WKT so they
survive writing (GPKG/pyogrio only persists the active polygon geometry
column). The candidate points are a separate layer with real Point geometry.

The GeoPackage is assembled in a local scratch directory then moved to the
output path. Writing SQLite/GPKG directly on NFS fails with
``Failed to start transaction`` (no POSIX fcntl locks).

Usage
-----
    python commune_centroids.py                      # all defaults
    python commune_centroids.py --workers 32 --n-candidates 5
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

try:  # joblib >= 1.3 exposes parallel_config; fall back to parallel_backend.
    from joblib import parallel_config as _parallel_config  # type: ignore
except Exception:  # pragma: no cover - older joblib
    _parallel_config = None
    from joblib import parallel_backend as _parallel_backend  # type: ignore

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
DEFAULT_N_CANDIDATES = 5
DEFAULT_KMEANS_RESTARTS = 8
DEFAULT_KMEANS_MAX_ITER = 200

CANDIDATE_LAYER = "commune_candidates"

LOGGER = logging.getLogger("commune_centroids")


# --------------------------------------------------------------------------- #
# Population-weighted k-means++ (self-contained, NumPy only)
# --------------------------------------------------------------------------- #
def _kpp_init(
    X: np.ndarray, w: np.ndarray, k: int, rng: np.random.Generator
) -> np.ndarray:
    """Weighted k-means++ seeding.

    First centre is drawn with probability proportional to weight; each further
    centre is drawn with probability proportional to weight times squared
    distance to the nearest chosen centre (the classic D^2 rule, made
    population-aware).
    """
    n = X.shape[0]
    centers = np.empty((k, X.shape[1]), dtype=np.float64)
    wsum = w.sum()
    i0 = rng.choice(n, p=w / wsum) if wsum > 0 else rng.integers(n)
    centers[0] = X[i0]
    closest = ((X - centers[0]) ** 2).sum(1)
    for j in range(1, k):
        probs = w * closest
        s = probs.sum()
        if s <= 0:  # all remaining points coincide with chosen centres
            centers[j] = X[rng.integers(n)]
        else:
            centers[j] = X[rng.choice(n, p=probs / s)]
        closest = np.minimum(closest, ((X - centers[j]) ** 2).sum(1))
    return centers


def _assign(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Nearest-centre assignment via the (x^2 - 2 x.c + c^2) expansion.

    Uses a BLAS matmul instead of an (n, k, 2) broadcast so that large urban
    communes (tens of thousands of cells) stay fast.
    """
    x2 = (X ** 2).sum(1)[:, None]
    c2 = (centers ** 2).sum(1)[None, :]
    d = x2 - 2.0 * (X @ centers.T) + c2
    return d.argmin(1)


def _weighted_kmeans(
    X: np.ndarray,
    w: np.ndarray,
    k: int,
    n_init: int,
    max_iter: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Weighted Lloyd's algorithm with k-means++ restarts.

    Returns ``(labels, centers)`` for the restart with the lowest
    population-weighted inertia. Empty clusters are re-seeded at the globally
    worst-fit point so that exactly ``k`` centres are always returned.
    """
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    best_inertia = np.inf
    best_labels: np.ndarray | None = None
    best_centers: np.ndarray | None = None

    for _ in range(n_init):
        centers = _kpp_init(X, w, k, rng)
        labels = _assign(X, centers)
        for _ in range(max_iter):
            new_centers = centers.copy()
            for j in range(k):
                m = labels == j
                if m.any():
                    ww = w[m]
                    new_centers[j] = (X[m] * ww[:, None]).sum(0) / ww.sum()
                else:
                    # Re-seed empty cluster at the point that fits its current
                    # centre worst (largest weighted squared distance).
                    d = w * ((X - centers[labels]) ** 2).sum(1)
                    new_centers[j] = X[int(d.argmax())]
            new_labels = _assign(X, new_centers)
            converged = np.array_equal(new_labels, labels) and np.allclose(
                new_centers, centers
            )
            centers, labels = new_centers, new_labels
            if converged:
                break

        inertia = float((w * ((X - centers[labels]) ** 2).sum(1)).sum())
        if inertia < best_inertia:
            best_inertia, best_labels, best_centers = inertia, labels, centers

    assert best_labels is not None and best_centers is not None
    return best_labels, best_centers


# --------------------------------------------------------------------------- #
# Per-commune candidate worker (runs in a joblib/loky subprocess)
# --------------------------------------------------------------------------- #
def _cluster_commune(
    cid,
    X: np.ndarray,
    w: np.ndarray,
    poly_wkb: bytes,
    k: int,
    n_init: int,
    max_iter: int,
    seed: int,
) -> list[tuple]:
    """Return population-cluster candidates for one commune.

    Output rows: ``(cid, rank, pop, x, y, inside_raw)`` ordered by cluster
    population (descending). Every returned (x, y) lies inside the commune:
    barycenters that fall outside a concave/donut polygon are snapped to the
    assigned populated cell nearest to them.
    """
    n = X.shape[0]
    if n == 0:  # handled by the caller's representative_point() fallback
        return []

    poly = shapely.from_wkb(poly_wkb)
    shapely.prepare(poly)

    n_distinct = len(np.unique(X, axis=0))
    k_eff = min(k, n_distinct)

    # Few points (or few distinct locations): each populated cell is its own
    # trivial cluster; take the k_eff heaviest cells directly. Cells are inside
    # by construction, so no snap/containment test is needed.
    if n <= k or k_eff <= 1:
        order = np.argsort(-w, kind="stable")[:k_eff]
        return [
            (cid, rank, float(w[i]), float(X[i, 0]), float(X[i, 1]), 1)
            for rank, i in enumerate(order, start=1)
        ]

    labels, centers = _weighted_kmeans(
        X, w, k_eff, n_init=n_init, max_iter=max_iter, seed=seed
    )
    pops = np.array([w[labels == j].sum() for j in range(k_eff)])
    order = np.argsort(-pops, kind="stable")

    rows: list[tuple] = []
    rank = 1
    for j in order:
        if pops[j] <= 0:
            continue
        cx, cy = float(centers[j, 0]), float(centers[j, 1])
        inside_raw = 1
        if not shapely.contains(poly, shapely.points(cx, cy)):
            m = labels == j
            Xm = X[m]
            d = (Xm[:, 0] - cx) ** 2 + (Xm[:, 1] - cy) ** 2
            snap = Xm[int(d.argmin())]
            cx, cy = float(snap[0]), float(snap[1])
            inside_raw = 0
        rows.append((cid, rank, float(pops[j]), cx, cy, inside_raw))
        rank += 1
    return rows


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


def spatial_join_cells(
    pop_grid_gdf: gpd.GeoDataFrame,
    communes: gpd.GeoDataFrame,
    id_col: str,
) -> pd.DataFrame:
    """Assign each populated cell to the commune that contains it.

    Returns a plain DataFrame ``[id_col, x, y, population]`` (coordinates as
    float64 NumPy arrays), shared by both the weighted-centroid and the
    candidate-clustering steps so the join is only paid for once.
    """
    LOGGER.info(
        "Spatial join of %s pop cells to %d communes...",
        f"{len(pop_grid_gdf):,}",
        len(communes),
    )
    t0 = time.time()
    joined = gpd.sjoin(
        pop_grid_gdf[["population", "geometry"]],
        communes[[id_col, "geometry"]],
        how="inner",
        predicate="within",
    )
    LOGGER.info("sjoin matched %s cells in %.1fs", f"{len(joined):,}", time.time() - t0)
    return pd.DataFrame(
        {
            id_col: joined[id_col].to_numpy(),
            "x": joined.geometry.x.to_numpy(np.float64),
            "y": joined.geometry.y.to_numpy(np.float64),
            "population": joined["population"].to_numpy(np.float64),
        }
    )


def weighted_centroids_from_join(
    joined: pd.DataFrame,
    communes: gpd.GeoDataFrame,
    id_col: str,
) -> gpd.GeoDataFrame:
    """Population-weighted centroid of each commune from the shared join.

    Formula: (sum(pop * x) / sum(pop), sum(pop * y) / sum(pop)). Communes with
    no intersecting pop cells fall back to the geometric centroid and pop == 0.
    """
    df = joined.copy()
    df["wx"] = df["population"] * df["x"]
    df["wy"] = df["population"] * df["y"]
    agg = df.groupby(id_col, sort=False)[["wx", "wy", "population"]].sum()
    agg["x_w"] = agg["wx"] / agg["population"]
    agg["y_w"] = agg["wy"] / agg["population"]

    centroids = pd.DataFrame(
        {
            id_col: agg.index,
            "pop": agg["population"].to_numpy(),
            "x_w": agg["x_w"].to_numpy(),
            "y_w": agg["y_w"].to_numpy(),
        }
    )
    out = communes[[id_col]].merge(centroids, on=id_col, how="left")
    out = gpd.GeoDataFrame(
        out,
        geometry=gpd.points_from_xy(out["x_w"], out["y_w"]),
        crs=communes.crs,
    ).drop(columns=["x_w", "y_w"])

    missing = out.geometry.is_empty | out.geometry.isna()
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


def compute_population_candidates(
    joined: pd.DataFrame,
    communes: gpd.GeoDataFrame,
    id_col: str,
    k: int,
    workers: int,
    n_init: int,
    max_iter: int,
    seed: int,
) -> gpd.GeoDataFrame:
    """Ranked, interior-guaranteed population-cluster candidates per commune.

    Clustering is dispatched per commune across all workers (loky). Communes
    absent from ``joined`` (no populated cells) get a single fallback candidate
    at ``representative_point()``. Returns a long-format point GeoDataFrame:
    ``[id_col, cand_rank, cand_pop, inside_raw, geometry]``.
    """
    poly_wkb = {
        cid: shapely.to_wkb(geom)
        for cid, geom in communes.set_index(id_col).geometry.items()
    }

    tasks = [
        (cid, sub[["x", "y"]].to_numpy(np.float64), sub["population"].to_numpy(np.float64))
        for cid, sub in joined.groupby(id_col, sort=False)
    ]
    n_jobs = workers if workers > 0 else (os.cpu_count() or 4)
    LOGGER.info(
        "Population clustering: %d communes  workers=%d  k=%d  restarts=%d",
        len(tasks),
        n_jobs,
        k,
        n_init,
    )

    t0 = time.time()
    gen = (
        delayed(_cluster_commune)(
            cid, X, w, poly_wkb[cid], k, n_init, max_iter, seed
        )
        for cid, X, w in tqdm(tasks, desc="cluster communes", unit="gmina")
    )
    # Pin inner BLAS/OpenMP threads to 1 so per-commune workers do not
    # oversubscribe the machine.
    if _parallel_config is not None:
        with _parallel_config(backend="loky", n_jobs=n_jobs, inner_max_num_threads=1):
            results = Parallel()(gen)
    else:  # pragma: no cover - older joblib
        with _parallel_backend("loky", n_jobs=n_jobs):
            results = Parallel(n_jobs=n_jobs)(gen)

    rows = [r for res in results for r in res]
    LOGGER.info(
        "Clustering produced %s candidate points for %d communes (%.1fs)",
        f"{len(rows):,}",
        len(tasks),
        time.time() - t0,
    )

    cand = gpd.GeoDataFrame(
        {
            id_col: [r[0] for r in rows],
            "cand_rank": np.array([r[1] for r in rows], dtype=np.int32),
            "cand_pop": np.array([r[2] for r in rows], dtype=np.int64),
            "inside_raw": np.array([r[5] for r in rows], dtype=np.int8),
        },
        geometry=gpd.points_from_xy(
            [r[3] for r in rows], [r[4] for r in rows]
        ),
        crs=communes.crs,
    )

    # Fallback for communes with no populated cells: representative_point().
    have = set(joined[id_col].unique())
    missing = communes.loc[~communes[id_col].isin(have)]
    if len(missing):
        LOGGER.warning(
            "%d communes have no populated cells; using representative_point() "
            "as their sole candidate",
            len(missing),
        )
        rep = missing.geometry.representative_point()
        fallback = gpd.GeoDataFrame(
            {
                id_col: missing[id_col].to_numpy(),
                "cand_rank": np.ones(len(missing), dtype=np.int32),
                "cand_pop": np.zeros(len(missing), dtype=np.int64),
                "inside_raw": np.ones(len(missing), dtype=np.int8),
            },
            geometry=rep.values,
            crs=communes.crs,
        )
        cand = pd.concat([cand, fallback], ignore_index=True)
        cand = gpd.GeoDataFrame(cand, geometry="geometry", crs=communes.crs)

    cand = cand.sort_values([id_col, "cand_rank"]).reset_index(drop=True)
    return cand


def write_gpkg_nfs_safe(
    gdf: gpd.GeoDataFrame,
    output: Path,
    *,
    layer: str = "communes",
    extra_layers: dict[str, gpd.GeoDataFrame] | None = None,
    tmp_dir: Path | None = None,
) -> None:
    """Write one or more layers to a GeoPackage via local scratch, then move.

    GPKG is SQLite and needs POSIX fcntl locks that NFS typically lacks, which
    surfaces as ``Failed to start transaction``; assembling on local scratch
    and moving the finished file avoids it.
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
        for name, extra in (extra_layers or {}).items():
            extra.to_file(work_gpkg, driver="GPKG", layer=name)
        if output.exists():
            output.unlink()
        shutil.move(str(work_gpkg), str(output))
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def summarise_candidates(
    cand: gpd.GeoDataFrame, id_col: str
) -> pd.DataFrame:
    """Collapse the long candidate layer into per-commune summary columns.

    Returns a DataFrame ``[id_col, pop_anchor, pop_candidates, n_pop_candidates]``
    where ``pop_anchor`` is the rank-1 candidate as WKT Point and
    ``pop_candidates`` is the ordered WKT MultiPoint of all candidates.
    """
    cand = cand.sort_values([id_col, "cand_rank"])
    recs = []
    for cid, sub in cand.groupby(id_col, sort=False):
        pts = list(sub.geometry.values)
        anchor = shapely.to_wkt(pts[0])
        multi = shapely.to_wkt(shapely.multipoints(pts))
        recs.append((cid, anchor, multi, len(pts)))
    return pd.DataFrame(
        recs, columns=[id_col, "pop_anchor", "pop_candidates", "n_pop_candidates"]
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute commune centroids, population, and population-"
        "cluster candidate points from the 100 m grid."
    )
    p.add_argument("--pop-grid", type=Path, default=DEFAULT_POP_GRID,
                   help="Input population-grid Parquet (lon/lat/population).")
    p.add_argument("--communes", type=Path, default=DEFAULT_COMMUNES,
                   help="Input commune polygons shapefile.")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                   help="Output GeoPackage path.")
    p.add_argument("--id-col", default=DEFAULT_ID_COL,
                   help="Commune ID column name.")
    p.add_argument("--workers", type=int, default=os.cpu_count() or 4,
                   help="Number of joblib workers (point-in-polygon and "
                        "per-commune clustering).")
    p.add_argument("--n-candidates", type=int, default=DEFAULT_N_CANDIDATES,
                   help="Max population-cluster candidates per commune (K).")
    p.add_argument("--kmeans-restarts", type=int, default=DEFAULT_KMEANS_RESTARTS,
                   help="k-means++ restarts per commune (best inertia kept).")
    p.add_argument("--kmeans-max-iter", type=int, default=DEFAULT_KMEANS_MAX_ITER,
                   help="Max Lloyd iterations per k-means restart.")
    p.add_argument("--seed", type=int, default=0,
                   help="Base RNG seed for deterministic clustering.")
    p.add_argument("--tmp-dir", type=Path, default=None,
                   help="LOCAL scratch dir for building the GeoPackage before "
                        "moving to --output (needed when --output is on NFS). "
                        "Default: system temp.")
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


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
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
    # PRG .shp/.dbf is UTF-8; without encoding=, pyogrio/fiona often Latin-1-mojibakes names.
    communes = gpd.read_file(args.communes, encoding="utf-8")
    LOGGER.info("  features=%d  CRS=%s", len(communes), communes.crs)
    communes = communes.to_crs("EPSG:3035")
    communes_border = communes.geometry.union_all()
    LOGGER.info("Built national communes union (EPSG:3035)")

    pop_grid_gdf = filter_pop_grid_by_communes(
        pop_grid_gdf, communes_border, workers=args.workers
    )

    LOGGER.info("Computing geometric centroids")
    communes["centroid"] = communes.geometry.centroid

    # Single shared join feeds both the weighted centroid and the clustering.
    joined = spatial_join_cells(pop_grid_gdf, communes, args.id_col)
    del pop_grid_gdf

    weighted = weighted_centroids_from_join(joined, communes, args.id_col)
    communes = communes.merge(
        pd.DataFrame(
            {
                args.id_col: weighted[args.id_col],
                "pop": weighted["pop"].fillna(0).astype(np.int64),
                "weighted_centroid": weighted.geometry.values,
            }
        ),
        on=args.id_col,
        how="left",
    )

    candidates = compute_population_candidates(
        joined,
        communes,
        args.id_col,
        k=args.n_candidates,
        workers=args.workers,
        n_init=args.kmeans_restarts,
        max_iter=args.kmeans_max_iter,
        seed=args.seed,
    )
    del joined

    summary = summarise_candidates(candidates, args.id_col)
    communes = communes.merge(summary, on=args.id_col, how="left")

    # GPKG only writes the active geometry column; persist points as WKT.
    out = communes.copy()
    out["centroid"] = gpd.GeoSeries(out["centroid"], crs=out.crs).to_wkt()
    out["weighted_centroid"] = gpd.GeoSeries(
        out["weighted_centroid"], crs=out.crs
    ).to_wkt()

    LOGGER.info(
        "Writing %d communes + %s candidate points -> %s",
        len(out),
        f"{len(candidates):,}",
        args.output,
    )
    write_gpkg_nfs_safe(
        out,
        args.output,
        layer="communes",
        extra_layers={CANDIDATE_LAYER: candidates},
        tmp_dir=args.tmp_dir,
    )

    size_mb = args.output.stat().st_size / 1e6
    LOGGER.info(
        "Done: %d communes, %s candidates (mean %.2f/commune), pop_sum=%s, "
        "%.1f MB, %.1fs total.",
        len(out),
        f"{len(candidates):,}",
        len(candidates) / max(len(out), 1),
        f"{int(out['pop'].sum()):,}",
        size_mb,
        time.time() - t0,
    )
    print(str(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
