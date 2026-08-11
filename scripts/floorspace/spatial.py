#!/usr/bin/env python3
"""
spatial.py
==========

Assign RCN transaction points to 2021 gminas and build gmina-level spatial
covariates.

The commune polygons (``communes_2021.gpkg``, EPSG:3035) are reprojected once
to EPSG:2180 (the RCN CRS) — reprojecting ~2,477 polygons is far cheaper than
reprojecting millions of points. Point-in-polygon assignment uses
``geopandas.sjoin`` (predicate ``within``) chunked across CPU cores.

A validation gate compares the geometry-derived powiat (first four digits of
the assigned gmina TERYT) against the powiat recorded in the RCN ``teryt``
field, logging and flagging the mismatch rate.

Public API
----------
load_communes(path) -> GeoDataFrame                (EPSG:2180, [id, pop, geometry])
assign_gmina(df, communes, workers) -> DataFrame   (adds gmina_teryt, teryt_ok)
gmina_land_covariate(land_df, communes, workers) -> DataFrame
"""

from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd

import geopandas as gpd
from joblib import Parallel, delayed

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(it=None, total=None, **_):
        return it if it is not None else range(total or 0)

from config import (
    COMMUNES_ID_COL,
    COMMUNES_POP_COL,
    CRS_WORK,
)

LOGGER = logging.getLogger("floorspace.spatial")


def load_communes(path):
    """Load commune polygons and reproject to the working CRS (EPSG:2180)."""
    LOGGER.info("Loading communes: %s", path)
    gdf = gpd.read_file(path)
    LOGGER.info("  %d communes, CRS=%s", len(gdf), gdf.crs)
    gdf = gdf.to_crs(CRS_WORK)
    cols = [COMMUNES_ID_COL, "geometry"]
    if COMMUNES_POP_COL in gdf.columns:
        cols.insert(1, COMMUNES_POP_COL)
    gdf = gdf[cols].rename(columns={COMMUNES_ID_COL: "gmina_teryt", COMMUNES_POP_COL: "pop"})
    return gdf


def _sjoin_chunk(points_xy, communes):
    """Worker: within-join a chunk of (x, y) points to communes."""
    pts = gpd.GeoDataFrame(
        {"_row": np.arange(len(points_xy))},
        geometry=gpd.points_from_xy(points_xy[:, 0], points_xy[:, 1]),
        crs=communes.crs,
    )
    joined = gpd.sjoin(pts, communes[["gmina_teryt", "geometry"]], how="left", predicate="within")
    # duplicate points on borders -> keep first
    joined = joined.drop_duplicates(subset="_row")
    return joined.set_index("_row")["gmina_teryt"].reindex(range(len(points_xy))).to_numpy()


def assign_gmina(df: pd.DataFrame, communes: gpd.GeoDataFrame, workers: int) -> pd.DataFrame:
    """Add ``gmina_teryt`` (and ``teryt_ok`` validation flag) to a point table."""
    xy = df[["x2180", "y2180"]].to_numpy()
    n_jobs = workers if workers and workers > 0 else (os.cpu_count() or 4)
    n_chunks = max(n_jobs, 1) * 4
    chunks = [c for c in np.array_split(xy, n_chunks) if len(c)]
    LOGGER.info("Spatial join: %s points, workers=%d, chunks=%d", f"{len(xy):,}", n_jobs, len(chunks))
    parts = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_sjoin_chunk)(c, communes) for c in tqdm(chunks, desc="sjoin", unit="chunk")
    )
    gmina = np.concatenate(parts)
    out = df.copy()
    out["gmina_teryt"] = gmina
    matched = pd.notna(out["gmina_teryt"]).mean()
    LOGGER.info("  matched to a gmina: %.2f%%", 100 * matched)

    # validation: derived powiat vs recorded RCN teryt (first 4 digits)
    derived_pow = out["gmina_teryt"].astype("string").str[:4]
    out["teryt_ok"] = derived_pow.eq(out["powiat_teryt"].astype("string"))
    ok = out.loc[out["gmina_teryt"].notna(), "teryt_ok"].mean()
    LOGGER.info("  powiat agreement (geom vs RCN teryt): %.2f%%", 100 * ok)
    return out


def gmina_land_covariate(land_df: pd.DataFrame, communes: gpd.GeoDataFrame, workers: int) -> pd.DataFrame:
    """Median residential-land zl/m2 per gmina from undeveloped-land points."""
    assigned = assign_gmina(land_df, communes, workers)
    assigned = assigned[assigned["gmina_teryt"].notna()]
    agg = (
        assigned.groupby("gmina_teryt")
        .agg(land_ppm2_med=("land_ppm2", "median"), n_land=("land_ppm2", "size"))
        .reset_index()
    )
    agg["log_land"] = np.log(agg["land_ppm2_med"])
    LOGGER.info("Land covariate built for %d gminas", len(agg))
    return agg
