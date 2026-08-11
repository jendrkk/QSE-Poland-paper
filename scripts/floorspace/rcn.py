#!/usr/bin/env python3
"""
rcn.py
======

Read and clean the three RCN GeoPackages into analysis-ready tables.

The GeoPackages are large (1.2-7 GB). Rather than depend on GDAL/pyogrio to
materialise millions of polygons, this module reads the SQLite feature tables
directly, streams rows in ``fid`` ranges (parallelised over CPU cores), and
parses the GPKG geometry BLOBs by hand:

* ``lokale``  -> POINT               -> (x, y) exactly.
* ``budynki`` -> (MULTI)POLYGON      -> exterior-ring centroid (representative
                                        point; buildings are tiny relative to
                                        gminas so this is safe for assignment).
* ``dzialki`` -> (MULTI)POLYGON      -> exterior-ring centroid.

Coordinates are in EPSG:2180 (metres). All numeric fields are TEXT in the
source and are parsed defensively (space-stripped, decimal-comma normalised).

Public API
----------
build_micro(cfg, workers, sample_frac) -> pandas.DataFrame
    Stacked flats (+ houses) micro table with parsed price/area/covariates and
    (x2180, y2180) coordinates, after all row-level cleaning and outlier trims
    EXCEPT the spatial (gmina) assignment, which happens in ``spatial.py``.
build_land(cfg, workers, sample_frac) -> pandas.DataFrame
    Undeveloped residential-land transactions (x, y, land price/m2) for the
    gmina land-price covariate.
"""

from __future__ import annotations

import logging
import sqlite3
import struct
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from joblib import Parallel, delayed
except ImportError:  # pragma: no cover - joblib present on HPC
    Parallel = None

    def delayed(fn):
        return fn

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(it=None, total=None, **_):
        return it if it is not None else range(total or 0)

from config import (
    CleaningConfig,
    BUD_HOUSE_NIER,
    BUD_HOUSE_RODZAJ,
    DZI_UNDEVELOPED,
    TRANS_ARM_LENGTH,
    MARKET_PRIMARY,
)

LOGGER = logging.getLogger("floorspace.rcn")


# --------------------------------------------------------------------------- #
# GPKG geometry BLOB parsing
# --------------------------------------------------------------------------- #
_ENV_SIZE = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}


def _wkb_start(blob: bytes) -> int:
    """Return the byte offset of the standard WKB inside a GPKG geometry BLOB."""
    # bytes: 'G','P', version, flags, srs(4)
    flags = blob[3]
    env = (flags >> 1) & 0x07
    return 8 + _ENV_SIZE.get(env, 0)


def _read_point(blob: bytes):
    """Parse a GPKG POINT BLOB -> (x, y). Returns (nan, nan) on failure."""
    try:
        off = _wkb_start(blob)
        bo = blob[off]
        end = "<" if bo == 1 else ">"
        # skip 1 byte order + 4 type
        x, y = struct.unpack_from(end + "dd", blob, off + 5)
        return x, y
    except Exception:
        return np.nan, np.nan


def _read_repr_point(blob: bytes):
    """Parse a GPKG (MULTI)POLYGON BLOB -> exterior-ring centroid (x, y).

    Uses the first polygon's outer ring and the shoelace centroid; falls back
    to the vertex mean if the ring is degenerate. Returns (nan, nan) on
    failure.
    """
    try:
        off = _wkb_start(blob)
        bo = blob[off]
        end = "<" if bo == 1 else ">"
        gtype = struct.unpack_from(end + "I", blob, off + 1)[0] % 1000
        p = off + 5
        if gtype == 6:  # MultiPolygon: descend into first polygon
            # numPolygons (skip), then the first polygon has its own byte order+type
            p += 4
            bo = blob[p]
            end = "<" if bo == 1 else ">"
            gtype = struct.unpack_from(end + "I", blob, p + 1)[0] % 1000
            p += 5
        if gtype != 3:  # not a polygon; treat as point-like
            x, y = struct.unpack_from(end + "dd", blob, p)
            return x, y
        n_rings = struct.unpack_from(end + "I", blob, p)[0]
        p += 4
        if n_rings == 0:
            return np.nan, np.nan
        n_pts = struct.unpack_from(end + "I", blob, p)[0]
        p += 4
        coords = np.frombuffer(blob, dtype=(end + "f8"), count=2 * n_pts, offset=p)
        xs = coords[0::2]
        ys = coords[1::2]
        return _ring_centroid(xs, ys)
    except Exception:
        return np.nan, np.nan


def _ring_centroid(xs: np.ndarray, ys: np.ndarray):
    """Shoelace centroid of a closed ring; vertex-mean fallback."""
    if xs.size < 3:
        return float(xs.mean()), float(ys.mean())
    x1, y1 = xs[:-1], ys[:-1]
    x2, y2 = xs[1:], ys[1:]
    cross = x1 * y2 - x2 * y1
    area = cross.sum() / 2.0
    if abs(area) < 1e-9:
        return float(xs.mean()), float(ys.mean())
    cx = ((x1 + x2) * cross).sum() / (6.0 * area)
    cy = ((y1 + y2) * cross).sum() / (6.0 * area)
    return float(cx), float(cy)


# --------------------------------------------------------------------------- #
# Numeric parsing
# --------------------------------------------------------------------------- #
def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(
        s.astype("string").str.replace(" ", "", regex=False).str.replace(",", ".", regex=False),
        errors="coerce",
    )


# --------------------------------------------------------------------------- #
# Chunked parallel reader
# --------------------------------------------------------------------------- #
def _fid_bounds(gpkg: Path, table: str):
    con = sqlite3.connect(gpkg)
    lo, hi, n = con.execute(f"SELECT MIN(fid), MAX(fid), COUNT(*) FROM {table}").fetchone()
    con.close()
    return lo, hi, n


def _read_chunk(gpkg, table, columns, where, geom_kind, lo, hi):
    """Worker: read one fid range, parse geometry, return a DataFrame."""
    con = sqlite3.connect(gpkg)
    con.text_factory = bytes  # geom is bytes; decode text cols manually
    cols_sql = ", ".join(columns + ["geom"])
    sql = f"SELECT {cols_sql} FROM {table} WHERE fid >= ? AND fid < ?"
    if where:
        sql += f" AND ({where})"
    rows = con.execute(sql, (lo, hi)).fetchall()
    con.close()
    if not rows:
        return None
    ncol = len(columns)
    data = {c: [] for c in columns}
    xs = np.empty(len(rows), dtype="f8")
    ys = np.empty(len(rows), dtype="f8")
    reader = _read_point if geom_kind == "point" else _read_repr_point
    for i, r in enumerate(rows):
        for j, c in enumerate(columns):
            v = r[j]
            data[c].append(v.decode("utf-8") if isinstance(v, bytes) else v)
        g = r[ncol]
        if g is None:
            xs[i] = np.nan
            ys[i] = np.nan
        else:
            xs[i], ys[i] = reader(g)
    df = pd.DataFrame(data)
    df["x2180"] = xs
    df["y2180"] = ys
    return df


def read_layer(gpkg, table, columns, where, geom_kind, workers, chunk_rows=200_000):
    """Stream a feature table in parallel fid-range chunks; parse geometry."""
    lo, hi, n = _fid_bounds(gpkg, table)
    LOGGER.info("Reading %s: %s rows in [%s, %s], where=%s", table, f"{n:,}", lo, hi, where or "-")
    edges = list(range(lo, hi + 1, chunk_rows)) + [hi + 1]
    ranges = list(zip(edges[:-1], edges[1:]))
    n_jobs = workers if workers and workers > 0 else 1
    if Parallel is not None and n_jobs != 1:
        parts = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_read_chunk)(gpkg, table, columns, where, geom_kind, a, b)
            for a, b in tqdm(ranges, desc=f"read {table}", unit="chunk")
        )
    else:
        parts = [
            _read_chunk(gpkg, table, columns, where, geom_kind, a, b)
            for a, b in tqdm(ranges, desc=f"read {table}", unit="chunk")
        ]
    parts = [p for p in parts if p is not None and len(p)]
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    LOGGER.info("  -> %s rows read from %s", f"{len(df):,}", table)
    return df


# --------------------------------------------------------------------------- #
# Shared cleaning helpers
# --------------------------------------------------------------------------- #
def _year_from_dokdata(s: pd.Series) -> pd.Series:
    y = pd.to_numeric(s.astype("string").str.slice(0, 4), errors="coerce")
    return y


def _coalesce_price(df, comp_col, tran_id_col="tran_lokalny_id_iip"):
    """Unit price = comp -> nier -> tran, with the tran fallback only for
    single-unit transactions (one retained row per tran id)."""
    comp = _to_num(df[comp_col])
    nier = _to_num(df["nier_cena_brutto"])
    tran = _to_num(df["tran_cena_brutto"]) if "tran_cena_brutto" in df else pd.Series(np.nan, index=df.index)
    price = comp.where(comp > 0, nier)
    # single-unit mask for tran fallback
    counts = df.groupby(tran_id_col)[tran_id_col].transform("size")
    single = counts.eq(1)
    price = price.where(price > 0, tran.where(single))
    return price


def _robust_trim(df, cfg: CleaningConfig):
    """Drop |z|>robust_z on log(price/m2) within powiat x type x market x year
    strata (median/MAD scale), falling back to the coarser powiat x type
    stratum where the fine stratum has fewer than ``robust_min_stratum`` rows.

    Fully vectorised (groupby transforms only) for speed and pandas-version
    robustness at 1e6+ rows.
    """
    n0 = len(df)
    work = df.copy()
    work["_lp"] = np.log(work["ppm2"].to_numpy())

    keys_fine = ["powiat_teryt", "property_type", "market", "year"]
    keys_coarse = ["powiat_teryt", "property_type"]

    def _key(cols):
        return (
            work[cols]
            .astype("string")
            .fillna("NA")
            .agg("|".join, axis=1)
        )

    size_fine = work.groupby(keys_fine)["_lp"].transform("size")
    use_fine = size_fine >= cfg.robust_min_stratum
    work["_strat"] = np.where(use_fine, _key(keys_fine), _key(keys_coarse))

    med = work.groupby("_strat")["_lp"].transform("median")
    work["_ad"] = (work["_lp"] - med).abs()
    mad = work.groupby("_strat")["_ad"].transform("median") * 1.4826
    keep = (mad <= 0) | (work["_ad"] <= cfg.robust_z * mad)

    out = df.loc[keep.to_numpy()]
    LOGGER.info("  robust trim: %s -> %s (dropped %s)", f"{n0:,}", f"{len(out):,}", f"{n0 - len(out):,}")
    return out


# --------------------------------------------------------------------------- #
# Layer builders
# --------------------------------------------------------------------------- #
def _clean_flats(df, cfg: CleaningConfig):
    df = df.copy()
    df["price"] = _coalesce_price(df, "lok_cena_brutto")
    df["area"] = _to_num(df["lok_pow_uzyt"])
    df["rooms"] = _to_num(df["lok_liczba_izb"])
    df["floor"] = _to_num(df["lok_nr_kond"])
    df["plot_area"] = np.nan
    df["year"] = _year_from_dokdata(df["dok_data"])
    df["market"] = np.where(df["tran_rodzaj_rynku"].eq(MARKET_PRIMARY), "primary", "secondary")
    df["property_type"] = "flat"
    df["powiat_teryt"] = df["teryt"].astype("string")
    df["unit_id"] = df["lok_id_lokalu"]
    if cfg.drop_public_sellers:
        df = df[~df["tran_sprzedajacy"].isin(list(config_public()))]
    return _finalise(df, cfg, area_min=cfg.flat_area_min, area_max=cfg.flat_area_max)


def _clean_houses(df, cfg: CleaningConfig):
    df = df.copy()
    df["price"] = _coalesce_price(df, "bud_cena_brutto")
    df["area"] = _to_num(df["bud_pow_uzyt"])
    df["rooms"] = np.nan
    df["floor"] = np.nan
    df["plot_area"] = _to_num(df["nier_pow_gruntu"])
    df["year"] = _year_from_dokdata(df["dok_data"])
    df["market"] = np.where(df["tran_rodzaj_rynku"].eq(MARKET_PRIMARY), "primary", "secondary")
    df["property_type"] = "house"
    df["powiat_teryt"] = df["teryt"].astype("string")
    df["unit_id"] = df["bud_id_budynku"]
    # plot sanity for houses
    df = df[(df["plot_area"].isna()) | ((df["plot_area"] >= cfg.house_plot_min) & (df["plot_area"] <= cfg.house_plot_max))]
    if cfg.drop_public_sellers:
        df = df[~df["tran_sprzedajacy"].isin(list(config_public()))]
    return _finalise(df, cfg, area_min=cfg.house_area_min, area_max=cfg.house_area_max)


def config_public():
    from config import PUBLIC_SELLERS
    return PUBLIC_SELLERS


def _finalise(df, cfg: CleaningConfig, area_min, area_max):
    """Shared row-level gates, dedup, ppm2, hard bounds. (Robust trim later,
    after both layers are combined, so strata are shared.)"""
    n0 = len(df)
    df = df[
        (df["price"] > 0)
        & (df["area"] >= area_min)
        & (df["area"] <= area_max)
        & (df["year"] >= cfg.year_min)
        & (df["year"] <= cfg.year_max)
        & df["x2180"].notna()
    ].copy()
    df["ppm2"] = df["price"] / df["area"]
    df = df[(df["ppm2"] >= cfg.ppm2_min) & (df["ppm2"] <= cfg.ppm2_max)]
    # dedup: same unit, date, price; and same (tran id, unit id)
    df = df.drop_duplicates(subset=["tran_lokalny_id_iip", "unit_id"])
    df = df.drop_duplicates(subset=["unit_id", "dok_data", "price"])
    LOGGER.info("  finalise %s: %s -> %s rows", df["property_type"].iloc[0] if len(df) else "?", f"{n0:,}", f"{len(df):,}")
    return df


def build_micro(cfg: CleaningConfig, workers: int, sample_frac: float | None = None):
    """Construct the stacked flats+houses micro table (pre spatial join)."""
    from config import LOKALE_GPKG, BUDYNKI_GPKG

    flat_cols = [
        "teryt", "tran_rodzaj_trans", "tran_rodzaj_rynku", "tran_sprzedajacy",
        "dok_data", "nier_udzial", "nier_cena_brutto", "tran_cena_brutto",
        "lok_funkcja", "lok_pow_uzyt", "lok_liczba_izb", "lok_nr_kond",
        "lok_cena_brutto", "lok_id_lokalu", "tran_lokalny_id_iip",
    ]
    flat_where = (
        "lok_funkcja='mieszkalna' AND tran_rodzaj_trans='wolnyRynek' "
        "AND (nier_udzial='1/1' OR nier_udzial IS NULL)"
    )
    flats_raw = read_layer(LOKALE_GPKG, "lokale", flat_cols, flat_where, "point", workers)
    if sample_frac:
        flats_raw = flats_raw.sample(frac=sample_frac, random_state=0)
    flats = _clean_flats(flats_raw, cfg)

    house_cols = [
        "teryt", "tran_rodzaj_trans", "tran_rodzaj_rynku", "tran_sprzedajacy",
        "dok_data", "nier_udzial", "nier_cena_brutto", "tran_cena_brutto",
        "nier_rodzaj", "bud_rodzaj", "bud_pow_uzyt", "nier_pow_gruntu",
        "bud_cena_brutto", "bud_id_budynku", "tran_lokalny_id_iip",
    ]
    house_where = (
        f"nier_rodzaj='{BUD_HOUSE_NIER}' AND bud_rodzaj='{BUD_HOUSE_RODZAJ}' "
        "AND bud_pow_uzyt IS NOT NULL AND bud_pow_uzyt<>'' "
        f"AND tran_rodzaj_trans='{TRANS_ARM_LENGTH}' "
        "AND (nier_udzial='1/1' OR nier_udzial IS NULL)"
    )
    houses_raw = read_layer(BUDYNKI_GPKG, "budynki", house_cols, house_where, "polygon", workers)
    if sample_frac:
        houses_raw = houses_raw.sample(frac=sample_frac, random_state=0)
    houses = _clean_houses(houses_raw, cfg)

    keep = [
        "price", "area", "ppm2", "property_type", "market", "rooms", "floor",
        "plot_area", "year", "x2180", "y2180", "powiat_teryt",
        "tran_lokalny_id_iip", "unit_id",
    ]
    micro = pd.concat([flats[keep], houses[keep]], ignore_index=True)
    micro = _robust_trim(micro, cfg)
    LOGGER.info(
        "MICRO assembled: %s rows (flats=%s, houses=%s)",
        f"{len(micro):,}",
        f"{(micro.property_type == 'flat').sum():,}",
        f"{(micro.property_type == 'house').sum():,}",
    )
    return micro


def build_land(cfg: CleaningConfig, workers: int, sample_frac: float | None = None):
    """Undeveloped residential-land transactions for the gmina land covariate."""
    from config import DZIALKI_GPKG

    cols = [
        "teryt", "tran_rodzaj_trans", "dok_data", "nier_udzial",
        "nier_cena_brutto", "dzi_cena_brutto", "dzi_pow_ewid",
        "dzi_przezn_wmpzp", "dzi_sposob_uzyt", "tran_lokalny_id_iip",
    ]
    where = (
        f"nier_rodzaj='{DZI_UNDEVELOPED}' AND tran_rodzaj_trans='{TRANS_ARM_LENGTH}' "
        "AND (dzi_przezn_wmpzp LIKE '%budownictwoMieszkaniowe%' "
        "OR dzi_sposob_uzyt='gruntyZabudowaneIZurbanizowane')"
    )
    raw = read_layer(DZIALKI_GPKG, "dzialki", cols, where, "polygon", workers)
    if sample_frac:
        raw = raw.sample(frac=sample_frac, random_state=0)
    df = raw.copy()
    price = _to_num(df["nier_cena_brutto"])
    price = price.where(price > 0, _to_num(df["dzi_cena_brutto"]))
    area = _to_num(df["dzi_pow_ewid"])
    df["land_price"] = price
    df["land_area"] = area
    df["year"] = _year_from_dokdata(df["dok_data"])
    df = df[(df["land_price"] > 0) & (df["land_area"] > 0) & df["x2180"].notna()]
    df["land_ppm2"] = df["land_price"] / df["land_area"]
    # broad land sanity gate
    df = df[(df["land_ppm2"] >= 1) & (df["land_ppm2"] <= 20_000)]
    df = df[(df["year"] >= cfg.year_min) & (df["year"] <= cfg.year_max)]
    df = df.drop_duplicates(subset=["tran_lokalny_id_iip"])
    df = df.rename(columns={"teryt": "powiat_teryt"})
    LOGGER.info("LAND assembled: %s residential-land transactions", f"{len(df):,}")
    return df[["x2180", "y2180", "powiat_teryt", "land_ppm2", "year"]]
