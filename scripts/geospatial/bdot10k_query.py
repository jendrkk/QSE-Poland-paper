#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bdot10k_query.py
================================================================================
Download the NUMBER OF STOREYS (liczba kondygnacji) and building function for
Polish buildings from BDOT10k, and estimate gross floor area for joining to RCN
transaction records.

WHY EGiB DID NOT WORK
--------------------------------------------------------------------------------
The previous script queried the national aggregated EGiB WFS
(PZGIK/EGIB/WFS/UslugaZbiorcza -> ms:budynki). That harmonised service *declares*
the columns KONDYGNACJE_NADZIEMNE / KONDYGNACJE_PODZIEMNE, but the aggregated
national feed is fed by ~380 county registers and almost none of them publish the
storey attribute through the shared WFS. Hence every value comes back as None.
The storeys you SEE on the geoportal map come from the county's own map service,
not from this WFS, so they are not reachable this way at national scale.

THE CORRECT SOURCE: BDOT10k, layer OT_BUBD_A
--------------------------------------------------------------------------------
BDOT10k (Baza Danych Obiektow Topograficznych 1:10 000), maintained centrally by
GUGiK, contains a building layer OT_BUBD_A whose attributes
    - liczbaKondygnacji          (number of above-ground storeys)  -- MANDATORY
    - funkcjaOgolnaBudynku        (general function)               -- MANDATORY
    - funkcjaSzczegolowaBudynku   (detailed function)              -- MANDATORY
are obligatory under the 2021 regulation, so coverage is effectively national,
INCLUDING rural areas -- exactly where RCN is thin. Each building is a footprint
polygon, so
        gross_floor_area ~= footprint_area_m2 * liczbaKondygnacji
gives the m^2 proxy needed for the hedonic regression.

HOW THE DATA IS OBTAINED (verified against the live services, 2026-08)
--------------------------------------------------------------------------------
GUGiK publishes ready county packages. The download WMS
    https://mapy.geoportal.gov.pl/wss/service/PZGIK/BDOT/WMS/PobieranieBDOT10k
has three queryable layers: Powiaty, Wojewodztwa, Panstwo. A GetFeatureInfo on
'Powiaty' at a coordinate returns, among others:
    JPT_KOD_JE = '1412'                          (powiat TERYT)
    WOJ        = '14'
    URL_GPKG   = https://opendata.geoportal.gov.pl/bdot10k/schemat2021/GPKG/14/1412_GPKG.zip
    URL        = https://opendata.geoportal.gov.pl/bdot10k/schemat2021/14/1412_GML.zip
So the per-county package URL is fully deterministic:
    https://opendata.geoportal.gov.pl/bdot10k/schemat2021/GPKG/{WOJ}/{POWIAT}_GPKG.zip
This script resolves the county from coordinates via the WMS (robust, no external
boundary file needed), downloads & caches the county GPKG once, reads OT_BUBD_A,
and either (a) returns buildings near a point, or (b) spatially joins storeys to a
whole table of RCN transactions grouped by county.

USAGE
--------------------------------------------------------------------------------
  # single point (mirrors the old script, but storeys are now populated):
  python bdot10k_query.py --lon 21.363187 --lat 52.231227 --radius 500

  # batch-join an RCN transactions CSV/Parquet (must have lon/lat columns):
  python bdot10k_query.py --transactions data/rcn/transactions.csv \
         --lon-col longitude --lat-col latitude --out data/processed/bdot10k/rcn_with_floors.csv

Dependencies: requests, geopandas, shapely, pyproj, pandas  (pyogrio recommended)
              pip install requests geopandas shapely pyproj pandas pyogrio
================================================================================
"""

from __future__ import annotations

import argparse
import io
import os
import re
import sys
import zipfile
from pathlib import Path
from typing import Optional

import requests

# Geospatial stack is imported lazily inside functions that need it so that
# --help and the WMS resolver work even in a minimal environment.

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
WMS_PBDOT = "https://mapy.geoportal.gov.pl/wss/service/PZGIK/BDOT/WMS/PobieranieBDOT10k"
OPENDATA_TMPL = "https://opendata.geoportal.gov.pl/bdot10k/schemat2021/GPKG/{woj}/{powiat}_GPKG.zip"
EPSG_PL = 2180          # PUWG 1992 (metric, national)
EPSG_WGS84 = 4326

# Cache/output roots. Override with --cache-dir / --out.
DEFAULT_CACHE = Path("data/processed/bdot10k/cache")

# Regex identifying the building layer inside the BDOT10k package.
BUBD_LAYER_RE = re.compile(r"BUBD_A", re.IGNORECASE)

# Substring used to flag residential buildings. It matches every encoding the
# dictionaries use ("mieszkalny", "budynki mieszkalne", "mieszkaniowy", ...).
RESIDENTIAL_SUBSTR = "mieszk"

USER_AGENT = "QSE-Poland-BDOT10k/1.0 (research; hedonic floor-area estimation)"
HTTP_TIMEOUT = 120


# ==============================================================================
# 1. Resolve county (powiat) from coordinates via the download WMS
# ==============================================================================
def _to_epsg2180(lon: float, lat: float) -> tuple[float, float]:
    """WGS84 lon/lat -> EPSG:2180 (X=easting, Y=northing as used in BBOX/1.1.1)."""
    from pyproj import Transformer
    tr = Transformer.from_crs(EPSG_WGS84, EPSG_PL, always_xy=True)
    x, y = tr.transform(lon, lat)
    return x, y


def resolve_powiat(lon: float, lat: float, session: Optional[requests.Session] = None) -> dict:
    """
    Return {'powiat': '1412', 'woj': '14', 'nazwa': 'Powiat minski',
            'url_gpkg': ..., 'url_gml': ...} for the county containing (lon, lat).

    Uses WMS 1.1.1 GetFeatureInfo on the 'Powiaty' layer. Version 1.1.1 is used
    deliberately: it fixes axis order to (easting, northing), sidestepping the
    EPSG:2180 axis-order trap in WMS 1.3.0.
    """
    s = session or requests.Session()
    x, y = _to_epsg2180(lon, lat)
    half = 100.0  # 200 m query box centred on the point
    params = {
        "SERVICE": "WMS",
        "VERSION": "1.1.1",
        "REQUEST": "GetFeatureInfo",
        "LAYERS": "Powiaty",
        "QUERY_LAYERS": "Powiaty",
        "SRS": "EPSG:2180",
        "BBOX": f"{x-half},{y-half},{x+half},{y+half}",
        "WIDTH": "3",
        "HEIGHT": "3",
        "X": "1",
        "Y": "1",
        "INFO_FORMAT": "text/plain",
        "FEATURE_COUNT": "1",
    }
    r = s.get(WMS_PBDOT, params=params, timeout=HTTP_TIMEOUT,
              headers={"User-Agent": USER_AGENT})
    r.raise_for_status()
    txt = r.text

    def grab(key: str) -> Optional[str]:
        m = re.search(rf"{key}\s*=\s*'([^']*)'", txt)
        return m.group(1) if m else None

    powiat = grab("JPT_KOD_JE")
    woj = grab("WOJ")
    url_gpkg = grab("URL_GPKG")
    url_gml = grab("URL")
    if not powiat:
        raise RuntimeError(
            f"Could not resolve powiat for lon={lon}, lat={lat}.\n"
            f"WMS GetFeatureInfo returned:\n{txt[:500]}"
        )
    if not woj:
        woj = powiat[:2]
    if not url_gpkg:
        url_gpkg = OPENDATA_TMPL.format(woj=woj, powiat=powiat)
    return {
        "powiat": powiat,
        "woj": woj,
        "nazwa": grab("JPT_NAZWA_"),
        "url_gpkg": url_gpkg,
        "url_gml": url_gml,
    }


# ==============================================================================
# 2. Download & cache the county BDOT10k package, extract OT_BUBD_A
# ==============================================================================
def download_powiat_package(powiat: str, woj: str, url_gpkg: Optional[str],
                            cache_dir: Path,
                            session: Optional[requests.Session] = None) -> Path:
    """Download (once) and unzip the county GPKG package. Returns extraction dir."""
    s = session or requests.Session()
    cache_dir = Path(cache_dir)
    out_dir = cache_dir / f"{powiat}_GPKG"
    if out_dir.exists() and any(out_dir.rglob("*.gpkg")):
        return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    url = url_gpkg or OPENDATA_TMPL.format(woj=woj, powiat=powiat)
    print(f"  downloading {url}")
    with s.get(url, stream=True, timeout=HTTP_TIMEOUT,
               headers={"User-Agent": USER_AGENT}) as r:
        r.raise_for_status()
        buf = io.BytesIO(r.content)
    with zipfile.ZipFile(buf) as zf:
        zf.extractall(out_dir)
    return out_dir


def _find_building_gpkg_layer(extract_dir: Path):
    """Locate the (gpkg_path, layer_name) holding OT_BUBD_A inside extract_dir."""
    try:
        from pyogrio import list_layers
        def layers(p): return [row[0] for row in list_layers(p)]
    except Exception:
        import fiona
        def layers(p): return list(fiona.listlayers(p))

    for gpkg in sorted(extract_dir.rglob("*.gpkg")):
        # Fast path: a package named per-layer already tells us it's the building file.
        if BUBD_LAYER_RE.search(gpkg.name):
            lyrs = layers(gpkg)
            hit = next((l for l in lyrs if BUBD_LAYER_RE.search(l)), lyrs[0] if lyrs else None)
            if hit:
                return gpkg, hit
        for lyr in layers(gpkg):
            if BUBD_LAYER_RE.search(lyr):
                return gpkg, lyr
    raise FileNotFoundError(f"No OT_BUBD_A layer found under {extract_dir}")


def load_buildings(powiat: str, woj: str, cache_dir: Path,
                   url_gpkg: Optional[str] = None,
                   session: Optional[requests.Session] = None):
    """
    Return a GeoDataFrame of buildings for one county with harmonised columns:
        building_id, floors_above, function_general, function_detailed,
        footprint_m2, geometry (EPSG:2180)
    """
    import geopandas as gpd

    extract_dir = download_powiat_package(powiat, woj, url_gpkg, cache_dir, session)
    gpkg, layer = _find_building_gpkg_layer(extract_dir)
    gdf = gpd.read_file(gpkg, layer=layer)
    if gdf.crs is None:
        gdf = gdf.set_crs(EPSG_PL)
    elif gdf.crs.to_epsg() != EPSG_PL:
        gdf = gdf.to_crs(EPSG_PL)

    cols = {c.lower(): c for c in gdf.columns}

    def pick(*cands):
        for c in cands:
            if c.lower() in cols:
                return cols[c.lower()]
        return None

    c_floors = pick("liczbaKondygnacji", "LICZBAKONDYGNACJI", "liczba_kondygnacji")
    c_fgen = pick("funkcjaOgolnaBudynku", "FUNKCJAOGOLNABUDYNKU", "funkcja_ogolna_budynku")
    c_fdet = pick("funkcjaSzczegolowaBudynku", "FUNKCJASZCZEGOLOWABUDYNKU",
                  "funkcja_szczegolowa_budynku")
    c_id = pick("lokalnyId", "LOKALNYID", "idIIP", "gml_id", "ID_BUDYNKU")

    out = gpd.GeoDataFrame(geometry=gdf.geometry, crs=gdf.crs)
    out["building_id"] = gdf[c_id] if c_id else range(len(gdf))
    out["floors_above"] = gdf[c_floors] if c_floors else None
    out["function_general"] = gdf[c_fgen] if c_fgen else None
    out["function_detailed"] = gdf[c_fdet] if c_fdet else None
    out["footprint_m2"] = out.geometry.area  # EPSG:2180 -> metres
    return out


# ==============================================================================
# 3a. Single-point query (drop-in replacement for the old behaviour)
# ==============================================================================
def query_point(lon: float, lat: float, radius: float, cache_dir: Path):
    """Buildings within `radius` metres of (lon, lat), WITH storeys populated."""
    import geopandas as gpd
    from shapely.geometry import Point

    session = requests.Session()
    info = resolve_powiat(lon, lat, session)
    print(f"Powiat: {info['nazwa']} (TERYT {info['powiat']}, woj {info['woj']})")

    buildings = load_buildings(info["powiat"], info["woj"], cache_dir,
                               info.get("url_gpkg"), session)

    x, y = _to_epsg2180(lon, lat)
    pt = gpd.GeoSeries([Point(x, y)], crs=EPSG_PL).iloc[0]
    buildings = buildings.copy()
    buildings["distance_m"] = buildings.geometry.distance(pt)
    near = buildings[buildings["distance_m"] <= radius].sort_values("distance_m")

    near["floor_area_m2"] = near["footprint_m2"] * near["floors_above"].astype("float")
    return near


# ==============================================================================
# 3b. Batch join to an RCN transactions table
# ==============================================================================
def join_transactions(df, lon_col: str, lat_col: str, cache_dir: Path,
                      max_snap_m: float = 40.0):
    """
    Attach BDOT10k storeys + estimated gross floor area to every transaction.

    Strategy per county (downloaded once):
      1. point-in-polygon  (transaction sits on its building footprint)
      2. fallback: nearest building within `max_snap_m` metres
    Adds columns: powiat, floors_above, function_general, function_detailed,
                  footprint_m2, floor_area_m2, match_type, match_dist_m.
    """
    import geopandas as gpd
    import pandas as pd
    from pyproj import Transformer

    session = requests.Session()
    tr = Transformer.from_crs(EPSG_WGS84, EPSG_PL, always_xy=True)

    df = df.reset_index(drop=True).copy()
    xs, ys = tr.transform(df[lon_col].values, df[lat_col].values)
    pts = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(xs, ys), crs=EPSG_PL
    )

    # Assign each transaction to a county. One WMS call per UNIQUE rounded
    # coordinate keeps traffic minimal; nearby points share a county.
    key = list(zip(df[lon_col].round(3), df[lat_col].round(3)))
    pts["_key"] = key
    powiat_of = {}
    for k in dict.fromkeys(key):  # unique, order-preserving
        try:
            powiat_of[k] = resolve_powiat(k[0], k[1], session)["powiat"]
        except Exception as e:
            print(f"  ! powiat resolve failed for {k}: {e}")
            powiat_of[k] = None
    pts["powiat"] = pts["_key"].map(lambda k: powiat_of.get(k))

    results = []
    for powiat, grp in pts.groupby("powiat"):
        if powiat is None:
            grp = grp.assign(match_type="no_powiat")
            results.append(grp)
            continue
        woj = str(powiat)[:2]
        print(f"[powiat {powiat}] {len(grp)} transactions")
        try:
            b = load_buildings(powiat, woj, cache_dir, session=session)
        except Exception as e:
            print(f"  ! load failed: {e}")
            results.append(grp.assign(match_type="load_failed"))
            continue

        bcols = ["building_id", "floors_above", "function_general",
                 "function_detailed", "footprint_m2", "geometry"]
        b = b[bcols] if all(c in b.columns for c in bcols) else b

        # (1) point in polygon
        pip = gpd.sjoin(grp, b, how="left", predicate="within")
        pip = pip[~pip.index.duplicated(keep="first")]
        pip["match_type"] = pip["building_id"].notna().map(
            {True: "within", False: None})
        pip["match_dist_m"] = 0.0

        # (2) nearest for the unmatched
        unmatched = pip[pip["building_id"].isna()]
        if len(unmatched):
            near = gpd.sjoin_nearest(
                grp.loc[unmatched.index], b, how="left",
                max_distance=max_snap_m, distance_col="match_dist_m")
            near = near[~near.index.duplicated(keep="first")]
            near["match_type"] = near["building_id"].notna().map(
                {True: "nearest", False: "unmatched"})
            for idx in unmatched.index:
                if idx in near.index:
                    for c in ["building_id", "floors_above", "function_general",
                              "function_detailed", "footprint_m2",
                              "match_type", "match_dist_m"]:
                        if c in near.columns:
                            pip.loc[idx, c] = near.loc[idx, c]
        results.append(pip)

    joined = pd.concat(results).sort_index()
    joined["floors_above"] = pd.to_numeric(joined["floors_above"], errors="coerce")
    joined["floor_area_m2"] = joined["footprint_m2"] * joined["floors_above"]
    joined["is_residential"] = (
        joined["function_general"].astype(str).str.lower().str.contains(RESIDENTIAL_SUBSTR)
        | joined["function_detailed"].astype(str).str.lower().str.contains(RESIDENTIAL_SUBSTR)
    )
    return joined.drop(columns=[c for c in ["_key", "geometry", "index_right"]
                                if c in joined.columns])


# ==============================================================================
# CLI
# ==============================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description="BDOT10k storey / floor-area extractor")
    ap.add_argument("--lon", type=float, help="longitude (WGS84) for single-point mode")
    ap.add_argument("--lat", type=float, help="latitude (WGS84) for single-point mode")
    ap.add_argument("--radius", type=float, default=100, help="search radius [m]")
    ap.add_argument("--transactions", help="CSV/Parquet of RCN transactions (batch mode)")
    ap.add_argument("--lon-col", default="longitude")
    ap.add_argument("--lat-col", default="latitude")
    ap.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    ap.add_argument("--out", default=None, help="output CSV path")
    ap.add_argument("--max-snap-m", type=float, default=40.0,
                    help="max snap distance for nearest-building fallback [m]")
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # ---- batch mode ----
    if args.transactions:
        import pandas as pd
        p = Path(args.transactions)
        df = pd.read_parquet(p) if p.suffix.lower() in (".parquet", ".pq") \
            else pd.read_csv(p)
        if args.lon_col not in df.columns or args.lat_col not in df.columns:
            print(f"ERROR: columns {args.lon_col}/{args.lat_col} not in "
                  f"{list(df.columns)}", file=sys.stderr)
            return 2
        joined = join_transactions(df, args.lon_col, args.lat_col, cache_dir,
                                   max_snap_m=args.max_snap_m)
        out = Path(args.out) if args.out else p.with_name(p.stem + "_with_floors.csv")
        out.parent.mkdir(parents=True, exist_ok=True)
        joined.to_csv(out, index=False)
        n = joined["floors_above"].notna().sum()
        print(f"\nMatched storeys for {n}/{len(joined)} transactions "
              f"({100*n/len(joined):.1f}% coverage)")
        print(f"CSV saved: {out.resolve()}")
        return 0

    # ---- single-point mode ----
    if args.lon is None or args.lat is None:
        ap.error("provide --lon/--lat (point mode) or --transactions (batch mode)")

    near = query_point(args.lon, args.lat, args.radius, cache_dir)
    print("=" * 70)
    print(f"BUILDINGS WITHIN {args.radius:.0f} M: {len(near)}")
    print("=" * 70)
    show = near[["building_id", "floors_above", "function_general",
                 "footprint_m2", "floor_area_m2", "distance_m"]].copy()
    show["footprint_m2"] = show["footprint_m2"].round(1)
    show["floor_area_m2"] = show["floor_area_m2"].round(1)
    show["distance_m"] = show["distance_m"].round(1)
    with_pandas_display(show)

    out = Path(args.out) if args.out else cache_dir / "buildings_point.csv"
    near.drop(columns="geometry").to_csv(out, index=False)
    print(f"\nCSV saved: {out.resolve()}")
    return 0


def with_pandas_display(df):
    import pandas as pd
    with pd.option_context("display.max_rows", 200, "display.width", 160):
        print(df.to_string(index=False))


if __name__ == "__main__":
    raise SystemExit(main())