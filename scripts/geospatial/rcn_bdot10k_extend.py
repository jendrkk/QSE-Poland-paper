#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rcn_bdot10k_extend.py
================================================================================
Enrich RCN transaction buildings (budynki.gpkg) with the NUMBER OF STOREYS and
building function taken from BDOT10k, and impute a usable-floor-area proxy that
fills the (mostly empty) `bud_pow_uzyt` column of the RCN register.

CONTEXT / DATA STRUCTURE (verified on the actual files)
--------------------------------------------------------------------------------
data/raw/floorspace/budynki.gpkg   layer 'budynki'  MULTIPOLYGON  EPSG:2180
    6,233,436 rows. RCN building-level transactions. Key columns:
        teryt              powiat TERYT code (378 distinct) -> partition key
        bud_id_budynku     EGiB building id, e.g. '120908_2.0006.20_BUD'
        bud_rodzaj         RCN building kind (mieszkalny, gospodarczy, ...)
        bud_pow_uzyt       usable area -- populated for only ~1.29M / 6.23M (21%)
        geom               EGiB building footprint polygon
data/raw/floorspace/lokale.gpkg    layer 'lokale'   POINT         EPSG:2180
    2,593,413 rows. Flat-level transactions. `lok_pow_uzyt` IS populated, so
    flats already carry usable area and do NOT need area imputation; BDOT storeys
    are attached only as building context (optional, --lokale).

WHY A GEOMETRIC MATCH (NOT AN ID JOIN)
--------------------------------------------------------------------------------
RCN `bud_id_budynku` is an EGiB identifier. BDOT10k OT_BUBD_A carries its own
UUID (`lokalnyId`) and NO EGiB reference. The two datasets are independent
digitisations of the same physical buildings, so there is no shared key. The
bridge is geometry. RCN footprints and BDOT10k footprints overlap heavily but are
not identical, therefore the matching rule is:

    MAXIMUM INTERSECTION-AREA OVERLAP
      for each RCN footprint, pick the BDOT10k footprint with the largest area of
      intersection, provided that area is >= `min_overlap_frac` of the RCN
      footprint (default 0.10);
    NEAREST-WITHIN-THRESHOLD FALLBACK
      RCN footprints that intersect nothing are snapped to the nearest BDOT10k
      building within `snap_m` metres (default 25 m), for slight mis-registration;
    UNMATCHED
      whatever is left keeps floors = NaN (some RCN buildings genuinely have no
      BDOT10k counterpart -- e.g. demolished / very new / register lag).

AREA MODEL
--------------------------------------------------------------------------------
    footprint_m2   = area of the RCN footprint polygon (EPSG:2180, metric)
    gross_area_m2  = footprint_m2 * bdot_floors           (gross above-ground)
    usable_area_est_m2 = gross_area_m2 * k

`k` converts gross above-ground area to usable area (powierzchnia uzytkowa, per
PN-ISO 9836). It is < 1 because usable area excludes the footprint of external
and internal walls (~10-20 %) and because top storeys under a pitched roof
(poddasze) are counted at reduced area. Literature/practice put k in ~0.75-0.85
for Polish residential stock; the naive 0.9 tends to overstate. DO NOT GUESS if
you can measure: pass --calibrate to estimate k empirically from the ~1.29M RCN
rows that already report `bud_pow_uzyt`
        k_hat = median( bud_pow_uzyt / gross_area_m2 )     (robust, residential,
                                                            plausibility-bounded)
Otherwise the fixed --usable-factor (default 0.80) is used.

DESIGN (exact + fast)
--------------------------------------------------------------------------------
* Partition by `teryt`; one powiat == one BDOT10k package == one worker task.
* multiprocessing.Pool over powiats (default: all CPU cores). Each worker:
    read RCN subset (pyogrio where=) -> download+load BDOT10k powiat (reusing the
    already-working bdot10k_query.load_buildings) -> max-overlap match ->
    footprint/gross area -> write a per-powiat shard + return calibration samples.
* Main process: pick k (fixed or calibrated) -> stream shards into the final
    budynki_bdot10k.gpkg, adding usable_area_est_m2 in one assembly pass.
* Robust: a BDOT download/parse failure for a powiat marks that powiat unmatched
    instead of killing the run.

USAGE
--------------------------------------------------------------------------------
  python rcn_bdot10k_extend.py \
      --budynki data/raw/floorspace/budynki.gpkg \
      --out     data/raw/floorspace/budynki_bdot10k.gpkg \
      --calibrate --workers 8

  # fixed factor instead of calibration, only residential, quick subset:
  python rcn_bdot10k_extend.py --usable-factor 0.82 --only-powiats 1412,2261

  # also enrich flats with building context (keeps lok_pow_uzyt):
  python rcn_bdot10k_extend.py --lokale data/raw/floorspace/lokale.gpkg

Dependencies: geopandas, pyogrio, shapely, pandas, numpy, requests
              + bdot10k_query.py on the path (same scripts/geospatial folder).
================================================================================
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

# geopandas / pyogrio imported lazily inside functions so --help stays light.

EPSG_PL = 2180
DEFAULT_MIN_OVERLAP_FRAC = 0.10
DEFAULT_SNAP_M = 25.0
DEFAULT_USABLE_FACTOR = 0.80
RESIDENTIAL_SUBSTR = "mieszk"

# Reuse the (already validated) BDOT downloader/loader from the query script.
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from bdot10k_query import load_buildings as _bdot_load_buildings
except Exception:  # pragma: no cover - import resolved at runtime on the user host
    _bdot_load_buildings = None


# ==============================================================================
# Core geometric matcher  (validated on real RCN geometry)
# ==============================================================================
def match_max_overlap(rcn, bdot, min_overlap_frac=DEFAULT_MIN_OVERLAP_FRAC,
                      snap_m=DEFAULT_SNAP_M):
    """
    Attach BDOT10k attributes to each RCN building by maximum intersection-area
    overlap, with a nearest-within-`snap_m` fallback.

    `rcn`  : GeoDataFrame of RCN footprints (EPSG:2180), any columns + geometry.
    `bdot` : GeoDataFrame with columns floors_above, function_general,
             function_detailed, building_id, geometry (EPSG:2180).

    Returns `rcn` plus: bdot_floors, bdot_function_general, bdot_function_detailed,
    bdot_building_id, match_type in {overlap, nearest, unmatched},
    match_overlap_frac, match_dist_m, footprint_m2, gross_area_m2.
    """
    import geopandas as gpd

    rcn = rcn.reset_index(drop=True).copy()
    # Repair invalid geometries so .area / .intersection are well defined.
    if not rcn.geometry.is_valid.all():
        rcn["geometry"] = rcn.geometry.buffer(0)
    rcn["footprint_m2"] = rcn.geometry.area
    n = len(rcn)

    # defaults (everything starts unmatched)
    rcn["bdot_floors"] = np.nan
    rcn["bdot_function_general"] = None
    rcn["bdot_function_detailed"] = None
    rcn["bdot_building_id"] = None
    rcn["match_type"] = "unmatched"
    rcn["match_overlap_frac"] = np.nan
    rcn["match_dist_m"] = np.nan

    if bdot is None or len(bdot) == 0:
        rcn["gross_area_m2"] = np.nan
        return rcn

    bdot = bdot.reset_index(drop=True).copy()
    if not bdot.geometry.is_valid.all():
        bdot["geometry"] = bdot.geometry.buffer(0)

    def _assign(ridx, bpos, mtype, frac=np.nan, dist=np.nan):
        b = bdot.iloc[bpos]
        rcn.loc[ridx, "bdot_floors"] = b.get("floors_above", np.nan)
        rcn.loc[ridx, "bdot_function_general"] = b.get("function_general")
        rcn.loc[ridx, "bdot_function_detailed"] = b.get("function_detailed")
        rcn.loc[ridx, "bdot_building_id"] = b.get("building_id")
        rcn.loc[ridx, "match_type"] = mtype
        rcn.loc[ridx, "match_overlap_frac"] = frac
        rcn.loc[ridx, "match_dist_m"] = dist

    # ---- (1) maximum intersection-area overlap -------------------------------
    left = rcn[["geometry"]].copy(); left["_rid"] = np.arange(n)
    right = bdot[["geometry"]].copy(); right["_bid"] = np.arange(len(bdot))
    pairs = gpd.sjoin(left, right, predicate="intersects", how="inner")
    if len(pairs):
        rid = pairs["_rid"].to_numpy()
        bid = pairs["_bid"].to_numpy()
        lg = gpd.GeoSeries(rcn.geometry.to_numpy()[rid], crs=EPSG_PL)
        rg = gpd.GeoSeries(bdot.geometry.to_numpy()[bid], crs=EPSG_PL)
        inter = lg.intersection(rg, align=False).area.to_numpy()
        cand = pd.DataFrame({"_rid": rid, "_bid": bid, "_inter": inter})
        # best (largest intersection) per RCN building
        cand = cand.sort_values("_inter").drop_duplicates("_rid", keep="last")
        frac = cand["_inter"].to_numpy() / rcn["footprint_m2"].to_numpy()[cand["_rid"].to_numpy()]
        cand = cand.assign(_frac=frac)
        cand = cand[cand["_frac"] >= min_overlap_frac]
        for _rid, _bid, _fr in zip(cand["_rid"], cand["_bid"], cand["_frac"]):
            _assign(int(_rid), int(_bid), "overlap", frac=float(_fr), dist=0.0)

    # ---- (2) nearest-within-threshold fallback -------------------------------
    todo = rcn.index[rcn["match_type"] == "unmatched"]
    if len(todo):
        need = gpd.GeoDataFrame(
            {"_rid": todo.to_numpy(), "geometry": rcn.geometry.loc[todo].to_numpy()},
            crs=EPSG_PL)
        rr = right.copy()
        nn = gpd.sjoin_nearest(need, rr, how="inner",
                               max_distance=snap_m, distance_col="_d")
        nn = nn.sort_values("_d").drop_duplicates("_rid", keep="first")
        for _rid, _bid, _d in zip(nn["_rid"], nn["_bid"], nn["_d"]):
            _assign(int(_rid), int(_bid), "nearest", dist=float(_d))

    rcn["bdot_floors"] = pd.to_numeric(rcn["bdot_floors"], errors="coerce")
    rcn["gross_area_m2"] = rcn["footprint_m2"] * rcn["bdot_floors"]
    return rcn


# ==============================================================================
# BDOT loader wrapper (per powiat)
# ==============================================================================
def load_bdot_powiat(powiat: str, cache_dir: Path):
    """Return BDOT10k buildings GeoDataFrame for a powiat, or None on failure."""
    if _bdot_load_buildings is None:
        raise RuntimeError("bdot10k_query.load_buildings not importable "
                           "(place bdot10k_query.py next to this script)")
    woj = str(powiat)[:2]
    return _bdot_load_buildings(str(powiat), woj, Path(cache_dir))


# ==============================================================================
# Per-powiat worker
# ==============================================================================
def _worker(task: dict) -> dict:
    """Process one powiat end to end; write a shard; return summary + calib samples."""
    import geopandas as gpd
    import pyogrio

    powiat = task["powiat"]
    src = task["src"]
    layer = task["layer"]
    teryt_col = task["teryt_col"]
    cache_dir = Path(task["cache_dir"])
    shard_dir = Path(task["shard_dir"])
    powuzyt_col = task["powuzyt_col"]
    is_point = task["is_point"]
    min_frac = task["min_overlap_frac"]
    snap_m = task["snap_m"]

    res = {"powiat": powiat, "n": 0, "matched": 0, "shard": None,
           "ratios": np.empty(0), "error": None}
    try:
        where = f"{teryt_col} IS NULL" if powiat is None else f"{teryt_col}='{powiat}'"
        rcn = pyogrio.read_dataframe(src, layer=layer, where=where)
        if len(rcn) == 0:
            return res
        if rcn.crs is None:
            rcn = rcn.set_crs(EPSG_PL)
        elif rcn.crs.to_epsg() != EPSG_PL:
            rcn = rcn.to_crs(EPSG_PL)
        res["n"] = len(rcn)

        # BDOT for this powiat (None powiat -> cannot resolve -> all unmatched)
        bdot = None
        if powiat is not None:
            try:
                bdot = load_bdot_powiat(powiat, cache_dir)
            except Exception as e:              # download/parse failure
                res["error"] = f"bdot_load: {e}"

        if is_point:
            enriched = _match_points(rcn, bdot, snap_m)
        else:
            enriched = match_max_overlap(rcn, bdot, min_frac, snap_m)

        res["matched"] = int(enriched["bdot_floors"].notna().sum())

        # calibration samples: usable / gross where both available (buildings only)
        if (not is_point) and powuzyt_col in enriched.columns:
            pu = pd.to_numeric(enriched[powuzyt_col], errors="coerce")
            g = enriched["gross_area_m2"]
            ratio = (pu / g).to_numpy()
            ratio = ratio[np.isfinite(ratio) & (ratio > 0.3) & (ratio < 1.2)]
            res["ratios"] = ratio

        shard = shard_dir / f"{powiat if powiat is not None else 'NULL'}.gpkg"
        pyogrio.write_dataframe(enriched, shard, layer=layer)
        res["shard"] = str(shard)
    except Exception as e:
        res["error"] = f"{e}\n{traceback.format_exc()}"
    return res


def _match_points(pts, bdot, snap_m):
    """Attach BDOT storeys/function to flat POINTS (point-in-polygon, nearest fb)."""
    import geopandas as gpd
    pts = pts.reset_index(drop=True).copy()
    pts["bdot_floors"] = np.nan
    pts["bdot_function_general"] = None
    pts["bdot_function_detailed"] = None
    pts["bdot_building_id"] = None
    pts["match_type"] = "unmatched"
    pts["match_dist_m"] = np.nan
    if bdot is None or len(bdot) == 0:
        return pts
    b = bdot.reset_index(drop=True)
    within = gpd.sjoin(pts[["geometry"]].assign(_rid=np.arange(len(pts))),
                       b.assign(_bid=np.arange(len(b))),
                       predicate="within", how="inner").drop_duplicates("_rid")
    for _rid, _bid in zip(within["_rid"], within["_bid"]):
        bi = b.iloc[int(_bid)]
        pts.loc[int(_rid), ["bdot_floors", "bdot_function_general",
                            "bdot_function_detailed", "bdot_building_id",
                            "match_type", "match_dist_m"]] = [
            bi.get("floors_above"), bi.get("function_general"),
            bi.get("function_detailed"), bi.get("building_id"), "within", 0.0]
    todo = pts.index[pts["match_type"] == "unmatched"]
    if len(todo):
        need = gpd.GeoDataFrame({"_rid": todo.to_numpy(),
                                 "geometry": pts.geometry.loc[todo].to_numpy()},
                                crs=EPSG_PL)
        nn = gpd.sjoin_nearest(need, b.assign(_bid=np.arange(len(b))), how="inner",
                               max_distance=snap_m, distance_col="_d") \
            .sort_values("_d").drop_duplicates("_rid")
        for _rid, _bid, _d in zip(nn["_rid"], nn["_bid"], nn["_d"]):
            bi = b.iloc[int(_bid)]
            pts.loc[int(_rid), ["bdot_floors", "bdot_function_general",
                                "bdot_function_detailed", "bdot_building_id",
                                "match_type", "match_dist_m"]] = [
                bi.get("floors_above"), bi.get("function_general"),
                bi.get("function_detailed"), bi.get("building_id"), "nearest",
                float(_d)]
    pts["bdot_floors"] = pd.to_numeric(pts["bdot_floors"], errors="coerce")
    return pts


# ==============================================================================
# Orchestration
# ==============================================================================
def list_powiats(src: str, layer: str, teryt_col: str, only=None):
    import pyogrio
    df = pyogrio.read_dataframe(src, layer=layer, columns=[teryt_col],
                                read_geometry=False)
    vals = pd.unique(df[teryt_col])
    powiats = [None if (v is None or (isinstance(v, float) and np.isnan(v)))
               else str(v) for v in vals]
    if only:
        keep = set(only)
        powiats = [p for p in powiats if p in keep]
    # deterministic order, None last
    return sorted([p for p in powiats if p is not None]) + \
           ([None] if None in powiats else [])


def calibrate_k(ratio_arrays, fallback, min_n=100):
    """Return (k, n_samples, calibrated?). Robust median of usable/gross ratios."""
    allr = np.concatenate([r for r in ratio_arrays if len(r)]) if ratio_arrays else np.empty(0)
    if len(allr) >= min_n:
        return float(np.median(allr)), len(allr), True
    return float(fallback), len(allr), False


def assemble(shards, out_path, layer, k):
    """Stream shards into the final GPKG, adding usable_area_est_m2 = gross * k."""
    import pyogrio
    out_path = Path(out_path)
    if out_path.exists():
        out_path.unlink()
    first = True
    total = 0
    for shard in shards:
        if shard is None:
            continue
        gdf = pyogrio.read_dataframe(shard)
        if "gross_area_m2" in gdf.columns:
            gdf["usable_area_est_m2"] = gdf["gross_area_m2"] * k
        # residential flag from BDOT function or RCN kind
        fg = gdf.get("bdot_function_general")
        fd = gdf.get("bdot_function_detailed")
        flag = pd.Series(False, index=gdf.index)
        for col in (fg, fd):
            if col is not None:
                flag = flag | col.astype(str).str.lower().str.contains(RESIDENTIAL_SUBSTR)
        if "bud_rodzaj" in gdf.columns:
            flag = flag | gdf["bud_rodzaj"].astype(str).str.lower().eq("mieszkalny")
        gdf["is_residential"] = flag
        pyogrio.write_dataframe(gdf, out_path, layer=layer,
                                append=not first)
        first = False
        total += len(gdf)
    return total


def main() -> int:
    ap = argparse.ArgumentParser(description="Enrich RCN budynki with BDOT10k storeys")
    ap.add_argument("--budynki", default="data/raw/floorspace/budynki.gpkg")
    ap.add_argument("--layer", default="budynki")
    ap.add_argument("--teryt-col", default="teryt")
    ap.add_argument("--powuzyt-col", default="bud_pow_uzyt")
    ap.add_argument("--out", default=None,
                    help="default: <budynki dir>/budynki_bdot10k.gpkg")
    ap.add_argument("--cache-dir", default="data/processed/bdot10k/cache")
    ap.add_argument("--shard-dir", default=None, help="temp dir for per-powiat shards")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--min-overlap-frac", type=float, default=DEFAULT_MIN_OVERLAP_FRAC)
    ap.add_argument("--snap-m", type=float, default=DEFAULT_SNAP_M)
    ap.add_argument("--usable-factor", type=float, default=DEFAULT_USABLE_FACTOR)
    ap.add_argument("--calibrate", action="store_true",
                    help="estimate usable factor k from rows that report bud_pow_uzyt")
    ap.add_argument("--only-powiats", default=None,
                    help="comma-separated TERYT codes (subset run)")
    ap.add_argument("--lokale", default=None,
                    help="path to lokale.gpkg to also enrich flats with building context")
    ap.add_argument("--lokale-layer", default="lokale")
    args = ap.parse_args()

    only = [s.strip() for s in args.only_powiats.split(",")] if args.only_powiats else None

    # ------------------------------------------------------------------ buildings
    src = args.budynki
    is_point = False
    layer = args.layer
    out = Path(args.out) if args.out else Path(src).with_name("budynki_bdot10k.gpkg")
    cache_dir = Path(args.cache_dir); cache_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = Path(args.shard_dir) if args.shard_dir else out.with_suffix("").parent / "_shards_budynki"
    shard_dir.mkdir(parents=True, exist_ok=True)

    powiats = list_powiats(src, layer, args.teryt_col, only)
    print(f"[budynki] {len(powiats)} powiats to process on {args.workers} workers")

    tasks = [dict(powiat=p, src=src, layer=layer, teryt_col=args.teryt_col,
                  cache_dir=str(cache_dir), shard_dir=str(shard_dir),
                  powuzyt_col=args.powuzyt_col, is_point=is_point,
                  min_overlap_frac=args.min_overlap_frac, snap_m=args.snap_m)
             for p in powiats]

    results = _run(tasks, args.workers)

    # choose k
    if args.calibrate:
        k, nk, ok = calibrate_k([r["ratios"] for r in results], args.usable_factor)
        if ok:
            print(f"[calibration] k_hat = {k:.3f}  (empirical median of "
                  f"bud_pow_uzyt / gross_area over n={nk:,} rows)")
        else:
            print(f"[calibration] only n={nk:,} usable rows (< 100) -> "
                  f"falling back to fixed k = {k}")
    else:
        k = args.usable_factor
        print(f"[factor] using fixed usable-factor k = {k}")

    shards = [r["shard"] for r in results]
    n_out = assemble(shards, out, layer, k)

    n_tot = sum(r["n"] for r in results)
    n_match = sum(r["matched"] for r in results)
    errs = [(r["powiat"], r["error"]) for r in results if r["error"]]
    print("=" * 70)
    print(f"[budynki] rows written : {n_out:,}")
    print(f"[budynki] storeys matched: {n_match:,}/{n_tot:,} "
          f"({100*n_match/max(n_tot,1):.1f}%)")
    if errs:
        print(f"[budynki] {len(errs)} powiats had BDOT issues (marked unmatched), e.g.:")
        for p, e in errs[:5]:
            print(f"    {p}: {str(e).splitlines()[0]}")
    print(f"[budynki] OUTPUT -> {out.resolve()}")

    # ------------------------------------------------------------------ lokale (opt)
    if args.lokale:
        lsrc = args.lokale
        lout = Path(lsrc).with_name("lokale_bdot10k.gpkg")
        lshard = out.with_suffix("").parent / "_shards_lokale"
        lshard.mkdir(parents=True, exist_ok=True)
        lpow = list_powiats(lsrc, args.lokale_layer, args.teryt_col, only)
        print(f"\n[lokale] {len(lpow)} powiats (flats keep lok_pow_uzyt; BDOT floors added as context)")
        ltasks = [dict(powiat=p, src=lsrc, layer=args.lokale_layer,
                       teryt_col=args.teryt_col, cache_dir=str(cache_dir),
                       shard_dir=str(lshard), powuzyt_col="lok_pow_uzyt",
                       is_point=True, min_overlap_frac=args.min_overlap_frac,
                       snap_m=args.snap_m) for p in lpow]
        lres = _run(ltasks, args.workers)
        ln = assemble([r["shard"] for r in lres], lout, args.lokale_layer, k=np.nan)
        lm = sum(r["matched"] for r in lres); lt = sum(r["n"] for r in lres)
        print(f"[lokale] rows {ln:,} | building matched {lm:,}/{lt:,} "
              f"({100*lm/max(lt,1):.1f}%) -> {lout.resolve()}")

    return 0


def _run(tasks, workers):
    if workers and workers > 1 and len(tasks) > 1:
        with mp.Pool(processes=workers) as pool:
            out = []
            for i, r in enumerate(pool.imap_unordered(_worker, tasks), 1):
                tag = r["powiat"]
                print(f"  [{i}/{len(tasks)}] powiat {tag}: "
                      f"{r['matched']}/{r['n']} matched"
                      + (f"  ERROR {str(r['error']).splitlines()[0]}" if r["error"] else ""))
                out.append(r)
            return out
    return [_worker(t) for t in tasks]


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)  # safe with GDAL across platforms
    raise SystemExit(main())