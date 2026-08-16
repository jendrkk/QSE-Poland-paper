#!/usr/bin/env python3
"""
houses.py
=========

Construct house (single- and low-multi-family) floorspace observations from the
BDOT10k-enriched building data, and net out the plot-land value so that the
resulting price/m2 measures *enclosed living space*, comparable to flats.

Two sources of house transactions (deduplicated by ``tran_lokalny_id_iip``):

1.  PRIMARY -- ``budynki_bdot10k.gpkg``: RCN whole-house sales
    (``nier_rodzaj = nieruchomoscGruntowaZabudowana``) whose building footprints
    were matched to BDOT10k buildings. Usable area is estimated as
    ``footprint_m2 * bdot_floors * ~0.73``. We keep only reliable ``overlap``
    matches (the ``nearest`` matches systematically grab small outbuildings) with
    plausible floor counts, and sum usable area over the residential buildings of
    each transaction.

2.  MARGINAL -- ``dzialki.gpkg`` (built-up parcels priced, not already in the
    primary set) joined to ``budynki_dzialki_bdot10k.gpkg`` for the residential
    usable area on that parcel. This recovers ~85k additional rural sales that the
    primary route misses. Plot area comes from the parcel POLYGON geometry
    (reliable m^2), not the ha-contaminated attribute fields.

Land netting (``net_land``):
    structure_price = P_total - p_land_local * plot_area   (p_land capped share)
    price_per_m2    = structure_price / usable_area
where ``p_land_local`` is the hierarchical (gmina->powiat->woj->national) median
undeveloped residential-land price from ``build_land_surface``.

All quality gates are configurable and audited (attrition logged at each step).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from config import (
    CleaningConfig,
    BUD_HOUSE_NIER,
    BUD_HOUSE_RODZAJ,
    BDOT_MATCH_OVERLAP,
    TRANS_ARM_LENGTH,
    DZI_DEVELOPED,
)
import rcn

LOGGER = logging.getLogger("floorspace.houses")


# --------------------------------------------------------------------------- #
# 1. PRIMARY house transactions from budynki_bdot10k
# --------------------------------------------------------------------------- #
def build_houses_primary(cfg: CleaningConfig, workers: int, sample_frac=None):
    from config import BUDYNKI_BDOT_GPKG

    cols = [
        "teryt", "tran_rodzaj_rynku", "dok_data",
        "nier_cena_brutto", "tran_cena_brutto", "nier_pow_gruntu",
        "usable_area_est_m2", "bdot_floors", "footprint_m2",
        "match_overlap_frac", "tran_lokalny_id_iip", "bud_id_budynku",
    ]
    where = (
        f"nier_rodzaj='{BUD_HOUSE_NIER}' AND is_residential=1 "
        f"AND match_type='{BDOT_MATCH_OVERLAP}' AND usable_area_est_m2 IS NOT NULL "
        f"AND tran_rodzaj_trans='{TRANS_ARM_LENGTH}' "
        "AND (nier_udzial='1/1' OR nier_udzial IS NULL) "
        f"AND bdot_floors>={cfg.house_min_floors} AND bdot_floors<={cfg.house_max_floors} "
        f"AND match_overlap_frac>={cfg.house_min_overlap}"
    )
    df = rcn.read_layer(BUDYNKI_BDOT_GPKG, "budynki", cols, where, "polygon", workers)
    if sample_frac:
        df = df.sample(frac=sample_frac, random_state=0)
    if not len(df):
        return _empty_house()

    for c in ["usable_area_est_m2", "bdot_floors", "footprint_m2", "match_overlap_frac"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["nier_num"] = rcn._to_num(df["nier_cena_brutto"])
    df["tran_num"] = rcn._to_num(df["tran_cena_brutto"])
    df["plot_num"] = rcn._to_num(df["nier_pow_gruntu"])
    df["year"] = rcn._year_from_dokdata(df["dok_data"])
    df["market"] = rcn.market_category(df["tran_rodzaj_rynku"])
    df = df[df["usable_area_est_m2"] > 0]

    # representative point = centroid of the largest building of each transaction
    df = df.sort_values("usable_area_est_m2", ascending=False)
    first = df.groupby("tran_lokalny_id_iip", sort=False).first()
    agg = df.groupby("tran_lokalny_id_iip", sort=False).agg(
        usable_area=("usable_area_est_m2", "sum"),
        bld_floors=("bdot_floors", "max"),
        footprint=("footprint_m2", "sum"),
        overlap_min=("match_overlap_frac", "min"),
    )
    h = agg.join(first[["nier_num", "tran_num", "plot_num", "market", "year",
                        "teryt", "x2180", "y2180", "bud_id_budynku"]])
    # transaction price: property price (nier); fall back to whole-transaction price
    single = df.groupby("tran_lokalny_id_iip", sort=False).size().eq(1)
    price = h["nier_num"].where(h["nier_num"] > 0)
    price = price.where(price.notna(), h["tran_num"].where(single.reindex(h.index)))
    h["price"] = price
    h = h.reset_index().rename(columns={"teryt": "powiat_teryt", "bud_id_budynku": "unit_id"})
    h["source"] = "house_budynki"
    # overlap-based quality weight in (0.5, 1]
    h["qweight"] = np.clip(h["overlap_min"].fillna(0.7), 0.5, 1.0)
    h = _plot_ha_fix(h, cfg)
    LOGGER.info("HOUSES primary (budynki_bdot10k): %s transactions", f"{len(h):,}")
    return _house_frame(h)


# --------------------------------------------------------------------------- #
# 2. MARGINAL house transactions from dzialki + budynki_dzialki
# --------------------------------------------------------------------------- #
def build_houses_marginal(cfg: CleaningConfig, workers: int, exclude_tran_ids, sample_frac=None):
    from config import DZIALKI_GPKG, BUDYNKI_DZIALKI_GPKG

    if not cfg.use_dzialki_marginal_houses:
        return _empty_house()

    # 2a. residential usable area per parcel from the building<->parcel bridge
    bcols = ["dzi_id_dzialki", "usable_area_est_m2", "bdot_floors", "overlap_frac", "is_residential"]
    bwhere = (
        f"is_residential=1 AND usable_area_est_m2 IS NOT NULL "
        f"AND bdot_floors>={cfg.house_min_floors} AND bdot_floors<={cfg.house_max_floors}"
    )
    bridge = rcn.read_layer(BUDYNKI_DZIALKI_GPKG, "budynki_dzialki", bcols, bwhere, "point", workers)
    for c in ["usable_area_est_m2", "bdot_floors"]:
        bridge[c] = pd.to_numeric(bridge[c], errors="coerce")
    parcel = bridge.groupby("dzi_id_dzialki").agg(
        usable_area=("usable_area_est_m2", "sum"),
        bld_floors=("bdot_floors", "max"),
        n_bld=("usable_area_est_m2", "size"),
    ).reset_index()
    LOGGER.info("  bridge: %s parcels with residential usable area", f"{len(parcel):,}")

    # 2b. built-up priced parcel transactions (with parcel geometry -> plot area)
    # NOTE: dzialki.gpkg has NO tran_cena_brutto; prices are nier_cena_brutto /
    # dzi_cena_brutto only. Use the property price (nier), fall back to the parcel
    # component price (dzi).
    dcols = [
        "teryt", "tran_rodzaj_rynku", "dok_data", "nier_cena_brutto", "dzi_cena_brutto",
        "nier_pow_gruntu", "dzi_id_dzialki", "tran_lokalny_id_iip",
    ]
    dwhere = (
        f"nier_rodzaj='{DZI_DEVELOPED}' AND tran_rodzaj_trans='{TRANS_ARM_LENGTH}' "
        "AND (nier_udzial='1/1' OR nier_udzial IS NULL) "
        "AND nier_cena_brutto IS NOT NULL AND nier_cena_brutto<>''"
    )
    dz = rcn.read_layer(DZIALKI_GPKG, "dzialki", dcols, dwhere, "polygon", workers)
    if sample_frac:
        dz = dz.sample(frac=sample_frac, random_state=0)
    dz["geom_area_m2"] = pd.to_numeric(dz["geom_area_m2"], errors="coerce")
    # drop transactions already captured by the primary (budynki) route
    excl = set(exclude_tran_ids)
    dz = dz[~dz["tran_lokalny_id_iip"].isin(excl)]
    dz = dz.merge(parcel[["dzi_id_dzialki", "usable_area", "bld_floors"]], on="dzi_id_dzialki", how="inner")

    dz["price"] = rcn._to_num(dz["nier_cena_brutto"])
    dz["price"] = dz["price"].where(dz["price"] > 0, rcn._to_num(dz["dzi_cena_brutto"]))
    dz["year"] = rcn._year_from_dokdata(dz["dok_data"])
    dz["market"] = rcn.market_category(dz["tran_rodzaj_rynku"])
    dz["plot_num"] = dz["geom_area_m2"]                     # reliable m^2 parcel area
    dz["footprint"] = np.nan
    dz = dz.rename(columns={"teryt": "powiat_teryt", "dzi_id_dzialki": "unit_id"})
    # one row per transaction: keep the parcel with the largest residential area
    dz = dz.sort_values("usable_area", ascending=False).drop_duplicates(subset="tran_lokalny_id_iip")
    dz["source"] = "house_dzialki"
    dz["qweight"] = 0.7                                     # marginal source: modest down-weight
    LOGGER.info("HOUSES marginal (dzialki+bridge): %s transactions", f"{len(dz):,}")
    return _house_frame(dz)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _plot_ha_fix(h, cfg: CleaningConfig):
    """Correct ha-coded nier_pow_gruntu using the building footprint: a plot
    smaller than its own footprint is ha-coded -> x10000."""
    plot = h["plot_num"].to_numpy(float)
    fp = h["footprint"].to_numpy(float)
    bad = np.isfinite(plot) & np.isfinite(fp) & (plot > 0) & (plot < fp)
    plot = np.where(bad, plot * 10_000.0, plot)
    h["plot_num"] = plot
    return h


def _house_frame(h):
    h["property_type"] = "house"
    h["rooms"] = np.nan
    h["floor"] = np.nan                     # flat-storey concept; N/A for houses
    h["plot_area"] = h["plot_num"] if "plot_num" in h else np.nan
    h["ppm2"] = np.nan                      # filled after land netting
    h["area"] = h["usable_area"]
    return h


def _empty_house():
    return pd.DataFrame(columns=[
        "tran_lokalny_id_iip", "usable_area", "bld_floors", "price", "plot_area",
        "market", "year", "powiat_teryt", "x2180", "y2180", "unit_id", "source",
        "qweight", "property_type", "rooms", "floor", "area", "ppm2",
    ])


def house_tran_ids_primary(houses_primary):
    return set(houses_primary["tran_lokalny_id_iip"].dropna().unique())


# --------------------------------------------------------------------------- #
# 3. local land-price surface (also the SAE log_land covariate)
# --------------------------------------------------------------------------- #
def build_land_surface(land_assigned: pd.DataFrame, all_gminas: pd.Series, cfg: CleaningConfig):
    """Hierarchical median undeveloped residential-land price per gmina.

    land_assigned : land points already tagged with ``gmina_teryt``.
    all_gminas    : Series of every gmina_teryt (so all get a value).
    Returns a DataFrame [gmina_teryt, land_ppm2, log_land, n_land, land_level]
    where the price is filled gmina -> powiat -> woj -> national.
    """
    la = land_assigned[land_assigned["gmina_teryt"].notna()].copy()
    la["powiat"] = la["gmina_teryt"].str[:4]
    la["woj"] = la["gmina_teryt"].str[:2]

    g = la.groupby("gmina_teryt")["land_ppm2"].agg(["median", "size"])
    g_ok = g[g["size"] >= cfg.land_min_txn_gmina]["median"]
    p_med = la.groupby("powiat")["land_ppm2"].median()
    w_med = la.groupby("woj")["land_ppm2"].median()
    nat = la["land_ppm2"].median()

    out = pd.DataFrame({"gmina_teryt": all_gminas.astype(str).unique()})
    out["powiat"] = out["gmina_teryt"].str[:4]
    out["woj"] = out["gmina_teryt"].str[:2]
    out["n_land"] = out["gmina_teryt"].map(g["size"]).fillna(0).astype(int)
    lvl = np.where(out["gmina_teryt"].isin(g_ok.index), "gmina",
          np.where(out["powiat"].isin(p_med.index), "powiat",
          np.where(out["woj"].isin(w_med.index), "woj", "national")))
    val = (out["gmina_teryt"].map(g_ok)
           .fillna(out["powiat"].map(p_med))
           .fillna(out["woj"].map(w_med))
           .fillna(nat))
    out["land_ppm2"] = val
    out["land_level"] = lvl
    out["log_land"] = np.log(out["land_ppm2"].clip(lower=1))
    LOGGER.info("Land surface: gmina-level for %d, powiat fallback %d, woj/nat %d",
                int((lvl == "gmina").sum()), int((lvl == "powiat").sum()),
                int(((lvl == "woj") | (lvl == "national")).sum()))
    return out[["gmina_teryt", "land_ppm2", "log_land", "n_land", "land_level"]]


# --------------------------------------------------------------------------- #
# 4. land netting -> structure price/m2
# --------------------------------------------------------------------------- #
def net_land(houses: pd.DataFrame, land_surface: pd.DataFrame, cfg: CleaningConfig):
    """Return houses with ``ppm2`` = structure price per m2 of usable area.

    method 'subtract' : (P - min(p_land*plot, cap*P)) / usable
    method 'none'     : P / usable (plot handled as a hedonic covariate instead)
    method 'regress'  : P / usable, and plot_area is added as a regressor upstream
    """
    h = houses[houses["gmina_teryt"].notna()].copy()
    n0 = len(h)
    h = h.merge(land_surface[["gmina_teryt", "land_ppm2"]], on="gmina_teryt", how="left")

    # impute missing plot with the powiat-median plot among houses
    h["powiat"] = h["gmina_teryt"].str[:4]
    plot = h["plot_area"].to_numpy(float)
    pmed = h.groupby("powiat")["plot_area"].transform("median")
    plot = np.where(np.isfinite(plot) & (plot > 0), plot, pmed)
    h["plot_used"] = plot

    price = h["price"].to_numpy(float)
    usable = h["area"].to_numpy(float)
    if cfg.house_land_netting == "subtract":
        land_val = h["land_ppm2"].to_numpy(float) * plot
        land_val = np.nan_to_num(land_val, nan=0.0)
        land_val = np.minimum(land_val, cfg.land_share_cap * price)
        structure = price - land_val
    else:  # 'none' or 'regress' -> no explicit subtraction
        structure = price
    ppm2 = structure / usable
    h["ppm2"] = ppm2

    m = (
        np.isfinite(ppm2)
        & (usable >= cfg.house_usable_min) & (usable <= cfg.house_usable_max)
        & (ppm2 >= cfg.house_ppm2_min) & (ppm2 <= cfg.house_ppm2_max)
        & (h["year"] >= cfg.year_min) & (h["year"] <= cfg.year_max)
    )
    out = h[m].copy()
    LOGGER.info("Land-netting (%s): %s -> %s houses (median structure zl/m2=%.0f)",
                cfg.house_land_netting, f"{n0:,}", f"{len(out):,}",
                float(np.nanmedian(out["ppm2"])) if len(out) else float("nan"))
    return out
