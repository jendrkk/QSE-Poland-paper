#!/usr/bin/env python3
"""
config.py
=========

Central configuration for the gmina-level residential floorspace price index
pipeline (``scripts/floorspace``).

Holds repository-relative default paths, coordinate reference systems, the
column roles for each RCN layer, the categorical code sets used for filtering,
and the default cleaning / outlier thresholds. Every value here can be
overridden from the command line in ``run_index.py``; this module only supplies
defaults and shared constants so the rest of the package stays declarative.

Nothing in this module has side effects or heavy imports, so it is safe to
import anywhere (including in worker processes).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# Repository-relative default paths
# --------------------------------------------------------------------------- #
REPO_ROOT = Path(__file__).resolve().parents[2]

FLOORSPACE_DIR = REPO_ROOT / "data" / "raw" / "floorspace"
LOKALE_GPKG = FLOORSPACE_DIR / "lokale.gpkg"
BUDYNKI_GPKG = FLOORSPACE_DIR / "budynki.gpkg"            # legacy (superseded)
BUDYNKI_BDOT_GPKG = FLOORSPACE_DIR / "budynki_bdot10k.gpkg"          # houses (BDOT10k floors -> usable area)
BUDYNKI_DZIALKI_GPKG = FLOORSPACE_DIR / "budynki_dzialki_bdot10k.gpkg"  # building<->parcel bridge
DZIALKI_GPKG = FLOORSPACE_DIR / "dzialki.gpkg"
GUS_MEDIAN_CSV = FLOORSPACE_DIR / "GUS_P3787_median_price_residential_floorspace_1m2.csv"
GUS_MEAN_CSV = FLOORSPACE_DIR / "GUS_P3788_mean_price_residential_floorspace_1m2.csv"

COMMUNES_GPKG = REPO_ROOT / "data" / "processed" / "shapefiles" / "communes_2021.gpkg"
COMMUNES_ID_COL = "JPT_KOD_JE"          # 7-digit TERYT gmina code
COMMUNES_POP_COL = "pop"

TERC_2021_CSV = REPO_ROOT / "data" / "raw" / "teryt" / "TERC_Urzedowy_2021-01-01.csv"

OUTPUT_DIR = REPO_ROOT / "data" / "processed" / "floorspace"
OUTPUT_STEM = "gmina_floorspace_index_2021"

# --------------------------------------------------------------------------- #
# Coordinate reference systems
# --------------------------------------------------------------------------- #
CRS_RCN = "EPSG:2180"       # RCN geometries (PL-1992)
CRS_COMMUNES = "EPSG:3035"  # communes_2021.gpkg as stored (LAEA Europe)
CRS_WORK = "EPSG:2180"      # working CRS for the spatial join (join in metres, PL grid)

# --------------------------------------------------------------------------- #
# Static index reference year
# --------------------------------------------------------------------------- #
REFERENCE_YEAR = 2021

# --------------------------------------------------------------------------- #
# Categorical code sets (RCN controlled vocabularies)
# --------------------------------------------------------------------------- #
TRANS_ARM_LENGTH = "wolnyRynek"                    # keep only free-market deals
MARKET_PRIMARY = "pierwotny"
MARKET_SECONDARY = "wtorny"
PUBLIC_SELLERS = ("jednostkaSamorzaduTerytorialnego", "skarbPanstwa")

# lokale: residential function
LOK_FUNC_RESIDENTIAL = "mieszkalna"

# budynki: residential whole-house sale
BUD_HOUSE_NIER = "nieruchomoscGruntowaZabudowana"  # a building on its own plot
BUD_HOUSE_RODZAJ = "mieszkalny"
BUD_FLAT_FOOTPRINT_NIER = "nieruchomoscLokalowa"   # footprint of a flat sale -> discard

# BDOT10k enrichment (budynki_bdot10k / budynki_dzialki)
BDOT_MATCH_OVERLAP = "overlap"     # reliable spatial match; "nearest" is noisy -> excluded by default
BDOT_USABLE_COL = "usable_area_est_m2"   # = footprint_m2 * bdot_floors * (~0.73 usable/gross factor)

# dzialki: land property kinds
DZI_UNDEVELOPED = "nieruchomoscGruntowaNiezabudowana"
DZI_DEVELOPED = "nieruchomoscGruntowaZabudowana"
# residential zoning / use signals used to keep "residential land"
DZI_RESIDENTIAL_ZONING = (
    "budownictwoMieszkanioweJednorodzinne",
    "budownictwoMieszkanioweWielorodzinne",
)
DZI_URBANISED_USE = "gruntyZabudowaneIZurbanizowane"

# ownership share values treated as "whole unit"
UDZIAL_WHOLE = "1/1"


@dataclass
class CleaningConfig:
    """Tunable cleaning / outlier thresholds. All overridable via CLI."""

    year_min: int = 2006
    year_max: int = 2026
    # hard domain gates
    ppm2_min: float = 1_000.0        # zl / m2
    ppm2_max: float = 50_000.0
    flat_area_min: float = 15.0      # m2
    flat_area_max: float = 400.0
    house_area_min: float = 25.0
    house_area_max: float = 1_000.0
    house_plot_min: float = 50.0     # m2 of plot for a house (sanity)
    house_plot_max: float = 20_000.0
    # robust within-stratum trim
    robust_z: float = 4.0            # |z| on log(price/m2), median/MAD scale
    robust_min_stratum: int = 30     # else fall back to coarser stratum
    # exclude public/discounted sellers as robustness (kept by default: arm's length already filters)
    drop_public_sellers: bool = False

    # --- BDOT10k house (self-constructed usable area) quality gates ---
    house_use_nearest: bool = False   # include match_type='nearest' (noisy: outbuildings) -- off
    house_min_overlap: float = 0.30   # min match_overlap_frac for overlap matches
    house_min_floors: int = 1
    house_max_floors: int = 5         # >=6 are mismatches / not single/low-multi-family houses
    house_usable_min: float = 25.0    # m2 of estimated usable area
    house_usable_max: float = 1000.0

    # --- land (dzialki) quality + netting ---
    land_area_source: str = "geometry"   # {"geometry","dzi_pow_ewid"}; geometry avoids ha unit errors
    land_ha_ratio_max: float = 0.02      # drop rows where recorded/geom area ratio < this (ha-coded)
    land_ppm2_min: float = 1.0
    land_ppm2_max: float = 20_000.0
    land_min_txn_gmina: int = 15         # min undeveloped-land txns for a gmina land price; else fall back
    # land netting for houses: floorspace_price = (P_total - p_land*plot)/usable
    house_land_netting: str = "subtract"   # {"subtract","regress","none"}
    land_share_cap: float = 0.60           # cap land value at this fraction of the transaction price
    house_ppm2_min: float = 500.0          # post-netting structure price/m2 bounds
    house_ppm2_max: float = 40_000.0
    # marginal rural houses recovered from dzialki+budynki_dzialki
    use_dzialki_marginal_houses: bool = True


@dataclass
class ModelConfig:
    """Hedonic + small-area model options."""

    time_fe: str = "year"            # {"year", "woj_year"}
    cluster_se: bool = True          # cluster hedonic SEs by gmina
    ridge: float = 1e-6              # tiny ridge for the FE normal equations
    min_txn_direct: int = 5          # gmina needs >= this many txns for a "direct" estimate
    bayes: bool = False              # full MCMC area model (numpyro) if available
    benchmark_to_gus: bool = False   # exact within-powiat rescale to GUS after shrinkage
    anchor: str = "median"           # GUS anchor used as prior mean {"median","mean"}


# GUS long-format dimension labels (as they appear in the BDL header)
GUS_MARKET_TOTAL = "ogółem"
GUS_SIZE_TOTAL = "ogółem"
GUS_SUPPRESSED = 0.0                 # BDL uses 0 for suppressed/missing -> treat as NaN
