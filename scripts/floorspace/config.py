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
BUDYNKI_GPKG = FLOORSPACE_DIR / "budynki.gpkg"
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
