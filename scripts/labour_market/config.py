#!/usr/bin/env python3
"""
config.py
=========

Central configuration for the gmina-level **labour-market** input builder
(``scripts/labour_market/wages``): mean wages and employment counts, by place
of **work** and place of **residence**, harmonised onto the 2021 TERYT gmina
frame, for three cross-sections (2011, 2021, and a recent ~2026 window).

The outputs are the ``labor_tidy_<year>.csv`` tables that the MRRH pipeline
(``Topic_11/mrrh_pipeline/dataio.py``) consumes, one row per gmina with columns

    region_id, nazwa, powiat,
    median_income_workplace, median_income_residence,
    employment_workplace, employment_residence

plus provenance flags and a shared TERYT crosswalk artefact.

Design mirrors ``scripts/floorspace/config.py``: repository-relative default
paths, no side effects, everything overridable from ``run_labour.py``. See
``METHODOLOGY_labour_market_wages.md`` for the statistical rationale.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# Repository-relative default paths
# --------------------------------------------------------------------------- #
# scripts/labour_market/wages/config.py -> parents[3] == repo root.
# Override with QSE_REPO_ROOT for out-of-tree testing (staged data).
REPO_ROOT = Path(os.environ.get("QSE_REPO_ROOT", Path(__file__).resolve().parents[3]))

RAW = REPO_ROOT / "data" / "raw"
LM = RAW / "labour_market"
WAGES_DIR = LM / "wages"
TERYT_DIR = RAW / "teryt"
BILATERAL_DIR = RAW / "bilateral"

# --- wages ---------------------------------------------------------------- #
P2497_WAGE_POWIAT = WAGES_DIR / "GUS_P2497_mean_wage_counties_2002_2025.csv"          # workplace, yearly, powiat
P4609_WAGE_GMINA = WAGES_DIR / "GUS_P4609_mean_wage_communes_2024_2026.csv"           # both concepts, monthly, gmina

# --- employment (workplace concept) --------------------------------------- #
P2172_EMP_WORK = LM / "GUS_P2172_employment_counted_at_workplace_2010_2021.csv"       # gmina, yearly 2010-2021
P4508_EMP_WORK = LM / "GUS_P4508_employment_counted_at_workplace_2023_2025.csv"       # gmina, yearly 2023-2025

# --- employment (residence concept) --------------------------------------- #
P4280_EMP_RES = LM / "GUS_P4280_Employment_counted_at_residence_2022_2026.csv"        # gmina, monthly 2022-2026

# --- working-age population (residence), used to disaggregate residence emp #
P3457_WORKAGE_2011 = LM / "GUS_P3457_working_age_pop_counted_at_residence_2011.csv"   # gmina, 2011
P4362_WORKAGE_2021 = LM / "GUS_P4362_working_age_pop_counted_at_residence_2021.csv"   # gmina, 2021

# --- census income-earner control totals (powiat), OPTIONAL/auto-detected -- #
#   P3357 (NSP 2011) / P4488 (NSP 2021): population by main source of income,
#   powiat level. The "maintained by own work" category is the residence-
#   employment control total that gmina residence employment is raked to.
#   File names are matched by glob so the user can drop them in as downloaded.
CENSUS_INCOME_2011_GLOB = "GUS_P3357*"
CENSUS_INCOME_2021_GLOB = "GUS_P4488*"

# --- TERYT snapshots + change files --------------------------------------- #
TERC_2011 = TERYT_DIR / "TERC_Urzedowy_2011-01-01.csv"
TERC_2021 = TERYT_DIR / "TERC_Urzedowy_2021-01-01.csv"
TERC_2026 = TERYT_DIR / "TERC_Urzedowy_2026-01-01.csv"
TERC_CHANGES_11_21 = TERYT_DIR / "TERC_Urzedowy_zmiany_2011-01-01_2021-01-01.xml"
TERC_CHANGES_21_26 = TERYT_DIR / "TERC_Urzedowy_zmiany_2021-01-01_2026-01-01.xml"

# --- bilateral commuting flows (residence x workplace) -------------------- #
FLOWS_2011 = BILATERAL_DIR / "bilateral_matrix_2011.xls"      # long format
FLOWS_2021 = BILATERAL_DIR / "bilateral_matrix_2021.xlsx"     # long format, sheet "Macierz przepływów"

# --- outputs -------------------------------------------------------------- #
OUTPUT_DIR = REPO_ROOT / "data" / "processed" / "labour_market"
OUTPUT_STEM = "labor_tidy"                       # -> labor_tidy_2011.csv, ...
CROSSWALK_STEM = "teryt_gmina_crosswalk_2021"    # -> teryt_gmina_crosswalk_2021.csv

# --------------------------------------------------------------------------- #
# Cross-sections
# --------------------------------------------------------------------------- #
REFERENCE_TERYT_YEAR = 2021          # all data harmonised onto this gmina frame
TARGET_YEARS = (2011, 2021, 2026)    # 2026 == recent trailing-window snapshot

# Recent window: trailing mean ending at the freshest fully-populated P4609
# month. February 2026 is the last complete month in the current file.
RECENT_LABEL_YEAR = 2026
RECENT_END = (2026, 2)               # (year, month) of the freshest populated month
RECENT_WINDOW_MONTHS = 12            # trailing months averaged (CLI: --recent-window)

# --------------------------------------------------------------------------- #
# TERYT change special cases (see METHODOLOGY §crosswalk). 6-digit codes.
# All other codes map to themselves (identity). Everything here is derived
# from TERC snapshots + change XMLs and Polish administrative history, and is
# CLI-overridable via --ostrowice-split.
# --------------------------------------------------------------------------- #
# 2011 codes absent from 2021 -> {2021 target: weight}. Weights sum to 1;
# used for count allocation and employment-weighted wage folding.
CROSSWALK_SPECIAL_2011 = {
    "022109": {"026501": 1.0},                   # Wałbrzych: regained city-county status (2013)
    "080910": {"086201": 1.0},                   # Zielona Góra (rural) merged into the city (2015)
    "320304": {"320302": 1.0},                   # Ostrowice dissolved (2019) -> Drawsko Pomorskie (primary)
    #   default sends all of Ostrowice to Drawsko Pomorskie; a documented
    #   alternative split with Złocieniec (320306) is available via CLI.
}
# 2026 codes absent from 2021 (new gminas) -> fold back to 2021 parent.
CROSSWALK_SPECIAL_2026 = {
    "120713": {"120705": 1.0},                   # Szczawa carved from Kamienica (2025)
    "200216": {"200209": 1.0},                   # Grabówka carved from Supraśl (2025)
}
# Optional Ostrowice split (population-based; ~majority to Drawsko Pomorskie).
OSTROWICE_SPLIT = {"320302": 0.72, "320306": 0.28}

# --------------------------------------------------------------------------- #
# GUS BDL parsing constants (shared with scripts/floorspace/gus.py conventions)
# --------------------------------------------------------------------------- #
GUS_SEP = ";"
GUS_ENCODING = "utf-8-sig"
GUS_SUPPRESSED = {"", "0"}           # BDL suppression / true-zero -> NaN
MONTHS_PL = {
    "styczeń": 1, "luty": 2, "marzec": 3, "kwiecień": 4, "maj": 5, "czerwiec": 6,
    "lipiec": 7, "sierpień": 8, "wrzesień": 9, "październik": 10,
    "listopad": 11, "grudzień": 12,
}
# P4609 wage-concept dimension labels
WAGE_CONCEPT_WORKPLACE = "wg siedziby podmiotu"    # by entity seat == workplace
WAGE_CONCEPT_RESIDENCE = "wg miejsca zamieszkania"  # by place of residence

# Gmina RODZ (final TERYT digit) -> class. 1 urban, 2 rural, 3 urban-rural.
# 4 (miasto) / 5 (obszar wiejski) are *parts* of a type-3 gmina and are dropped
# from the gmina universe to avoid double counting.
RODZ_WHOLE_GMINA = {"1", "2", "3"}
RODZ_CLASS = {1: "urban", 2: "rural", 3: "urban_rural"}


@dataclass
class LabourConfig:
    """Tunable build options. All overridable from the CLI."""

    recent_window_months: int = RECENT_WINDOW_MONTHS
    recent_end: tuple = RECENT_END
    # wage disaggregation
    benchmark_recent_to_powiat: bool = False   # recent year trusts P4609 gmina truth; no rake by default
    struct_shrinkage: float = 0.90             # exponent λ on modern within-powiat wage ratios
    #   (1.0 = transfer the full observed dispersion; <1 shrinks toward the powiat mean)
    # residence-employment recovery
    residence_emp_method: str = "flows_then_rake"  # {"flows_then_rake","workage_rake","off"}
    ostrowice_split: bool = False              # split Ostrowice across two absorbers instead of primary-only
    # imputation
    impute_hierarchy: tuple = ("powiat", "woj", "national")
    # numerics
    ridge: float = 1e-6


@dataclass
class ModelConfig:
    """Fay-Herriot / empirical-Bayes options for the within-powiat transfer."""

    covariates: tuple = ("log_emp_work", "urban", "urban_rural")
    use_powiat_fe: bool = True
    fh_max_iter: int = 200
    fh_tol: float = 1e-8
