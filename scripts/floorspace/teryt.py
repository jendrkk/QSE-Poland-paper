#!/usr/bin/env python3
"""
teryt.py
========

Helpers for the 2021 TERYT administrative coding.

A 7-digit gmina code ``WWPPGG R`` encodes voivodeship (WW), powiat (PP), gmina
(GG) and the gmina *type* R as its final digit:

    1 = gmina miejska (urban)
    2 = gmina wiejska (rural)
    3 = gmina miejsko-wiejska (urban-rural)
    4 = miasto w gminie miejsko-wiejskiej
    5 = obszar wiejski w gminie miejsko-wiejskiej

Gmina type is a structural covariate for the small-area model (housing-stock
composition differs sharply by type). Names come from ``TERC_Urzedowy_*.csv``.
"""

from __future__ import annotations

import logging

import pandas as pd

LOGGER = logging.getLogger("floorspace.teryt")

RODZ_CLASS = {1: "urban", 2: "rural", 3: "urban_rural", 4: "urban", 5: "rural"}


def rodz_class_from_code(code: str) -> str:
    try:
        return RODZ_CLASS.get(int(str(code)[-1]), "rural")
    except (ValueError, IndexError):
        return "rural"


def load_terc(path) -> pd.DataFrame:
    """Load the TERC table -> [teryt7, rodz, rodz_class, nazwa] for gmina rows."""
    df = pd.read_csv(path, sep=";", dtype=str).fillna("")
    gm = df[(df["GMI"] != "") & (df["RODZ"] != "")].copy()
    gm["teryt7"] = gm["WOJ"].str.zfill(2) + gm["POW"].str.zfill(2) + gm["GMI"].str.zfill(2) + gm["RODZ"]
    gm["rodz"] = gm["RODZ"].astype(int)
    gm["rodz_class"] = gm["rodz"].map(RODZ_CLASS).fillna("rural")
    gm = gm.rename(columns={"NAZWA": "nazwa"})
    LOGGER.info("TERC loaded: %d gmina-level rows", len(gm))
    return gm[["teryt7", "rodz", "rodz_class", "nazwa"]]
