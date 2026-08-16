#!/usr/bin/env python3
"""
teryt.py
========

The 2021-anchored TERYT gmina crosswalk — the backbone that makes wages,
employment, commuting flows and population share one identical set of spatial
units.

A gmina is identified by its 6-digit code ``WWPPGG`` (voivodeship, powiat,
gmina), dropping the final RODZ digit. RODZ 4 (miasto) and RODZ 5 (obszar
wiejski) are *parts* of a type-3 (urban-rural) gmina and are never independent
units here.

Empirics (verified against the three TERC snapshots and the two change files):
at 6-digit level the frame is almost static — 2011 vs 2021 differ by three
retired codes, 2021 vs 2026 by two new codes, with **no genuine areal
splits/merges** in the change records (all ``Wyodrebniono/Wlaczono`` fields are
empty). Every difference is therefore a documented 1:1 recode, a whole-gmina
merger, one 2019 dissolution, or a post-2021 carve-out — enumerated in
``config.CROSSWALK_SPECIAL_*``. Because true splits are absent, no bilateral
flow needs to be *split*; the crosswalk is a many-to-one map onto the 2021
frame, which is exactly what keeps it compatible with the commuting matrices.

Public API
----------
load_terc(path) -> DataFrame[code6, rodz, rodz_class, nazwa, powiat, woj]
build_crosswalk(cfg) -> (crosswalk_df, ref21_df)
harmonise(df, code_col, value_cols, src_year, crosswalk, how, weight_col=None)
    Aggregate a per-gmina table onto the 2021 frame ("sum" for counts,
    "wmean" for intensive quantities such as wages).
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from config import RODZ_CLASS, RODZ_WHOLE_GMINA

LOGGER = logging.getLogger("labour.teryt")


# --------------------------------------------------------------------------- #
# Code helpers
# --------------------------------------------------------------------------- #
def code6(kod: str) -> str:
    """First 6 digits of a (7-digit) TERYT code, zero-padded."""
    s = "".join(ch for ch in str(kod) if ch.isdigit())
    return s[:6].zfill(6)


def rodz_of(kod: str) -> str:
    s = "".join(ch for ch in str(kod) if ch.isdigit())
    return s[6] if len(s) >= 7 else ""


def rodz_class_from_code(kod: str) -> str:
    try:
        return RODZ_CLASS.get(int(rodz_of(kod)), "rural")
    except (ValueError, IndexError):
        return "rural"


# --------------------------------------------------------------------------- #
# TERC snapshot loader
# --------------------------------------------------------------------------- #
def load_terc(path) -> pd.DataFrame:
    """Load a TERC snapshot -> one row per whole gmina (RODZ in {1,2,3})."""
    df = pd.read_csv(path, sep=";", dtype=str, encoding="utf-8-sig").fillna("")
    gm = df[(df["GMI"] != "") & (df["RODZ"].isin(RODZ_WHOLE_GMINA))].copy()
    gm["code6"] = (gm["WOJ"].str.zfill(2) + gm["POW"].str.zfill(2) + gm["GMI"].str.zfill(2))
    gm["rodz"] = gm["RODZ"].astype(int)
    gm["rodz_class"] = gm["rodz"].map(RODZ_CLASS).fillna("rural")
    gm["powiat"] = gm["code6"].str[:4]
    gm["woj"] = gm["code6"].str[:2]
    gm = gm.rename(columns={"NAZWA": "nazwa"})
    gm = gm.drop_duplicates("code6").reset_index(drop=True)
    LOGGER.info("TERC %s: %d whole gminas", getattr(path, "name", path), len(gm))
    return gm[["code6", "rodz", "rodz_class", "nazwa", "powiat", "woj"]]


# --------------------------------------------------------------------------- #
# Crosswalk
# --------------------------------------------------------------------------- #
def build_crosswalk(cfg, C):
    """Build the 2021-anchored crosswalk from all source vintages.

    Because the six mover codes are disjoint across vintages and none collides
    with a 2021 code, a single *universal* map resolves any file regardless of
    which TERYT vintage GUS applied on export.

    Returns
    -------
    universal : DataFrame[src_code, dst_code, weight]
        identity for every 2021 code, plus the 2011-only and 2026-only special
        mappings. weight sums to 1 within src_code. Used for all harmonisation.
    labelled : DataFrame[src_year, src_code, dst_code, weight]
        per-vintage view, for the human-readable crosswalk artefact only.
    ref21 : DataFrame  (load_terc of the 2021 snapshot; the target frame)
    """
    ref21 = load_terc(C.TERC_2021)
    R21 = set(ref21["code6"])

    special = {
        2011: dict(C.CROSSWALK_SPECIAL_2011),
        2026: dict(C.CROSSWALK_SPECIAL_2026),
    }
    if cfg.ostrowice_split:
        special[2011]["320304"] = dict(C.OSTROWICE_SPLIT)

    labelled = []
    for c in sorted(R21):
        labelled.append((2021, c, c, 1.0))
    for src_year, terc_path in [(2011, C.TERC_2011), (2026, C.TERC_2026)]:
        src = set(load_terc(terc_path)["code6"])
        unresolved = []
        for c in sorted(src):
            if c in R21:
                labelled.append((src_year, c, c, 1.0))
            elif c in special[src_year]:
                tot = sum(special[src_year][c].values())
                for dst, w in special[src_year][c].items():
                    if dst not in R21:
                        raise ValueError(f"{src_year} special target {dst} not in 2021 frame")
                    labelled.append((src_year, c, dst, w / tot))
            else:
                unresolved.append(c)
        if unresolved:
            raise ValueError(
                f"{src_year}: {len(unresolved)} gmina codes not in 2021 frame and not "
                f"in CROSSWALK_SPECIAL_{src_year}: {unresolved}")
        LOGGER.info("Crosswalk %d->2021: %d source gminas mapped (%d special)",
                    src_year, len(src), len(special[src_year]))
    labelled = pd.DataFrame(labelled, columns=["src_year", "src_code", "dst_code", "weight"])

    # universal map = identity(2021) + special_2011 + special_2026
    uni_rows = [(c, c, 1.0) for c in sorted(R21)]
    for sy in (2011, 2026):
        for c, tgt in special[sy].items():
            tot = sum(tgt.values())
            for dst, w in tgt.items():
                uni_rows.append((c, dst, w / tot))
    universal = pd.DataFrame(uni_rows, columns=["src_code", "dst_code", "weight"]).drop_duplicates()
    chk = universal.groupby("src_code")["weight"].sum()
    assert np.allclose(chk.values, 1.0), "universal crosswalk weights do not sum to 1"
    LOGGER.info("Crosswalk built: universal=%d rows, target frame=%d gminas (2021)",
                len(universal), len(R21))
    return universal, labelled, ref21


def harmonise(df, code_col, value_cols, crosswalk, how="sum", weight_col=None):
    """Aggregate a per-source-gmina table onto the 2021 frame with the universal
    crosswalk.

    how="sum"   : extensive quantities (employment, population, flows).
    how="wmean" : intensive quantities (wages); needs weight_col (e.g. employment).
                  For 1:1 rows this is the identity; folding only bites on the
                  handful of merged/dissolved source gminas.
    """
    if isinstance(value_cols, str):
        value_cols = [value_cols]
    cw = crosswalk[["src_code", "dst_code", "weight"]]
    d = df.copy()
    d["_src6"] = d[code_col].astype(str).map(code6)
    m = d.merge(cw, left_on="_src6", right_on="src_code", how="inner")

    out_frames = []
    if how == "sum":
        for v in value_cols:
            m[v + "_alloc"] = m[v].astype(float) * m["weight"]
        agg = m.groupby("dst_code")[[v + "_alloc" for v in value_cols]].sum(min_count=1)
        agg.columns = value_cols
        out_frames.append(agg)
    elif how == "wmean":
        if weight_col is None:
            raise ValueError("how='wmean' requires weight_col")
        m["_w"] = m[weight_col].astype(float) * m["weight"]
        for v in value_cols:
            ok = m[v].notna() & m["_w"].notna() & (m["_w"] > 0)
            mm = m[ok].copy()
            mm["_num"] = mm[v].astype(float) * mm["_w"]
            g = mm.groupby("dst_code").agg(_num=("_num", "sum"), _den=("_w", "sum"))
            out_frames.append((g["_num"] / g["_den"]).rename(v))
    else:
        raise ValueError(how)

    res = pd.concat(out_frames, axis=1).reset_index().rename(columns={"dst_code": "code6"})
    return res
