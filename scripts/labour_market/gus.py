#!/usr/bin/env python3
"""
gus.py
======

Parsers for the GUS BDL wide CSV exports used by the labour-market builder.
Every file has the same shape: two id columns ``Kod;Nazwa`` followed by value
columns whose *names* encode dimensions separated by ``;`` inside the quoted
field, e.g.::

    "styczeń;wg siedziby podmiotu;ogółem;2026;[zł]"   (P4609 wage)
    "ogółem;2011;[osoba]"                              (P2172 employment)
    "w wieku produkcyjnym;ogółem;2011;[osoba]"         (P3457 working-age pop)

Conventions (shared with scripts/floorspace/gus.py): semicolon-delimited,
UTF-8-BOM, decimal comma, a value of ``""`` or ``"0"`` denotes GUS suppression
and maps to NaN. ``Kod`` is a 7-digit TERYT code; the gmina universe keeps only
whole gminas (RODZ in {1,2,3}); powiat rows have code[4:] == "000".

All parsers return **long tidy** frames keyed on the raw 7-digit ``kod`` plus a
6-digit ``code6``; harmonisation onto the 2021 frame is done later via
``teryt.harmonise``.
"""

from __future__ import annotations

import csv
import logging

import numpy as np
import pandas as pd

from config import GUS_SUPPRESSED, MONTHS_PL, RODZ_WHOLE_GMINA
from teryt import code6, rodz_of

LOGGER = logging.getLogger("labour.gus")


# --------------------------------------------------------------------------- #
# Low-level wide reader
# --------------------------------------------------------------------------- #
def _read_wide(path):
    """Yield (kod, nazwa, [(dim_field, raw_value), ...]) per data row."""
    with open(path, encoding="utf-8-sig", newline="") as fh:
        reader = csv.reader(fh, delimiter=";")
        header = next(reader)
        value_cols = [(i, h) for i, h in enumerate(header)
                      if ";" in h and (h.endswith("[zł]") or h.endswith("[osoba]"))]
        for row in reader:
            if not row or not row[0].strip():
                continue
            kod = row[0].strip()
            nazwa = row[1].strip() if len(row) > 1 else ""
            cells = []
            for i, h in value_cols:
                if i < len(row):
                    cells.append((h, row[i].strip()))
            yield kod, nazwa, cells


def _num(raw):
    if raw in GUS_SUPPRESSED:
        return np.nan
    try:
        return float(raw.replace(",", "."))
    except ValueError:
        return np.nan


def _is_gmina(kod):
    return rodz_of(kod) in RODZ_WHOLE_GMINA


def _is_powiat(kod):
    s = "".join(ch for ch in str(kod) if ch.isdigit())
    return len(s) == 7 and s[4:] == "000" and s[2:4] != "00" and s[:2] != "00"


# --------------------------------------------------------------------------- #
# Wages
# --------------------------------------------------------------------------- #
def load_wage_powiat_yearly(path):
    """P2497: workplace mean wage, powiat, yearly. -> [powiat, year, wage]."""
    rows = []
    for kod, nazwa, cells in _read_wide(path):
        if not _is_powiat(kod):
            continue
        powiat = "".join(ch for ch in kod if ch.isdigit())[:4]
        for h, raw in cells:
            parts = h.split(";")            # "ogółem;<year>;[zł]"
            year = int(parts[-2])
            v = _num(raw)
            rows.append((powiat, year, np.nan if (v is not None and v <= 0) else v))
    df = pd.DataFrame(rows, columns=["powiat", "year", "wage"])
    LOGGER.info("P2497 powiat wage: %d powiats, years %d-%d",
                df["powiat"].nunique(), df["year"].min(), df["year"].max())
    return df


def load_wage_gmina_monthly(path):
    """P4609: gmina mean wage, monthly, both concepts.

    Returns long [code6, nazwa, concept, year, month, wage] for whole gminas,
    concept in {"workplace","residence"}.
    """
    concept_map = {"wg siedziby podmiotu": "workplace",
                   "wg miejsca zamieszkania": "residence"}
    rows = []
    for kod, nazwa, cells in _read_wide(path):
        if not _is_gmina(kod):
            continue
        c6 = code6(kod)
        for h, raw in cells:
            # "<month>;<concept>;ogółem;<year>;[zł]"
            parts = h.split(";")
            month = MONTHS_PL.get(parts[0])
            concept = concept_map.get(parts[1])
            if month is None or concept is None:
                continue
            year = int(parts[3])
            v = _num(raw)
            rows.append((c6, nazwa, concept, year, month, np.nan if (v is not None and v <= 0) else v))
    df = pd.DataFrame(rows, columns=["code6", "nazwa", "concept", "year", "month", "wage"])
    LOGGER.info("P4609 gmina wage: %d gminas, %d (concept,year,month) cells/gmina",
                df["code6"].nunique(), df.groupby("code6").size().median())
    return df


# --------------------------------------------------------------------------- #
# Employment
# --------------------------------------------------------------------------- #
def load_emp_work_yearly(path):
    """P2172 / P4508: workplace employment, gmina, yearly. -> [code6, year, emp]."""
    rows = []
    for kod, nazwa, cells in _read_wide(path):
        if not _is_gmina(kod):
            continue
        c6 = code6(kod)
        for h, raw in cells:
            parts = h.split(";")            # "ogółem;<year>;[osoba]"
            year = int(parts[-2])
            rows.append((c6, year, _num(raw)))
    df = pd.DataFrame(rows, columns=["code6", "year", "emp"])
    # collapse RODZ (whole-gmina rows already unique per code6, but be safe)
    df = df.groupby(["code6", "year"], as_index=False)["emp"].sum(min_count=1)
    LOGGER.info("employment(workplace) %s: %d gminas, years %d-%d",
                getattr(path, "name", path), df["code6"].nunique(),
                df["year"].min(), df["year"].max())
    return df


def load_emp_residence_monthly(path):
    """P4280: residence employment, gmina, monthly. -> [code6, year, month, emp]."""
    rows = []
    for kod, nazwa, cells in _read_wide(path):
        if not _is_gmina(kod):
            continue
        c6 = code6(kod)
        for h, raw in cells:
            parts = h.split(";")            # "<month>;pracujący;ogółem;<year>;[osoba]"
            month = MONTHS_PL.get(parts[0])
            if month is None:
                continue
            year = int(parts[-2])
            rows.append((c6, year, month, _num(raw)))
    df = pd.DataFrame(rows, columns=["code6", "year", "month", "emp"])
    LOGGER.info("P4280 residence employment: %d gminas, %d-%d",
                df["code6"].nunique(), df["year"].min(), df["year"].max())
    return df


# --------------------------------------------------------------------------- #
# Working-age population (residence)
# --------------------------------------------------------------------------- #
def load_working_age(path, year):
    """P3457 (2011) / P4362 (2021): working-age population by residence, gmina.

    Extracts the 'w wieku produkcyjnym' (working-age) column. -> [code6, workage].
    """
    rows = []
    for kod, nazwa, cells in _read_wide(path):
        if not _is_gmina(kod):
            continue
        c6 = code6(kod)
        val = np.nan
        for h, raw in cells:
            parts = h.split(";")
            label = parts[0]
            if label == "w wieku produkcyjnym":
                val = _num(raw)
                break
        rows.append((c6, val))
    df = pd.DataFrame(rows, columns=["code6", "workage"]).drop_duplicates("code6")
    LOGGER.info("working-age pop %d: %d gminas (%d non-missing)",
                year, len(df), int(df["workage"].notna().sum()))
    return df


# --------------------------------------------------------------------------- #
# Census income-earner control totals (powiat), auto-detected
# --------------------------------------------------------------------------- #
def load_census_income_powiat(path):
    """P3357 (2011) / P4488 (2021): population by main source of income, powiat.

    Sums every 'praca' (own-work) category into a residence-employment control
    total per powiat. Robust to the exact column layout: any value column whose
    first dimension label contains 'praca' is treated as own-work income.
    Returns [powiat, income_earners].
    """
    acc = {}
    for kod, nazwa, cells in _read_wide(path):
        if not _is_powiat(kod):
            continue
        powiat = "".join(ch for ch in kod if ch.isdigit())[:4]
        tot = np.nan
        for h, raw in cells:
            label = h.split(";")[0].lower()
            if "praca" in label or "pracy" in label:
                v = _num(raw)
                if not np.isnan(v):
                    tot = (0.0 if np.isnan(tot) else tot) + v
        acc[powiat] = tot
    df = pd.DataFrame({"powiat": list(acc), "income_earners": list(acc.values())})
    LOGGER.info("census income-earners %s: %d powiats (%d non-missing)",
                getattr(path, "name", path), len(df), int(df["income_earners"].notna().sum()))
    return df
