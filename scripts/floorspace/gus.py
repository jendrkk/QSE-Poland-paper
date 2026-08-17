#!/usr/bin/env python3
"""
gus.py
======

Parse the two GUS BDL exports of transaction prices per 1 m2 of residential
premises (``lokale mieszkalne``) and reshape them into tidy long tables.

Input files (semicolon-delimited, UTF-8, quoted header)::

    GUS_P3787_median_price_residential_floorspace_1m2.csv   (median)
    GUS_P3788_mean_price_residential_floorspace_1m2.csv     (mean)

Header layout: two id columns ``Kod;Nazwa`` followed by value columns whose
names encode three dimensions separated by ';' inside the quoted field::

    "<market>;<size_class>;<year>;[zł]"

    market      in {ogółem, rynek pierwotny, rynek wtórny}
    size_class  in {ogółem, do 40 m2, od 40,1 do 60 m2, od 60,1 do 80 m2, od 80,1 m2}
    year        in {2010 .. 2024}

``Kod`` is ``WWPPGGG``; powiat aggregate rows have ``GGG == 000`` and
``PP != 00``. A value of 0 denotes GUS suppression and is mapped to NaN.

Because GUS is built from the same RCN registry (restricted to flats), these
figures serve as the powiat-level prior mean / auxiliary covariate and as a
validation benchmark for the cleaned RCN micro data.

Public API
----------
load_gus_long(median_csv, mean_csv) -> pandas.DataFrame
    Tidy long table: powiat, market, size_class, year, median, mean.
powiat_anchor(long, year, market="ogółem", size="ogółem") -> DataFrame
    One row per powiat with the median/mean anchor for the given cell.
national_series(long, market, size) -> DataFrame
    POLSKA yearly series (for validating the hedonic year FE path).
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("floorspace.gus")

_MARKET_CANON = {
    "ogółem": "total",
    "rynek pierwotny": "primary",
    "rynek wtórny": "secondary",
}


def _parse_one(path: Path, value_name: str) -> pd.DataFrame:
    """Read one wide BDL file into long form with a single value column."""
    with open(path, encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh, delimiter=";")
        header = next(reader)
        # header has a trailing empty field from the terminating ';'
        value_cols = [(i, h) for i, h in enumerate(header) if ";" in h and h.endswith("[zł]")]
        records = []
        for row in reader:
            if not row or not row[0]:
                continue
            kod = row[0].strip()
            nazwa = row[1].strip() if len(row) > 1 else ""
            for i, h in value_cols:
                if i >= len(row):
                    continue
                raw = row[i].strip()
                if raw == "" or raw == "0":
                    val = np.nan
                else:
                    try:
                        val = float(raw.replace(",", "."))
                    except ValueError:
                        val = np.nan
                market, size_class, year, _unit = h.split(";")
                records.append((kod, nazwa, market, size_class, int(year), val))
    df = pd.DataFrame.from_records(
        records,
        columns=["kod", "nazwa", "market_raw", "size_class", "year", value_name],
    )
    df["market"] = df["market_raw"].map(_MARKET_CANON).fillna(df["market_raw"])
    df.drop(columns=["market_raw"], inplace=True)
    return df


def _is_powiat_code(kod: str) -> bool:
    """WWPPGGG powiat-aggregate: 7 digits, GGG == 000, PP != 00."""
    return len(kod) == 7 and kod[4:] == "000" and kod[2:4] != "00" and kod[:2] != "00"


def load_gus_long(median_csv: Path, mean_csv: Path) -> pd.DataFrame:
    """Merge median and mean files into one tidy long table (powiat rows only).

    Returns columns: powiat (4-digit), nazwa, market, size_class, year,
    median, mean.
    """
    LOGGER.info("Parsing GUS median: %s", median_csv)
    med = _parse_one(median_csv, "median")
    LOGGER.info("Parsing GUS mean:   %s", mean_csv)
    mea = _parse_one(mean_csv, "mean")

    key = ["kod", "nazwa", "market", "size_class", "year"]
    df = med.merge(mea, on=key, how="outer")

    df = df[df["kod"].map(_is_powiat_code)].copy()
    df["powiat"] = df["kod"].str[:4]
    df.drop(columns=["kod"], inplace=True)

    n_pow = df["powiat"].nunique()
    LOGGER.info(
        "GUS long table: %s rows, %d powiats, years %d-%d",
        f"{len(df):,}",
        n_pow,
        int(df["year"].min()),
        int(df["year"].max()),
    )
    return df


def powiat_anchor(
    long: pd.DataFrame,
    year: int,
    market: str = "total",
    size: str = "ogółem",
) -> pd.DataFrame:
    """One row per powiat with the median/mean anchor for the requested cell."""
    cell = long[
        (long["year"] == year)
        & (long["market"] == market)
        & (long["size_class"] == size)
    ][["powiat", "median", "mean"]].copy()
    cell = cell.rename(columns={"median": "gus_median", "mean": "gus_mean"})
    n_ok = cell["gus_median"].notna().sum()
    LOGGER.info(
        "GUS anchor %d/%s/%s: %d powiats, %d with a median value",
        year, market, size, len(cell), int(n_ok),
    )
    return cell.reset_index(drop=True)


def anchor_for_year(
    long: pd.DataFrame,
    year: int,
    market: str = "total",
    size: str = "ogółem",
    last_year: int = 2024,
    growth: float | dict | None = None,
) -> pd.DataFrame:
    """Powiat price anchor for any MODEL year, extrapolating past the GUS horizon.

    For ``year <= last_year`` this is exactly :func:`powiat_anchor` for that year.
    For ``year > last_year`` (GUS stops at 2024) the ``last_year`` powiat values
    are scaled by ``growth`` -- either a scalar national factor or a dict keyed
    by 2-digit voivodeship code -- so the 2026 vector inherits the official 2024
    cross-section grown by the recent RCN trend. ``growth`` compounding is the
    caller's responsibility (pass the full last_year -> year factor).
    """
    if year <= last_year:
        return powiat_anchor(long, year, market=market, size=size)
    base = powiat_anchor(long, last_year, market=market, size=size)
    if growth is None:
        LOGGER.warning("anchor_for_year %d > last_year %d but no growth given; "
                       "returning %d level unscaled", year, last_year, last_year)
        return base
    if isinstance(growth, dict):
        woj = base["powiat"].str[:2]
        g = woj.map(growth).fillna(np.nanmedian(list(growth.values()))).to_numpy(float)
    else:
        g = float(growth)
    out = base.copy()
    out["gus_median"] = out["gus_median"].to_numpy(float) * g
    out["gus_mean"] = out["gus_mean"].to_numpy(float) * g
    LOGGER.info("GUS anchor extrapolated %d->%d (growth=%s)",
                last_year, year, growth if np.isscalar(growth) else "per-woj")
    return out


# whole-gmina TERYT type digits: 1 urban, 2 rural, 3 urban-rural (aggregate).
# 4/5 are the miasto/obszar-wiejski split of a type-3 gmina and 8 are big-city
# districts -- both are SUB-parts and must be excluded to avoid double counting
# (types {1,2,3} reconstruct the national dwelling total exactly, and match the
# 7-digit coding of communes_2021).
_WHOLE_GMINA_TYPES = {"1", "2", "3"}


def load_housing_stock(path: Path) -> pd.DataFrame:
    """Gmina-level total dwelling stock (P2166) as a tidy long table.

    The BDL export codes ``WWPPGGR`` (7 digits incl. the gmina TYPE digit R).
    We keep only whole-gmina rows (type 1/2/3) so the 7-digit code joins directly
    to ``communes_2021`` and the counts never double-count the 4/5 split parts of
    an urban-rural gmina.

    Returns columns: ``gmina_teryt`` (7-digit), ``year``, ``dwellings``. Only the
    ``mieszkania`` (dwellings) attribute is kept; zeros/blanks -> dropped.
    """
    with open(path, encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh, delimiter=";")
        header = next(reader)
        value_cols = [(i, h) for i, h in enumerate(header)
                      if ";" in h and "mieszkania" in h]
        records = []
        for row in reader:
            if not row or not row[0]:
                continue
            kod = row[0].strip()
            # keep whole gmina rows only: 7 digits, GGG != 000, PP != 00, type 1/2/3
            if len(kod) != 7 or kod[4:] == "000" or kod[2:4] == "00":
                continue
            if kod[-1] not in _WHOLE_GMINA_TYPES:
                continue
            for i, h in value_cols:
                if i >= len(row):
                    continue
                raw = row[i].strip()
                if raw in ("", "0"):
                    continue
                try:
                    val = float(raw.replace(",", "."))
                except ValueError:
                    continue
                # header: "ogółem;mieszkania;YEAR;[-]"
                _tot, _attr, year, _unit = h.split(";")
                records.append((kod, int(year), val))
    df = pd.DataFrame.from_records(records, columns=["gmina_teryt", "year", "dwellings"])
    df = df.groupby(["gmina_teryt", "year"], as_index=False)["dwellings"].sum()
    LOGGER.info("GUS housing stock: %s gmina x year rows, %d gminas, years %d-%d",
                f"{len(df):,}", df["gmina_teryt"].nunique(),
                int(df["year"].min()), int(df["year"].max()))
    return df


def housing_stock_weights(stock_long: pd.DataFrame, year: int) -> dict:
    """{gmina_teryt (7-digit) -> dwelling count} for the given year (nearest
    earlier year if the exact year is absent, e.g. model year 2026 -> 2025)."""
    yrs = sorted(stock_long["year"].unique())
    use = year if year in yrs else max([y for y in yrs if y <= year] or [min(yrs)])
    if use != year:
        LOGGER.info("housing-stock weights: year %d absent, using %d", year, use)
    sub = stock_long[stock_long["year"] == use]
    return dict(zip(sub["gmina_teryt"], sub["dwellings"].astype(float)))


def national_series(
    long: pd.DataFrame, market: str = "total", size: str = "ogółem"
) -> pd.DataFrame:
    """POLSKA-level yearly series is not in the powiat-filtered table; recompute
    the transaction-agnostic national path as the cross-powiat median of the
    powiat medians (a robust proxy used only for diagnostics)."""
    sub = long[(long["market"] == market) & (long["size_class"] == size)]
    out = (
        sub.groupby("year")["median"]
        .median()
        .reset_index()
        .rename(columns={"median": "gus_powiat_median_of_medians"})
    )
    return out
