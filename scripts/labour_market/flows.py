#!/usr/bin/env python3
"""
flows.py
========

Parse the bilateral commuting matrices (NSP 2011 tax-register, NSP 2021 census),
harmonise their residence and workplace codes onto the 2021 gmina frame with the
shared crosswalk, and return the per-gmina commuting margins used to recover the
census within-gmina diagonal.

Both source matrices are **off-diagonal only** (no within-gmina flows; the 2021
census additionally censors flows < 3 persons). After folding merged source
gminas (e.g. Zielona Góra rural into the city) a small amount of mass becomes
genuinely intra-gmina and is correctly assigned to the diagonal.

For each 2021 gmina n we return:
    outflows[n]  = sum_j!=n  flow(res=n, work=j)      (row sum, off-diagonal)
    inflows[n]   = sum_i!=n  flow(res=i, work=n)      (col sum, off-diagonal)

These feed the identity  R_n = max(L_n - inflows_n, 0) + outflows_n  in
``estimate.residence_employment``.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from teryt import code6, harmonise

LOGGER = logging.getLogger("labour.flows")


def _parse_2011(path):
    df = pd.read_excel(path, engine="xlrd", header=0)
    df = df.iloc[:, :3]
    df.columns = ["res", "work", "count"]
    return df


def _parse_2021(path):
    m = pd.read_excel(path, engine="openpyxl", sheet_name="Macierz przepływów", header=None)
    # row 0 is the multi-line header; cols: res_code, res_name, work_code, work_name, commuters
    m = m.iloc[1:, [0, 2, 4]].copy()
    m.columns = ["res", "work", "count"]
    return m


def load_flow_margins(path, src_year, crosswalk):
    """Return DataFrame[code6, outflows, inflows] on the 2021 frame."""
    raw = _parse_2011(path) if src_year == 2011 else _parse_2021(path)
    raw = raw.dropna(subset=["res", "work"])
    raw["res6"] = raw["res"].astype(str).map(code6)
    raw["work6"] = raw["work"].astype(str).map(code6)
    raw["count"] = pd.to_numeric(raw["count"], errors="coerce").fillna(0.0)

    cw = crosswalk[["src_code", "dst_code", "weight"]]
    cwr = cw.rename(columns={"src_code": "res6", "dst_code": "res21", "weight": "wr"})
    cww = cw.rename(columns={"src_code": "work6", "dst_code": "work21", "weight": "ww"})
    m = raw.merge(cwr, on="res6", how="inner").merge(cww, on="work6", how="inner")
    m["flow"] = m["count"] * m["wr"] * m["ww"]

    agg = m.groupby(["res21", "work21"], as_index=False)["flow"].sum()
    offdiag = agg[agg["res21"] != agg["work21"]]
    outflows = offdiag.groupby("res21")["flow"].sum().rename("outflows")
    inflows = offdiag.groupby("work21")["flow"].sum().rename("inflows")
    diag = agg[agg["res21"] == agg["work21"]]["flow"].sum()

    out = pd.concat([outflows, inflows], axis=1).reset_index().rename(columns={"index": "code6"})
    out = out.rename(columns={out.columns[0]: "code6"})
    out[["outflows", "inflows"]] = out[["outflows", "inflows"]].fillna(0.0)
    LOGGER.info("flows %d harmonised: %s OD pairs, total commuters=%.0f, folded-diagonal=%.0f",
                src_year, f"{len(agg):,}", agg['flow'].sum(), diag)
    return out
