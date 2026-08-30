"""
partitions.py — historical-partition analysis and counterfactuals.

Poland's gminas carry a `partition` label (P=Prussian, R=Russian, A=Austrian)
recording which 1795–1918 partition each commune would fall in under today's
borders. The label lives as a column in
``data/processed/shapefiles/communes_2021_partitions.gpkg`` (same geometry and
2477-gmina frame as ``communes_2021.gpkg``, plus the extra column).

This module needs only the *attribute table*, so it reads the GeoPackage with
stdlib ``sqlite3`` (a .gpkg IS a SQLite DB) — no geopandas dependency, so it
runs anywhere the model runs.

Three experiments, all on a solved baseline RunResult:

  gaps     descriptive/reduced-form: weighted regression of a recovered
           object (log A_n, log b_n, log CMA, log real_v) on partition dummies
           (+ optional controls: voivodeship FE, log population, log travel time
           to Warsaw). Reports the partition means and the pairwise gaps, i.e.
           the Poland analogue of the East–West gap in Topic 11.

  border   border-imposition counterfactual (Berlin-Wall / Redding–Sturm style):
           raise the commuting cost (and/or trade cost) on every gmina-pair that
           crosses a partition seam by a factor, hold fundamentals fixed, re-solve
           with exact-hat algebra, and read off the welfare cost of "re-drawing"
           the partition border today. Run per year to see whether road expansion
           has made the country more or less reliant on cross-seam integration.

  removal  gap-removal counterfactual (Topic 11 task (c)): equalise the partition
           MEAN of a fundamental (productivity and/or amenity) across P/R/A and
           re-solve, isolating the GE effect of the systematic partition gap.

All counterfactuals reuse ``counterfac.counter_facts`` verbatim, so they inherit
the same solver and the identity test.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np

from . import counterfac as cf
from .frame import code7

PARTITIONS = ("P", "R", "A")   # Prussian, Russian, Austrian
_PART_NAME = {"P": "Prussian", "R": "Russian", "A": "Austrian"}
_TERYT_KEYS = ("JPT_KOD_JE", "teryt7", "TERYT", "gmina_teryt", "code7", "JPT_KJ_I_1")


# --------------------------------------------------------------------------- #
# Loading the partition label (sqlite; no geopandas)
# --------------------------------------------------------------------------- #
def _feature_table(con) -> str:
    rows = con.execute(
        "SELECT table_name FROM gpkg_contents WHERE data_type='features'").fetchall()
    if not rows:
        raise ValueError("no feature table in gpkg_contents")
    return rows[0][0]


def load_partition(gpkg_path, codes, *, part_col="partition") -> np.ndarray:
    """Return the partition label for every code in `codes` (frame order).

    Reads only the attribute table via sqlite3. Raises if any frame code is
    missing a label (the delivered partition gpkg covers all 2477 gminas 1:1).
    """
    con = sqlite3.connect(str(gpkg_path))
    try:
        tbl = _feature_table(con)
        cols = [r[1] for r in con.execute(f'PRAGMA table_info("{tbl}")')]
        if part_col not in cols:
            raise ValueError(f"`{part_col}` not in {tbl}; columns={cols}")
        key = next((k for k in _TERYT_KEYS if k in cols), None)
        if key is None:
            raise ValueError(f"no TERYT key in {tbl}; columns={cols}")
        d = {code7(k): v for k, v in
             con.execute(f'SELECT "{key}", "{part_col}" FROM "{tbl}"')}
    finally:
        con.close()
    out = np.array([d.get(code7(c), None) for c in codes], dtype=object)
    miss = int(sum(v is None for v in out))
    if miss:
        raise ValueError(f"{miss} frame gminas have no partition label")
    return out


def load_column(gpkg_path, codes, col, *, dtype=float, fill=np.nan) -> np.ndarray:
    """Generic: pull one attribute column onto the frame order (sqlite only)."""
    con = sqlite3.connect(str(gpkg_path))
    try:
        tbl = _feature_table(con)
        cols = [r[1] for r in con.execute(f'PRAGMA table_info("{tbl}")')]
        if col not in cols:
            raise ValueError(f"`{col}` not in {tbl}")
        key = next((k for k in _TERYT_KEYS if k in cols), None)
        d = {code7(k): v for k, v in
             con.execute(f'SELECT "{key}", "{col}" FROM "{tbl}"')}
    finally:
        con.close()
    out = np.array([d.get(code7(c), np.nan) for c in codes], dtype=float)
    if fill is not None:
        out = np.where(np.isnan(out), fill, out)
    return out.astype(dtype)


# --------------------------------------------------------------------------- #
# Weights and controls
# --------------------------------------------------------------------------- #
def _weights(run, how="R_n") -> np.ndarray:
    if how in (None, "none"):
        return np.ones(run.frame["N"])
    if how == "R_n":
        return np.asarray(run.inputs["R_n"], float)
    if how == "L_n":
        return np.asarray(run.inputs["L_n"], float)
    raise ValueError(f"unknown weight `{how}`")


def warsaw_traveltime(run) -> np.ndarray:
    """Min travel time (minutes) from each gmina to a Warsaw workplace.

    Warsaw = TERYT voivodeship 14, powiat 65 (m.st. Warszawa, code 1465xxx).
    Uses the run's own tau matrix [residence, workplace]."""
    codes = np.asarray(run.codes)
    tau = np.asarray(run.inputs["tau"], float)
    war = np.array([str(c).zfill(7).startswith("1465") for c in codes])
    if not war.any():
        return np.full(len(codes), np.nan)
    return tau[:, war].min(axis=1)


def build_controls(run, names, gpkg_path=None) -> tuple[np.ndarray, list[str]]:
    """Assemble a design block of controls. Supported names:
        woj            voivodeship fixed effects (dummies, first dropped)
        logpop         log population (from partition gpkg `pop` column)
        log_tt_warsaw  log(min travel time to Warsaw + 1)
    Returns (X_controls [N,k], column_labels)."""
    N = run.frame["N"]
    cols, labels = [], []
    codes = np.asarray(run.codes)
    for nm in names or []:
        if nm == "woj":
            woj = np.array([str(c).zfill(7)[:2] for c in codes])
            for w in sorted(set(woj))[1:]:               # drop first as base
                cols.append((woj == w).astype(float)); labels.append(f"woj_{w}")
        elif nm == "logpop":
            if gpkg_path is None:
                raise ValueError("logpop needs --gpkg")
            pop = load_column(gpkg_path, codes, "pop")
            cols.append(np.log(np.where(pop > 0, pop, np.nan))); labels.append("logpop")
        elif nm == "log_tt_warsaw":
            tt = warsaw_traveltime(run)
            cols.append(np.log(tt + 1.0)); labels.append("log_tt_warsaw")
        else:
            raise ValueError(f"unknown control `{nm}`")
    if not cols:
        return np.empty((N, 0)), []
    return np.column_stack(cols), labels


# --------------------------------------------------------------------------- #
# (1) Descriptive / reduced-form partition gaps
# --------------------------------------------------------------------------- #
_OBJECTS = {
    "A_n": ("calibrated", "log A_n (productivity)"),
    "b_n": ("calibrated", "log b_n (amenity)"),
    "CMA": ("calibrated", "log CMA (market access)"),
    "real_v": ("calibrated", "log v_n/CPI (real income)"),
}


def _wls(y, X, w):
    """Weighted least squares; returns (beta, robust HC0 se)."""
    W = np.sqrt(w)
    Xw, yw = X * W[:, None], y * W
    beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    resid = y - X @ beta
    XtWX_inv = np.linalg.pinv(Xw.T @ Xw)
    meat = (X * (w * resid**2)[:, None]).T @ X
    cov = XtWX_inv @ meat @ XtWX_inv
    return beta, np.sqrt(np.clip(np.diag(cov), 0, None))


def partition_gaps(run, part, *, objects=None, weight="R_n",
                   controls=None, gpkg_path=None) -> dict:
    """Weighted regression of each recovered log-object on partition dummies
    (+ optional controls). Base category = Russian ('R', the largest / Warsaw
    partition). Returns per-object means, dummy coefficients (gaps vs base) and SEs."""
    objects = objects or list(_OBJECTS)
    w = _weights(run, weight)
    Xc, clabels = build_controls(run, controls or [], gpkg_path)
    base = "R"
    others = [p for p in PARTITIONS if p != base]
    D = np.column_stack([(part == p).astype(float) for p in others])
    out = {"weight": weight, "base": base, "controls": clabels, "objects": {}}
    for key in objects:
        sec, lab = _OBJECTS[key]
        y = np.log(np.asarray(getattr(run, sec)[key], float))
        ok = np.isfinite(y) & np.all(np.isfinite(Xc), axis=1) if Xc.size else np.isfinite(y)
        X = np.column_stack([np.ones(ok.sum()), D[ok]] +
                            ([Xc[ok]] if Xc.size else []))
        beta, se = _wls(y[ok], X, w[ok])
        means = {p: float(np.average(y[part == p], weights=w[part == p]))
                 for p in PARTITIONS}
        gaps = {others[i]: (float(beta[1 + i]), float(se[1 + i])) for i in range(len(others))}
        out["objects"][key] = dict(label=lab, means=means, gaps_vs_base=gaps,
                                   n=int(ok.sum()))
    return out


# --------------------------------------------------------------------------- #
# (2) Border-imposition counterfactual
# --------------------------------------------------------------------------- #
def border_matrix(part) -> np.ndarray:
    """B[n,i] = 1 iff residence n and workplace/destination i are in different
    partitions (a partition-seam crossing)."""
    p = np.asarray(part, dtype=object)
    return (p[:, None] != p[None, :]).astype(float)


def impose_border(run, part, *, commute_cost=1.5, trade_cost=1.0,
                  channels="commute", **cf_kw) -> dict:
    """Raise cross-seam commuting cost by `commute_cost` and/or trade cost by
    `trade_cost` (multiplicative on the crossing pairs), fundamentals fixed.
    channels: 'commute' | 'trade' | 'both'. Returns the counter_facts dict plus
    the aggregate welfare change (percent)."""
    n = run.frame["N"]
    B = border_matrix(part)
    kap = np.where(B > 0, commute_cost, 1.0) if channels in ("commute", "both") \
        else np.ones((n, n))
    dch = np.where(B > 0, trade_cost, 1.0) if channels in ("trade", "both") \
        else np.ones((n, n))
    f64 = lambda a: np.asarray(a, float)
    res = cf.counter_facts(
        aChange=np.ones(n), bChange=np.ones((n, n)), kapChange=kap, dChange=dch,
        wObs=f64(run.calibrated["w_n"]), vObs=f64(run.calibrated["v_n"]),
        lamObs=f64(run.inputs["uncondCom"]), lObs=f64(run.inputs["L_n"]),
        rObs=f64(run.inputs["R_n"]), piObs=f64(run.calibrated["tradesh"]),
        alp=run.params["alpha"], epsi=run.params["epsi"], delta=run.params["delta"],
        sigg=run.params["sigg"], nu=run.params["nu"], **cf_kw)
    res["welfare_pct"] = (res["welf"] - 1) * 100
    res["channels"] = channels
    res["commute_cost"] = commute_cost
    res["trade_cost"] = trade_cost
    return res


# --------------------------------------------------------------------------- #
# (3) Gap-removal counterfactual
# --------------------------------------------------------------------------- #
def remove_gap(run, part, *, target="A_n", weight="R_n", **cf_kw) -> dict:
    """Equalise the partition MEAN (pop-weighted geometric mean) of `target`
    across P/R/A by rescaling each gmina's fundamental, holding within-partition
    variation fixed; re-solve. target: 'A_n' (productivity, aChange) or
    'b_n' (amenity, bChange as a row-broadcast N×N hat)."""
    n = run.frame["N"]
    w = _weights(run, weight)
    if target == "A_n":
        y = np.log(np.asarray(run.calibrated["A_n"], float))
        overall = np.average(y, weights=w)
        pm = {p: np.average(y[part == p], weights=w[part == p]) for p in PARTITIONS}
        aChange = np.array([np.exp(overall - pm[p]) for p in part])
        bChange = np.ones((n, n))
    elif target == "b_n":
        y = np.log(np.asarray(run.calibrated["b_n"], float))
        overall = np.average(y, weights=w)
        pm = {p: np.average(y[part == p], weights=w[part == p]) for p in PARTITIONS}
        row = np.array([np.exp(overall - pm[p]) for p in part])
        aChange = np.ones(n)
        bChange = np.tile(row[:, None], (1, n))   # residence-amenity hat, by row
    else:
        raise ValueError("target must be 'A_n' or 'b_n'")
    f64 = lambda a: np.asarray(a, float)
    res = cf.counter_facts(
        aChange=aChange, bChange=bChange, kapChange=np.ones((n, n)),
        dChange=np.ones((n, n)),
        wObs=f64(run.calibrated["w_n"]), vObs=f64(run.calibrated["v_n"]),
        lamObs=f64(run.inputs["uncondCom"]), lObs=f64(run.inputs["L_n"]),
        rObs=f64(run.inputs["R_n"]), piObs=f64(run.calibrated["tradesh"]),
        alp=run.params["alpha"], epsi=run.params["epsi"], delta=run.params["delta"],
        sigg=run.params["sigg"], nu=run.params["nu"], **cf_kw)
    res["welfare_pct"] = (res["welf"] - 1) * 100
    res["target"] = target
    res["partition_means"] = {p: float(v) for p, v in pm.items()}
    return res
