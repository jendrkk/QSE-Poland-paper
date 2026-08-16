#!/usr/bin/env python3
"""
estimate.py
===========

Statistical core of the labour-market builder.

Wages (powiat -> gmina disaggregation)
--------------------------------------
Workplace mean wages exist at gmina only from 2024 (P4609); for 2011 and 2021
only the powiat aggregate (P2497) is observed. We transfer the *within-powiat*
wage structure learned on the modern gmina cross-section back in time and pin
the level to the observed powiat total, mirroring the Fay-Herriot + exact
within-powiat benchmark used in ``scripts/floorspace``:

  1. Fit  log w_it = c_p(i) + x_it'β + e_it  on the recent gmina cross-section
     (c_p = powiat fixed effect), so β is the *within-powiat* gradient of log
     wage on covariates x (log workplace employment, gmina type).
  2. For a historical year, form within-powiat-centred covariates and predict a
     shrunk log deviation  d_i = λ · β'(x_i − x̄_p(i)).
  3. Rake exactly to the observed powiat wage:
         w_i = W_p · exp(d_i) / Σ_{j∈p} s_j exp(d_j),   s_j = emp_work_j / Σ emp_work
     so the employment-weighted gmina mean reproduces P2497 by construction and
     every gmina receives a value (no missingness on the 2021 frame).

Residence employment (2011/2021)
--------------------------------
The commuting matrices are off-diagonal only, so residence employment is
recovered from the accounting identity with register workplace employment:
     R_n = max(L_n − inflows_n, 0) + outflows_n,
then (if the census income-earner control P3357/P4488 is present) raked within
powiat to the census total, which corrects the register's undercount of
small-firm and farm employment. Working-age population is the fallback shape.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("labour.estimate")


# --------------------------------------------------------------------------- #
# Hierarchical imputation (powiat -> woj -> national median)
# --------------------------------------------------------------------------- #
def impute_hierarchical(df, value_col, hierarchy=("powiat", "woj", "national")):
    s = df[value_col].astype(float).copy()
    for level in hierarchy:
        if s.isna().sum() == 0:
            break
        if level == "national":
            s = s.fillna(s.median())
        else:
            key = df["code6"].str[:4] if level == "powiat" else df["code6"].str[:2]
            s = s.fillna(df.assign(_v=s).groupby(key)["_v"].transform("median"))
    return s


# --------------------------------------------------------------------------- #
# Recent trailing-window gmina wage
# --------------------------------------------------------------------------- #
def recent_window_wage(wage_gmina_long, concept, end, window_months):
    """Trailing-mean gmina wage over `window_months` ending at `end`=(year,month).

    Returns [code6, wage] (raw 2026-vintage codes; harmonise later).
    """
    ey, em = end
    idx = ey * 12 + (em - 1)
    lo = idx - (window_months - 1)
    w = wage_gmina_long[wage_gmina_long["concept"] == concept].copy()
    w["idx"] = w["year"] * 12 + (w["month"] - 1)
    w = w[(w["idx"] >= lo) & (w["idx"] <= idx)]
    out = (w.groupby("code6")["wage"].mean().rename("wage").reset_index())
    LOGGER.info("recent %s wage: %d months ending %d-%02d, %d gminas",
                concept, window_months, ey, em, len(out))
    return out


def recent_wage(wage_gmina_long, concept, end, window_months):
    """Recent gmina wage with graceful fallback + provenance.

    Coalesces, per gmina: trailing-window mean -> own full-history mean
    (recovers gminas whose recent months are confidentiality-suppressed, e.g.
    single-dominant-employer gminas) -> NaN (left for hierarchical imputation
    downstream). Returns [code6, wage, wsrc] with wsrc in
    {"window","own_history","missing"}.
    """
    win = recent_window_wage(wage_gmina_long, concept, end, window_months)
    hist = (wage_gmina_long[wage_gmina_long["concept"] == concept]
            .groupby("code6")["wage"].mean().rename("hist").reset_index())
    d = win.merge(hist, on="code6", how="outer")
    d["wsrc"] = np.where(d["wage"].notna(), "window",
                         np.where(d["hist"].notna(), "own_history", "missing"))
    d["wage"] = d["wage"].fillna(d["hist"])
    n_fb = int((d["wsrc"] == "own_history").sum())
    if n_fb:
        LOGGER.info("recent %s wage: %d gminas fell back to own history (recent suppressed)",
                    concept, n_fb)
    return d[["code6", "wage", "wsrc"]]


# --------------------------------------------------------------------------- #
# Within-powiat transfer model
# --------------------------------------------------------------------------- #
def fit_transfer_model(gmina_frame, covariates, ridge=1e-6):
    """OLS of log wage on covariates with powiat fixed effects (within transform).

    gmina_frame: [code6, wage, <covariates...>]  (recent cross-section)
    Returns beta (dict covariate->coef) and residual sd.
    """
    d = gmina_frame.dropna(subset=["wage"] + list(covariates)).copy()
    d = d[d["wage"] > 0]
    d["powiat"] = d["code6"].str[:4]
    y = np.log(d["wage"].to_numpy(float))
    X = d[list(covariates)].to_numpy(float)
    # within (powiat-demean) transform
    g = d["powiat"].to_numpy()
    ybar = pd.Series(y).groupby(g).transform("mean").to_numpy()
    Xdf = pd.DataFrame(X, columns=list(covariates))
    Xbar = Xdf.groupby(g).transform("mean").to_numpy()
    yt, Xt = y - ybar, X - Xbar
    XtX = Xt.T @ Xt + ridge * np.eye(Xt.shape[1])
    beta = np.linalg.solve(XtX, Xt.T @ yt)
    resid = yt - Xt @ beta
    dof = max(len(yt) - Xt.shape[1] - d["powiat"].nunique(), 1)
    sd = float(np.sqrt((resid @ resid) / dof))
    bd = dict(zip(covariates, beta))
    LOGGER.info("transfer model: n=%d, powiats=%d, beta=%s, resid_sd=%.4f",
                len(d), d["powiat"].nunique(), {k: round(v, 4) for k, v in bd.items()}, sd)
    return bd, sd


def disaggregate_wage(year, powiat_wage, gmina_cov, beta, covariates,
                      weight_col, shrinkage, hierarchy):
    """Predict + exactly-benchmark gmina wage for one historical year.

    powiat_wage : [powiat, wage]
    gmina_cov   : [code6, <covariates...>, weight_col]  (all 2021-frame gminas)
    Returns [code6, wage_hat, has_direct=False].
    """
    d = gmina_cov.copy()
    d["powiat"] = d["code6"].str[:4]
    for c in covariates:
        d[c] = impute_hierarchical(d, c, hierarchy)
    w = d[weight_col].astype(float)
    d["_w"] = impute_hierarchical(d.assign(**{weight_col: w}), weight_col, hierarchy).clip(lower=1.0)

    # within-powiat centred covariates -> shrunk log deviation
    dev = np.zeros(len(d))
    for c in covariates:
        xc = d[c].to_numpy(float)
        xbar = pd.Series(xc).groupby(d["powiat"].to_numpy()).transform("mean").to_numpy()
        dev += beta[c] * (xc - xbar)
    d["_dev"] = shrinkage * dev

    d = d.merge(powiat_wage.rename(columns={"wage": "_Wp"}), on="powiat", how="left")
    if d["_Wp"].isna().any():
        d["_Wp"] = impute_hierarchical(d, "_Wp", hierarchy)

    # exact within-powiat rake: employment-weighted mean of exp(dev) -> 1
    d["_e"] = np.exp(d["_dev"])
    num = d["_w"] * d["_e"]
    den = num.groupby(d["powiat"]).transform("sum")
    wbar = (d["_w"]).groupby(d["powiat"]).transform("sum")
    scale = wbar / den                     # so Σ s_j exp(dev_j) normalised
    d["wage_hat"] = d["_Wp"] * d["_e"] * scale
    d["has_direct"] = False
    LOGGER.info("wage disaggregation %d: %d gminas, median=%.0f zł",
                year, len(d), np.nanmedian(d["wage_hat"]))
    return d[["code6", "wage_hat", "has_direct"]]


# --------------------------------------------------------------------------- #
# Residence employment recovery
# --------------------------------------------------------------------------- #
def residence_employment(emp_work, flow_margins, workage, census_control,
                         hierarchy, method="flows_then_rake"):
    """Recover gmina residence employment on the 2021 frame.

    emp_work       : [code6, emp]           register workplace employment L_n
    flow_margins   : [code6, outflows, inflows]
    workage        : [code6, workage]       fallback shape
    census_control : [powiat, income_earners] or None (level rake)
    Returns [code6, emp_res, res_source]
    """
    d = emp_work.rename(columns={"emp": "L"}).merge(flow_margins, on="code6", how="outer")
    d = d.merge(workage, on="code6", how="outer")
    d["powiat"] = d["code6"].str[:4]
    d["L"] = impute_hierarchical(d, "L", hierarchy)
    for c in ("outflows", "inflows"):
        d[c] = d[c].fillna(0.0)

    if method == "workage_rake":
        d["R_raw"] = d["workage"].astype(float)
        src = "workage_shape"
    else:  # flows_then_rake
        Dn = np.maximum(d["L"] - d["inflows"], 0.0)
        d["R_raw"] = Dn + d["outflows"]
        # where the identity degenerates (R_raw==0 but people live there), fall back to workage
        deg = d["R_raw"] <= 0
        d.loc[deg, "R_raw"] = impute_hierarchical(d.assign(_x=np.where(deg, np.nan, d["R_raw"])),
                                                  "_x", hierarchy)[deg]
        src = "flows_register"

    if census_control is not None:
        d = d.merge(census_control, on="powiat", how="left")
        pow_raw = d.groupby("powiat")["R_raw"].transform("sum")
        rake = d["income_earners"] / pow_raw
        rake = rake.replace([np.inf, -np.inf], np.nan).fillna(1.0)
        d["emp_res"] = d["R_raw"] * rake
        d["res_source"] = src + "+census_rake"
    else:
        d["emp_res"] = d["R_raw"]
        d["res_source"] = src
        LOGGER.warning("no census income-earner control -> residence employment is "
                       "register/flows-recovered but NOT level-calibrated (undercounts "
                       "small-firm/farm employment)")

    LOGGER.info("residence employment: %d gminas, total=%.0f (source=%s)",
                len(d), d["emp_res"].sum(), d["res_source"].iloc[0])
    return d[["code6", "emp_res", "res_source"]]
