#!/usr/bin/env python3
"""
estimate.py
===========

Econometric core of the floorspace-index pipeline.

Stage 3 -- national hedonic (``fit_hedonic``)
    Pooled OLS of log(price/m2) on centred structural controls, plus full-set
    gmina fixed effects and year (or voivodeship x year) fixed effects with the
    2021 reference dropped. Controls are centred at their sample means, so each
    gmina fixed effect equals the expected log(price/m2) of a mean-characteristics
    dwelling in that gmina *in 2021* -- exactly the "direct" small-area estimate
    theta_hat_g. Heteroskedasticity-robust (HC1) sandwich variances give the
    sampling variance D_g of each gmina effect. Solved as a sparse normal-
    equations system (no external FE library required).

Stage 4 -- Fay-Herriot small-area model (``fay_herriot``)
    Area-level model  theta_hat_g = x_g'beta + v_g + e_g,  e_g ~ N(0, D_g),
    v_g ~ N(0, A), with x_g including log(GUS powiat 2021) as the dominant
    predictor (the hierarchical prior mean), gmina type, log land price,
    log population. REML/moment estimation of (beta, A); empirical-Bayes
    posterior means shrink data-rich gminas toward their direct estimate and
    thin / zero-data gminas toward the GUS-anchored prediction. Prasad-Rao MSE.

Everything is numpy + scipy.sparse; both are standard on the HPC image.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import scipy.sparse as sp

LOGGER = logging.getLogger("floorspace.estimate")


# --------------------------------------------------------------------------- #
# Stage 3: hedonic
# --------------------------------------------------------------------------- #
def _build_controls(micro: pd.DataFrame):
    """Return centred control matrix Z (dense) and its column names.

    Missing rooms/floor/plot are mean-imputed with an accompanying missingness
    indicator so their rows stay in the regression.
    """
    n = len(micro)
    area = micro["area"].to_numpy(float)
    log_area = np.log(area)
    log_area2 = log_area ** 2
    house = (micro["property_type"].to_numpy() == "house").astype(float)
    primary = (micro["market"].to_numpy() == "primary").astype(float)

    rooms = micro["rooms"].to_numpy(float)
    has_rooms = np.isfinite(rooms).astype(float)
    rooms = np.where(np.isfinite(rooms), rooms, np.nanmedian(rooms[np.isfinite(rooms)]) if np.isfinite(rooms).any() else 0.0)

    floor = micro["floor"].to_numpy(float)
    has_floor = np.isfinite(floor).astype(float)
    floor = np.where(np.isfinite(floor), floor, 0.0)

    plot = micro["plot_area"].to_numpy(float)
    log_plot = np.where((house == 1) & np.isfinite(plot) & (plot > 0), np.log(plot), 0.0)

    cols = {
        "log_area": log_area,
        "log_area2": log_area2,
        "house": house,
        "primary": primary,
        "rooms": rooms,
        "has_rooms": has_rooms,
        "floor": floor,
        "has_floor": has_floor,
        "log_plot_house": log_plot,
    }
    names = list(cols)
    Z = np.column_stack([cols[k] for k in names]).astype(float)
    # centre continuous / all controls so gmina FE = prediction at the mean dwelling
    Zc = Z - Z.mean(axis=0, keepdims=True)
    return Zc, names, Z.mean(axis=0)


def fit_hedonic(micro: pd.DataFrame, time_fe: str = "year", ridge: float = 1e-6,
                ref_year: int = 2021):
    """Estimate the national hedonic and return per-gmina direct estimates.

    Returns
    -------
    direct : DataFrame [gmina_teryt, theta_hat, D_g, n_obs]
    info   : dict with control coefficients, year-FE path, R2, sigma2.
    """
    micro = micro[micro["gmina_teryt"].notna()].copy()
    y = np.log(micro["ppm2"].to_numpy(float))

    # gmina design (full one-hot, absorbs the intercept)
    gcat = pd.Categorical(micro["gmina_teryt"].astype(str))
    G = sp.csr_matrix(
        (np.ones(len(micro)), (np.arange(len(micro)), gcat.codes)),
        shape=(len(micro), len(gcat.categories)),
    )

    # time design (reference 2021 dropped)
    if time_fe == "woj_year":
        tkey = micro["powiat_teryt"].astype(str).str[:2] + "_" + micro["year"].astype(int).astype(str)
    else:
        tkey = micro["year"].astype(int).astype(str)
    tcat = pd.Categorical(tkey)
    # Reference = the 2021 cell(s). For woj_year there is one 2021 cell per
    # voivodeship; ALL of them are dropped so each gmina effect is normalised to
    # its own 2021 level (the woj-2021 baseline is absorbed into the gmina FE).
    ref_set = {lab for lab in tcat.categories if lab.endswith(str(ref_year))}
    if not ref_set:
        ref_set = {tcat.categories[-1]}
        LOGGER.warning("Reference year %d absent from data; using %s as time baseline",
                       ref_year, sorted(ref_set))
    keep_t = [i for i, lab in enumerate(tcat.categories) if lab not in ref_set]
    ref = "|".join(sorted(ref_set))
    codes = tcat.codes
    remap = {c: j for j, c in enumerate(keep_t)}
    rows_t = np.where(np.isin(codes, keep_t))[0]
    cols_t = np.array([remap[codes[i]] for i in rows_t])
    T = sp.csr_matrix(
        (np.ones(len(rows_t)), (rows_t, cols_t)),
        shape=(len(micro), len(keep_t)),
    )

    Zc, znames, zmean = _build_controls(micro)
    Zs = sp.csr_matrix(Zc)

    X = sp.hstack([G, T, Zs], format="csr")
    nG, nT, nZ = G.shape[1], T.shape[1], Zs.shape[1]
    p = X.shape[1]
    LOGGER.info("Hedonic design: n=%s, params=%d (gmina=%d, time=%d, controls=%d)",
                f"{len(micro):,}", p, nG, nT, nZ)

    XtX = (X.T @ X).toarray()
    XtX[np.diag_indices_from(XtX)] += ridge
    Xty = X.T @ y
    XtX_inv = np.linalg.inv(XtX)
    beta = XtX_inv @ Xty

    resid = y - X @ beta
    dof = len(y) - p
    sigma2 = float(resid @ resid / dof)
    tss = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float(resid @ resid) / tss

    # HC1 sandwich for the gmina block diagonal (D_g)
    e = resid
    Xe = X.multiply(e[:, None]).tocsr()
    meat = (Xe.T @ Xe).toarray()
    scale = len(y) / dof
    cov = XtX_inv @ meat @ XtX_inv * scale
    D_g = np.clip(np.diag(cov)[:nG], 1e-9, None)

    theta_hat = beta[:nG]
    counts = np.asarray(G.sum(axis=0)).ravel()

    direct = pd.DataFrame(
        {
            "gmina_teryt": list(gcat.categories),
            "theta_hat": theta_hat,
            "D_g": D_g,
            "n_obs": counts.astype(int),
        }
    )

    ctrl = dict(zip(znames, beta[nG + nT: nG + nT + nZ]))
    time_labels = [lab for lab in tcat.categories if lab not in ref_set]
    time_coef = dict(zip(time_labels, beta[nG: nG + nT]))
    info = {
        "control_coef": ctrl,
        "control_means": dict(zip(znames, zmean)),
        "time_ref": ref,
        "time_coef": time_coef,
        "sigma2": sigma2,
        "r2": r2,
        "n": len(micro),
        "n_gminas": nG,
    }
    LOGGER.info("Hedonic fit: R2=%.3f, sigma2=%.4f, gminas with direct est=%d", r2, sigma2, nG)
    LOGGER.info("  key coefs: log_area=%.3f house=%.3f primary=%.3f",
                ctrl.get("log_area", np.nan), ctrl.get("house", np.nan), ctrl.get("primary", np.nan))
    return direct, info


# --------------------------------------------------------------------------- #
# Stage 4: Fay-Herriot small-area model
# --------------------------------------------------------------------------- #
def _safe_solve(Amat, b):
    try:
        return np.linalg.solve(Amat, b)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(Amat) @ b


def _safe_inv(Amat):
    try:
        return np.linalg.inv(Amat)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(Amat)


def _reml_A(theta, X, D, tol=1e-8, max_iter=200):
    """Fay-Herriot moment/REML estimate of the model variance A >= 0 and GLS beta."""
    n, p = X.shape
    A = max(np.var(theta) - np.mean(D), 1e-6)
    for _ in range(max_iter):
        w = 1.0 / (A + D)
        XtWX = (X * w[:, None]).T @ X
        XtWy = (X * w[:, None]).T @ theta
        beta = _safe_solve(XtWX, XtWy)
        r = theta - X @ beta
        # moment condition: sum r^2/(A+D) = n - p
        lhs = np.sum(r ** 2 / (A + D))
        target = max(n - p, 1)
        f = lhs - target
        df = -np.sum(r ** 2 / (A + D) ** 2)          # monotone decreasing in A
        step = f / df if df != 0 else 0.0
        A_new = max(A - step, 1e-8)
        if abs(A_new - A) < tol:
            A = A_new
            break
        A = A_new
    w = 1.0 / (A + D)
    XtWX = (X * w[:, None]).T @ X
    XtWX_inv = _safe_inv(XtWX)
    beta = XtWX_inv @ ((X * w[:, None]).T @ theta)
    return A, beta, XtWX_inv


def fay_herriot(
    direct: pd.DataFrame,
    cov: pd.DataFrame,
    covariate_cols,
    anchor_col: str = "log_gus",
):
    """Two-stage small-area estimator.

    Parameters
    ----------
    direct : DataFrame [gmina_teryt, theta_hat, D_g, n_obs]  (data gminas only)
    cov    : DataFrame with one row per *all* gminas and the covariate columns
             (must include ``gmina_teryt`` and every name in covariate_cols).
    covariate_cols : list of covariate column names (incl. the GUS anchor).

    Returns
    -------
    DataFrame over all gminas with theta_tilde (posterior log price/m2),
    theta_sd, shrinkage weight, source flag.
    """
    all_g = cov[["gmina_teryt"] + covariate_cols].copy()
    df = all_g.merge(direct, on="gmina_teryt", how="left")

    # design matrix (intercept + covariates); impute missing covariates by column
    # mean and drop any constant (zero-variance) covariate to keep the design full rank.
    xcols = [np.ones(len(df))]
    used = ["(intercept)"]
    for c in covariate_cols:
        v = _impute(df[c].to_numpy(float))
        if np.nanstd(v) > 1e-12:
            xcols.append(v)
            used.append(c)
        else:
            LOGGER.info("Fay-Herriot: dropping constant covariate '%s'", c)
    Xall = np.column_stack(xcols)
    LOGGER.info("Fay-Herriot design covariates: %s", used)

    has_direct = df["theta_hat"].notna().to_numpy()
    Xd = Xall[has_direct]
    theta = df.loc[has_direct, "theta_hat"].to_numpy(float)
    D = df.loc[has_direct, "D_g"].to_numpy(float)

    A, beta, XtWX_inv = _reml_A(theta, Xd, D)
    LOGGER.info("Fay-Herriot: A=%.4f (model sd=%.3f), n_direct=%d, p=%d",
                A, np.sqrt(A), len(theta), Xd.shape[1])

    mu = Xall @ beta                          # prior mean for every gmina
    theta_tilde = mu.copy()
    theta_var = np.empty(len(df))
    gamma = np.zeros(len(df))

    # data gminas: shrink toward mu
    r = theta - Xd @ beta
    g_w = A / (A + D)
    theta_tilde[has_direct] = mu[has_direct] + g_w * r
    gamma[has_direct] = g_w
    # Prasad-Rao MSE: g1 + g2 (+ small g3 omitted for stability)
    g1 = g_w * D
    g2 = np.einsum("ij,jk,ik->i", Xd, XtWX_inv, Xd) * (1 - g_w) ** 2
    theta_var[has_direct] = g1 + g2

    # zero-data gminas: prediction from covariates, variance = A + x'Var(beta)x
    nd = ~has_direct
    Xz = Xall[nd]
    theta_var[nd] = A + np.einsum("ij,jk,ik->i", Xz, XtWX_inv, Xz)

    out = df[["gmina_teryt", "n_obs"] + covariate_cols].copy()
    out["theta_tilde"] = theta_tilde
    out["theta_sd"] = np.sqrt(np.clip(theta_var, 0, None))
    out["shrinkage_to_data"] = gamma        # 1 = pure RCN, 0 = pure GUS/model
    out["has_rcn"] = has_direct
    out["index_zl_m2"] = np.exp(theta_tilde)
    return out


def _impute(v):
    v = v.astype(float)
    m = np.nanmean(v) if np.isfinite(v).any() else 0.0
    return np.where(np.isfinite(v), v, m)


def benchmark_to_gus(index_df: pd.DataFrame, gus_anchor: pd.DataFrame, weight_col="pop"):
    """Optional exact within-powiat rescale so the weighted mean of gmina
    indices matches the GUS powiat 2021 value."""
    df = index_df.copy()
    df["powiat"] = df["gmina_teryt"].str[:4]
    w = df[weight_col].to_numpy(float) if weight_col in df else np.ones(len(df))
    df["_w"] = np.where(np.isfinite(w) & (w > 0), w, 1.0)
    g = gus_anchor.set_index("powiat")["gus_median"].to_dict()
    factors = {}
    for pw, sub in df.groupby("powiat"):
        target = g.get(pw, np.nan)
        if not np.isfinite(target):
            continue
        wmean = np.average(sub["index_zl_m2"], weights=sub["_w"])
        if wmean > 0:
            factors[pw] = target / wmean
    df["index_zl_m2"] = df.apply(
        lambda r: r["index_zl_m2"] * factors.get(r["powiat"], 1.0), axis=1
    )
    df["theta_tilde"] = np.log(df["index_zl_m2"])
    LOGGER.info("Benchmarked to GUS in %d powiats", len(factors))
    return df.drop(columns=["_w"])
