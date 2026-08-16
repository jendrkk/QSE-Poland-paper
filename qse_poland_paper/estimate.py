"""
estimate.py — in-sample estimation of the commuting decay phi (= epsi*mu).

Commuting gravity with origin AND destination fixed effects (multimodal note
eq. 15):

    log lambda_ni = a_n + b_i - phi * log(tau_ni) + u_ni

The slope on log tau identifies -phi. This is the one parameter that must never
be imported; epsilon is then calibrated and mu = phi/epsi is the residual.

The two-way fixed effects are absorbed by alternating within-transformations
(exact for two factors), then phi is a univariate OLS slope on the double-demeaned
variables. Estimation uses off-diagonal pairs with strictly positive flows
(configurable). Robust (HC1) standard error reported.

Future seams (declared, not implemented): delta_iv (housing supply elasticity
via a Saiz-type instrument) and modal_params ((rho, zeta_rail) by station-level
minimum distance).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GravityFit:
    phi: float
    se: float
    r2: float
    n_pairs: int
    include_diagonal: bool


def _absorb_two_way(rows, cols, y, x, n, maxiter=200, tol=1e-10):
    """Return double-demeaned (y, x) after absorbing row and col fixed effects,
    via alternating projections over the observed pairs."""
    y = y.astype(float).copy()
    x = x.astype(float).copy()

    def demean(v):
        v = v.copy()
        for _ in range(maxiter):
            # row means
            rsum = np.bincount(rows, weights=v, minlength=n)
            rcnt = np.bincount(rows, minlength=n)
            v = v - np.where(rcnt[rows] > 0, rsum[rows] / np.maximum(rcnt[rows], 1), 0.0)
            # col means
            csum = np.bincount(cols, weights=v, minlength=n)
            ccnt = np.bincount(cols, minlength=n)
            adj = np.where(ccnt[cols] > 0, csum[cols] / np.maximum(ccnt[cols], 1), 0.0)
            v = v - adj
            if np.max(np.abs(adj)) < tol:
                break
        return v

    return demean(y), demean(x)


def estimate_phi(comMat, tau, include_diagonal=False,
                 maxiter=200, tol=1e-10) -> GravityFit:
    """Estimate phi from the commuting gravity. comMat is [res, work]; any positive
    monotone transform of flows (counts or probabilities) gives the same slope."""
    n = comMat.shape[0]
    mask = comMat > 0
    if not include_diagonal:
        np.fill_diagonal(mask, False)
    rows, cols = np.where(mask)
    y = np.log(comMat[rows, cols])
    x = np.log(tau[rows, cols])

    yd, xd = _absorb_two_way(rows, cols, y, x, n, maxiter, tol)

    sxx = float(xd @ xd)
    beta = float(xd @ yd) / sxx
    resid = yd - beta * xd
    k = 2 * n - 1 + 1                       # (row + col FE, less 1) + slope
    dof = max(len(y) - k, 1)
    # HC1 robust SE on the partialled-out regressor
    xx_inv = 1.0 / sxx
    meat = float((xd ** 2) @ (resid ** 2))
    var_hc1 = xx_inv * meat * xx_inv * (len(y) / dof)
    se = float(np.sqrt(var_hc1))
    ss_tot = float(yd @ yd)
    r2 = 1.0 - float(resid @ resid) / ss_tot if ss_tot > 0 else float("nan")

    return GravityFit(phi=-beta, se=se, r2=r2, n_pairs=int(len(y)),
                      include_diagonal=include_diagonal)


# ---- future seams ---------------------------------------------------------- #
def delta_iv(*args, **kwargs):
    raise NotImplementedError("housing-supply elasticity IV is a documented future seam")


def modal_params(*args, **kwargs):
    raise NotImplementedError("(rho, zeta_rail) identification is a documented future seam")
