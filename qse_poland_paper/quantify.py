"""
quantify.py — model inversion (recover fundamentals from observables).

Direct port of the verified reference (mrrh_pipeline/quantify.py ==
progs/solveProductTradeTK.m, SW2020 eqs. 10 & 12; and the ARSW-style amenity
residual). Matrix orientation is [n, i] = [residence, workplace] for commuting
and [n, i] with spending destination i in columns for trade shares, identical to
the reference.

(a) solve_product_trade -> fundamental productivity A_n, trade shares pi_ni,
    tradeshOwn, tradable price index P_n.
(b) commuter_market_access / qol_residual -> commuter market access CMA_n and
    residential amenity b_n (geometric-mean-normalised).
"""
from __future__ import annotations

import numpy as np


def solve_product_trade(L_n, R_n, w_n, v_n, dni,
                        sigg, nu, fixC=1.0,
                        relax=0.25, maxiter=5000, prec=6, tol=1e-8):
    """Recover A_n satisfying income = expenditure (SW2020 eq. 12).

    Returns (A_n, tradesh[n,i] (destination i in columns), tradeshOwn, P_n,
    iters, final_gap, converged).
    """
    rrho = nu
    n = len(L_n)
    product = np.ones(n)
    income = w_n * L_n
    it = 0
    expend = income.copy()
    converged = False
    for it in range(maxiter):
        num = product ** (sigg - 1) * L_n ** (1 - (1 - sigg) * rrho) * w_n ** (1 - sigg)
        nummat = dni ** (1 - sigg) * np.tile(num, (n, 1))      # [n, i], origins in cols
        tradesh = nummat / nummat.sum(axis=1, keepdims=True)
        expend = tradesh.T @ (v_n * R_n)
        gap = np.abs(income - expend)
        if np.all(np.round(gap, prec) == 0) or np.max(gap) < tol:
            converged = True
            break
        product = relax * (product * (income / expend)) + (1 - relax) * product
        product = product / product.mean()

    # final trade shares with spending DESTINATION i in columns, plus P_n
    num = product ** (sigg - 1) * L_n ** (1 - (1 - sigg) * rrho) * w_n ** (1 - sigg)
    nummat = dni ** (1 - sigg) * np.tile(num[:, None], (1, n))
    tradesh = nummat / nummat.sum(axis=0, keepdims=True)
    tradeshOwn = np.diag(tradesh)
    P_n = (sigg / (sigg - 1)
           * (L_n / (sigg * fixC * tradeshOwn)) ** (1 / (1 - sigg))
           * (w_n / product))
    return (product, tradesh, tradeshOwn, P_n, it,
            float(np.max(np.abs(income - expend))), converged)


def commuter_market_access(tau, w_n, epsi, mu):
    """CMA_n = Σ_i tau_ni**(-epsi*mu) w_i**epsi  (sum over workplaces i)."""
    return (tau ** (-epsi * mu) * (w_n ** epsi)[None, :]).sum(axis=1)


def qol_residual(R_n, L, P_n, Q_n, tau, w_n, alp, epsi, mu):
    """Residential amenity b_n (geomean 1), plus CMA_n and CPI_n = P^alp Q^(1-alp).

        b_n ∝ (R_n/L) * (P_n^alp Q_n^(1-alp))^epsi / CMA_n
    """
    CMA = commuter_market_access(tau, w_n, epsi, mu)
    CPI = P_n ** alp * Q_n ** (1 - alp)
    b = (R_n / L) * (CPI ** epsi) / CMA
    b = b / np.exp(np.log(b).mean())          # geometric mean 1
    return b, CMA, CPI
