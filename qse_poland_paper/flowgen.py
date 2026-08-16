"""
flowgen.py — generate a bilateral commuting matrix when flows are unobserved.

Port of the toolkit's ``getBiTK.m`` (used by ``OwnData.m``). Given workplace
wages, a commuting-cost matrix, and the residence/workplace employment margins,
it inverts a workplace attractiveness B_i that rationalises observed workplace
employment under the model's conditional commuting probabilities, then builds the
implied unconditional flows.

This is the path for the 2026 cross-section, which has all data except the
gmina×gmina commuting matrix. The commuting decay phi (= epsi*mu) is borrowed
from the 2021 calibration, per the project design.

    lambda_{i|n} ∝ B_i w_i**epsi tau_ni**(-epsi*mu)     (row-normalised over i)
    L_i_hat      = Σ_n lambda_{i|n} R_n
    B_i          <- B_i * (L_i / L_i_hat)               (fixed point)
    uncondCom    = (R_n/ΣR_n) ⊙ lambda_{i|n}
    comMat       = uncondCom * Lbar
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GeneratedFlows:
    comMat: np.ndarray        # (N,N) [res, work]
    cond: np.ndarray          # (N,N) lambda_{i|n}, rows sum to 1
    B_i: np.ndarray           # (N,) recovered workplace attractiveness
    L_i_hat: np.ndarray       # (N,) predicted workplace employment
    iters: int
    converged: bool
    max_emp_err: float


def generate_flows(w_n, tau, R_n, L_n, Lbar, epsi, mu,
                   maxiter=2000, tol=1e-3):
    """Return GeneratedFlows. R_n/L_n are the (normalised) residence/workplace
    margins; Lbar the total worker mass they should scale to (Σ R_n)."""
    n = len(w_n)
    cost = tau ** (-epsi * mu)                     # [n, i]
    wpow = (w_n ** epsi)[None, :]                  # [1, i]
    B = np.ones(n)
    L_i_hat = L_n.copy()
    converged = False
    it = 0
    for it in range(maxiter):
        num = B[None, :] * wpow * cost             # [n, i]
        cond = num / num.sum(axis=1, keepdims=True)
        L_i_hat = cond.T @ R_n                     # predicted workplace employment
        obj = np.sum(np.abs(L_n - L_i_hat)) * 100.0
        if obj < tol:
            converged = True
            break
        B = B * (L_n / L_i_hat)

    num = B[None, :] * wpow * cost
    cond = num / num.sum(axis=1, keepdims=True)
    L_i_hat = cond.T @ R_n
    choiceR = (R_n / R_n.sum())[:, None]
    uncond = choiceR * cond
    comMat = uncond * Lbar
    return GeneratedFlows(comMat=comMat, cond=cond, B_i=B, L_i_hat=L_i_hat,
                          iters=it, converged=converged,
                          max_emp_err=float(np.max(np.abs(L_n - L_i_hat))))
