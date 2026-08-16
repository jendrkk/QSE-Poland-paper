"""
io/trade.py — bilateral iceberg trade cost d_ni.

No subnational bilateral goods-flow data exist for Poland, so d_ni is a modelling
construct. Following the multimodal-extension note (§9) and the toolkit's
``OwnData.m`` convention, we build it from the road travel-time matrix:

    d_ni = (tau_ni / min(tau))**psi ,     d_nn set from its own tau (>= 1)

This is single-mode (road) freight, which captures the first-order motorway
effect; a freight modal aggregator is a documented future seam. A pre-computed
distance/cost matrix can be supplied instead via ``from_matrix``.
"""
from __future__ import annotations

import numpy as np


def build_dni(tau: np.ndarray, psi: float, symmetrize: bool = True,
              force_diag_one: bool = False) -> np.ndarray:
    """d_ni = (t/min t)**psi, where t is the (optionally symmetrised) travel time.

    Trade costs are conventionally symmetric; symmetrising the travel time fed to
    d_ni (geometric mean of t_ni and t_in) removes the small internal-consistency
    residual that road-time asymmetry would otherwise induce in the inversion,
    while leaving the commuting cost tau (used for kappa and market access)
    untouched.
    """
    t = np.sqrt(tau * tau.T) if symmetrize else tau
    tmin = float(np.min(t[t > 0]))
    dni = (t / tmin) ** psi
    if force_diag_one:
        np.fill_diagonal(dni, 1.0)
    return dni


def from_matrix(mat: np.ndarray, psi: float | None = None, already_cost: bool = True,
                force_diag_one: bool = True) -> np.ndarray:
    """Use a supplied N×N matrix as the trade cost. If already_cost, use as-is;
    otherwise treat as distance and raise to psi."""
    mat = np.asarray(mat, dtype=float)
    if already_cost:
        dni = mat.copy()
    else:
        if psi is None:
            raise ValueError("psi required when already_cost=False")
        dni = (mat / float(np.min(mat[mat > 0]))) ** psi
    if force_diag_one:
        np.fill_diagonal(dni, 1.0)
    return dni
