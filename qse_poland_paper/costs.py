"""
costs.py — the commuting-cost aggregator (extension seam).

In the baseline (single road mode, no congestion) the effective commuting cost
is simply the observed road travel time: ``effective_tau(tau) == tau``. The whole
point of the MRRH multimodal extension is that *only this function* changes when
modes/congestion are added — the trade block, price index, housing market, GE
fixed point and welfare formula are untouched because they see ``tau`` (or its
effective replacement τ̃) and nothing else.

The seams below are implemented as clearly-typed stubs so the wiring exists but
the baseline never activates them.
"""
from __future__ import annotations

import numpy as np


def effective_tau(tau: np.ndarray, params=None, extra_modes=None,
                  congestion=None) -> np.ndarray:
    """Baseline: return road travel time unchanged. Hooks are inert unless given."""
    if extra_modes:
        return modal_aggregate(tau, extra_modes, params)
    if congestion is not None:
        raise NotImplementedError("congestion is a documented future seam (Extension IV)")
    return tau


def modal_aggregate(t_car: np.ndarray, modes, params) -> np.ndarray:
    """Nested-Fréchet effective commuting time τ̃ (multimodal note eq. 14).

    τ̃_ni = [ Σ_m e^{-epsi*zeta_m*s} (t^m_ni)^{-phi*s} ]^{-1/(phi*s)} · ... with
    s = 1/(1-rho). Implemented for the general finite mode set; NOT used in the
    baseline. `modes` = list of dicts {t: N×N array, zeta: float}, with t_car the
    reference (zeta_car = 0). `params` must carry phi, epsi, rho.
    """
    raise NotImplementedError(
        "modal_aggregate is the rail/mode-choice seam; not part of the baseline. "
        "Provide phi, epsi, rho and per-mode (t, zeta), then implement eq. 14 here.")


def congestion(tau_free: np.ndarray, car_flow, capacity, xi: float) -> np.ndarray:
    """BPR-type congested car time (multimodal note eq. 22). Future seam."""
    raise NotImplementedError("congestion inner fixed point is a documented future seam")
