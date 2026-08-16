"""
counterfac.py — exact-hat-algebra counterfactual solver.

Direct port of the verified reference (mrrh_pipeline/counterfac.py ==
progs/counterFactsTK.m and the eight updateXxxTK.m functions, Codebook §A.2).
Forcing variables are relative changes (hats): aChange (N productivity),
bChange (N×N residence amenity), kapChange (N×N commuting cost), dChange (N×N
trade cost). Unchanged primitives are ones.

The road-network general-equilibrium experiment at the heart of the paper is a
pure commuting-cost shock: ``kapChange = (tau'/tau)`` (and, if trade cost is
rebuilt from tau, ``dChange = (dni'/dni)``), holding fundamentals fixed. See
``build_tau_change`` / ``network_counterfactual``.
"""
from __future__ import annotations

import numpy as np


# ---- inner-loop updates (verbatim structure from *TK.m) -------------------- #
def _upd_res_wage(bC, wC, kapC, lamObs, vObs, wObs, epsi):
    numr = (bC * lamObs * kapC ** (-epsi)) @ (wC ** (1 + epsi) * wObs)
    den = (bC * lamObs * kapC ** (-epsi)) @ (wC ** epsi)
    return (numr / den) / vObs


def _upd_empl(lamC, lamObs, lObs, lBar):
    return lBar * ((lamObs * lamC).sum(axis=0) / lObs)


def _upd_resid(lamC, lamObs, rObs, lBar):
    return lBar * ((lamObs * lamC).sum(axis=1) / rObs)


def _upd_house(vC, rC, delta):
    return (vC * rC) ** (1.0 / (1.0 + delta))


def _upd_tradesh(lC, dC, wC, aC, piObs, sigg, nu):
    n = len(lC)
    num = aC ** (sigg - 1) * lC ** (1 - (1 - sigg) * nu) * wC ** (1 - sigg)
    nummat = dC ** (1 - sigg) * np.tile(num[:, None], (1, n))
    denom = (piObs * nummat).sum(axis=0)
    return nummat / np.tile(denom, (n, 1))


def _upd_prices(lC, wC, piC, aC, dC, sigg, nu):
    return (lC ** (1 - (1 - sigg) * nu) / np.diag(piC)) ** (1.0 / (1 - sigg)) \
        * np.diag(dC) * wC / aC


def _upd_wage(lC, piC, vC, rC, lObs, wObs, piObs, vObs, rObs):
    n = len(lC)
    nummat = piObs * piC
    vr = np.tile((vC * rC * vObs * rObs)[None, :], (n, 1))
    num = (nummat * vr).sum(axis=1)
    denom = wObs * lObs * lC
    return num / denom


def _upd_lam(bC, pC, qC, wC, kapC, lamObs, alp, epsi):
    n = len(pC)
    pq = np.tile((pC ** alp * qC ** (1 - alp))[:, None], (1, n))
    wm = np.tile(wC[None, :], (n, 1))
    nummat = bC * pq ** (-epsi) * (wm / kapC) ** epsi
    denom = (lamObs * nummat).sum()
    return nummat / denom


# ---- outer solver ---------------------------------------------------------- #
def counter_facts(aChange, bChange, kapChange, dChange,
                  wObs, vObs, lamObs, lObs, rObs, piObs,
                  alp, epsi, delta, sigg, nu,
                  tol=1e-4, maxit=100_000, relax=0.25):
    """Solve for relative changes. Returns dict of hats + welfare scalar + iters.

    Under identity forcings (all changes == 1) this must return every hat == 1
    and welfare == 1; that identity is the primary correctness test.
    """
    n = len(aChange)
    lBar = lObs.sum()
    wC = np.ones(n)
    lamC = np.ones((n, n))
    k = 0
    converged = False
    for k in range(maxit):
        vC = _upd_res_wage(bChange, wC, kapChange, lamObs, vObs, wObs, epsi)
        lC = _upd_empl(lamC, lamObs, lObs, lBar)
        rC = _upd_resid(lamC, lamObs, rObs, lBar)
        qC = _upd_house(vC, rC, delta)
        piC = _upd_tradesh(lC, dChange, wC, aChange, piObs, sigg, nu)
        pC = _upd_prices(lC, wC, piC, aChange, dChange, sigg, nu)
        wT = _upd_wage(lC, piC, vC, rC, lObs, wObs, piObs, vObs, rObs)
        wnew = (wT * wObs) / np.mean(wT * wObs)
        wT = wnew / wObs
        lamT = _upd_lam(bChange, pC, qC, wC, kapChange, lamObs, alp, epsi)
        if np.all(np.abs(wC - wT) < tol) and np.all(np.abs(lamC - lamT) < tol):
            wC, lamC = wT, lamT
            converged = True
            break
        wC = relax * wT + (1 - relax) * wC
        lamC = relax * lamT + (1 - relax) * lamC

    vC = _upd_res_wage(bChange, wC, kapChange, lamObs, vObs, wObs, epsi)
    lC = _upd_empl(lamC, lamObs, lObs, lBar)
    rC = _upd_resid(lamC, lamObs, rObs, lBar)
    qC = _upd_house(vC, rC, delta)
    piC = _upd_tradesh(lC, dChange, wC, aChange, piObs, sigg, nu)
    pC = _upd_prices(lC, wC, piC, aChange, dChange, sigg, nu)

    pq = np.tile((pC ** alp * qC ** (1 - alp))[:, None], (1, n))
    wm = np.tile(wC[None, :], (n, 1))
    welf = bChange ** (1.0 / epsi) * (kapChange * pq) ** (-1) * wm * lamC ** (-1.0 / epsi)

    return dict(w=wC, v=vC, q=qC, pi=piC, lam=lamC, p=pC, r=rC, l=lC,
                welf=float(welf[0, 0]), welf_mat_sd=float(np.std(welf)),
                iters=k, converged=converged)


# ---- convenience builders for the road-network experiment ------------------ #
def build_tau_change(tau_base: np.ndarray, tau_new: np.ndarray) -> np.ndarray:
    """kapChange_ni = tau_new / tau_base (commuting-cost hat, in *time* units;
    tau**(-phi) is applied consistently inside the model)."""
    return tau_new / tau_base


def network_counterfactual(base_run, tau_new, dni_new=None, **overrides):
    """Run a road-network counterfactual on a solved baseline RunResult.

    kapChange = tau_new / tau_base. If dni_new is given, dChange = dni_new/dni_base;
    otherwise trade costs are held fixed (commuting-only shock).
    """
    inp = base_run.inputs
    par = base_run.params
    n = base_run.frame["N"]
    f64 = lambda a: np.asarray(a, dtype=np.float64)   # upcast float32-stored arrays
    tau_new = f64(tau_new)
    kapChange = build_tau_change(f64(inp["tau"]), tau_new)
    dChange = np.ones((n, n)) if dni_new is None else (f64(dni_new) / f64(inp["dni"]))
    return counter_facts(
        aChange=np.ones(n), bChange=np.ones((n, n)),
        kapChange=kapChange, dChange=dChange,
        wObs=f64(base_run.calibrated["w_n"]), vObs=f64(base_run.calibrated["v_n"]),
        lamObs=f64(inp["uncondCom"]), lObs=f64(inp["L_n"]), rObs=f64(inp["R_n"]),
        piObs=f64(base_run.calibrated["tradesh"]),
        alp=par["alpha"], epsi=par["epsi"], delta=par["delta"],
        sigg=par["sigg"], nu=par["nu"],
        **overrides)
