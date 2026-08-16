"""
solve.py — calibrate and solve the MRRH model for one year.

calibrate_year(cfg, year, borrowed_phi=None) performs the full baseline pipeline
for a single cross-section and returns an exhaustive RunResult:

  1. build the canonical 2021 frame (identical order across years)
  2. load labour (w, workplace/residence employment), travel time, floor-space price
  3. assemble the commuting matrix
        - observed flows (2011, 2021): off-diagonal harmonised to the 2021 frame,
          diagonal recovered as own_n = max(R_n - outflows_n, 0)
        - no flows (2026): generate via flowgen using a borrowed phi
  4. build model observables with the MRRH normalisations
  5. estimate phi from the commuting gravity (observed years); borrow it (2026)
  6. invert productivities/trade shares/price index (quantify.solve_product_trade)
  7. recover residential amenity b_n and market access (quantify.qol_residual)
  8. run invariant checks
"""
from __future__ import annotations

import numpy as np

from . import __version__, quantify, validate, flowgen, estimate as est
from .config import RunConfig
from .costs import effective_tau
from .frame import build_frame
from .io.labour import load_labour
from .io.ttm import load_ttm
from .io.floorspace import load_floorspace
from .io.trade import build_dni
from .io.flows import load_flows, ipf_reconcile
from .result import RunResult


def _normalise_margins(comMat):
    """MRRH normalisation from a commuting matrix (mirrors ReadData.m)."""
    L = float(comMat.sum())
    uncond = comMat / L
    cond = uncond / uncond.sum(axis=1, keepdims=True)
    L_n = uncond.sum(axis=0) * L
    L_n = L_n / L_n.mean()
    R_n = uncond.sum(axis=1) * L
    R_n = R_n / R_n.mean()
    return L, uncond, cond, L_n, R_n


def calibrate_year(cfg: RunConfig, year: int, borrowed_phi: float | None = None) -> RunResult:
    P = cfg.params
    spec = cfg.years[year]
    ap = cfg.paths.abspath

    # 1. frame (deterministic canonical order; same across years) ------------- #
    fr = build_frame(ap(spec.wages), ap(spec.floorspace))

    # 2. inputs --------------------------------------------------------------- #
    lab = load_labour(ap(spec.wages), fr)
    ttm = load_ttm(ap(spec.ttm), fr, params=P)
    tau = effective_tau(ttm.tau, params=P)                 # baseline: == tau
    fs = load_floorspace(ap(spec.floorspace), fr)
    Q_n = fs.Q_n

    diagnostics: dict = {}
    flow_diag_clip = {}
    generated = spec.flows is None
    gravity_flows = None            # OBSERVED off-diagonal flows used for the gravity

    # 3-4. commuting matrix + normalised margins ------------------------------ #
    if not generated:
        fl = load_flows(ap(spec.flows), spec.flow_source_year or year, fr, ap(cfg.paths.crosswalk))
        own = np.maximum(lab.Rr_n - fl.outflows, 0.0)      # diagonal recovery
        n_clip = int(np.sum(lab.Rr_n - fl.outflows < 0))
        clip_mass = float(np.sum(np.maximum(fl.outflows - lab.Rr_n, 0.0)))
        comMat = fl.off_diag.copy()
        np.fill_diagonal(comMat, own)
        gravity_flows = fl.off_diag            # genuine observed data for the gravity
        # cross-check workplace margin against labour BEFORE reconciliation
        work_from_flows = comMat.sum(axis=0)
        wp_rel_err = float(np.max(np.abs(work_from_flows - lab.Lw_n)
                                  / np.maximum(lab.Lw_n, 1.0)))
        diagonal_alt = np.maximum(lab.Lw_n - fl.inflows, 0.0)   # L_n - inflows check
        diag_recon = float(np.max(np.abs(own - diagonal_alt) / np.maximum(own, 1.0)))
        flow_diag_clip = dict(diagonal_clipped_units=n_clip, diagonal_clipped_mass=clip_mass,
                              workplace_margin_rel_err=wp_rel_err,
                              diagonal_reconciliation_rel=diag_recon,
                              own_share_preIPF=float(own.sum() / comMat.sum()),
                              zero_workplace_units=int((work_from_flows == 0).sum()))
        diagnostics["flows"] = fl.diagnostics
        comMat_observed = comMat.copy()        # observed (pre-IPF) for gravity/own-share viz
        # Reconcile to trusted labour margins (phi-invariant; fixes censoring/vintage)
        if cfg.solver.reconcile_margins:
            comMat, ipf_diag = ipf_reconcile(
                comMat, row_target=lab.Rr_n, col_target=lab.Lw_n, tau=tau,
                maxiter=cfg.solver.ipf_maxiter, tol=cfg.solver.ipf_tol)
            flow_diag_clip.update(ipf_diag)
            flow_diag_clip["reconciled_to_labour_margins"] = True
        flow_diag_clip["own_commute_share"] = float(np.diag(comMat).sum() / comMat.sum())
        L, uncond, cond, L_n, R_n = _normalise_margins(comMat)
        w_n = lab.w_n / lab.w_n.mean()
        v_n = cond @ w_n
    else:
        # 2026: generate flows from margins + travel times with borrowed phi
        if borrowed_phi is None:
            src = spec.borrow_params_from
            raise ValueError(f"year {year} has no flows; a borrowed phi from {src} "
                             "must be supplied (run that year first)")
        phi_used = borrowed_phi
        mu_used = phi_used / P.epsi
        L_n0 = lab.Lw_n / lab.Lw_n.mean()
        R_n0 = lab.Rr_n / lab.Rr_n.mean()
        w_n = lab.w_n / lab.w_n.mean()
        gen = flowgen.generate_flows(w_n, tau, R_n0, L_n0, Lbar=R_n0.sum(),
                                     epsi=P.epsi, mu=mu_used,
                                     maxiter=cfg.solver.flowgen_maxiter,
                                     tol=cfg.solver.flowgen_tol)
        comMat = gen.comMat
        comMat_observed = comMat            # generated flows are the "observed" object here
        L, uncond, cond, L_n, R_n = _normalise_margins(comMat)
        v_n = cond @ w_n
        diagnostics["flowgen"] = dict(iters=gen.iters, converged=gen.converged,
                                      max_emp_err=gen.max_emp_err,
                                      own_commute_share=float(np.diag(comMat).sum() / comMat.sum()))

    # trade cost
    dni = build_dni(tau, P.psi)

    # 5. phi -------------------------------------------------------------------#
    if generated:
        phi = borrowed_phi
        grav = dict(phi=phi, source=f"borrowed from {spec.borrow_params_from}")
    elif P.phi_mode == "estimate":
        # estimate on OBSERVED off-diagonal flows (never the IPF-reconciled matrix,
        # whose synthetic floor would contaminate the decay)
        gf = est.estimate_phi(gravity_flows, tau,
                              include_diagonal=cfg.solver.gravity_include_diagonal,
                              maxiter=cfg.solver.gravity_fe_maxiter,
                              tol=cfg.solver.gravity_fe_tol)
        phi = gf.phi
        grav = dict(phi=gf.phi, se=gf.se, r2=gf.r2, n_pairs=gf.n_pairs,
                    include_diagonal=gf.include_diagonal, source="gravity")
    else:
        phi = P.phi
        grav = dict(phi=phi, source="fixed (epsi*mu)")
    mu_eff = phi / P.epsi                                   # resolved time elasticity

    # 6. productivity inversion + trade ---------------------------------------#
    A_n, tradesh, tradeshOwn, P_idx, it_prod, gap_prod, conv_prod = quantify.solve_product_trade(
        L_n, R_n, w_n, v_n, dni, sigg=P.sigg, nu=P.nu, fixC=P.fixC,
        relax=cfg.solver.prod_relax, maxiter=cfg.solver.prod_maxiter,
        prec=cfg.solver.prod_precision, tol=cfg.solver.prod_tol)

    # 7. amenity + market access ----------------------------------------------#
    b_n, CMA, CPI = quantify.qol_residual(R_n, L, P_idx, Q_n, tau, w_n,
                                          alp=P.alpha, epsi=P.epsi, mu=mu_eff)

    # derived objects for visualization
    real_v = v_n / CPI                                     # real residence income
    # income = expenditure check in the EQUILIBRIUM orientation (row-normalised
    # expenditure shares, i.e. the in-loop trade shares), which is the condition
    # the solver actually enforces. The returned `tradesh` is the column-normalised
    # variant used as piObs by the counterfactual solver.
    _num = A_n ** (P.sigg - 1) * L_n ** (1 - (1 - P.sigg) * P.nu) * w_n ** (1 - P.sigg)
    _pi_eq = dni ** (1 - P.sigg) * np.tile(_num, (fr.N, 1))
    _pi_eq = _pi_eq / _pi_eq.sum(axis=1, keepdims=True)
    income = w_n * L_n
    expend = _pi_eq.T @ (v_n * R_n)

    # 8. validation ------------------------------------------------------------#
    checks = validate.check(N=fr.N, L_n=L_n, R_n=R_n, uncondCom=uncond,
                            A_n=A_n, b_n=b_n, tau=tau, income=income, expend=expend)

    # assemble RunResult -------------------------------------------------------#
    params_used = dict(alpha=P.alpha, epsi=P.epsi, mu=mu_eff, phi=phi, sigg=P.sigg,
                       nu=P.nu, delta=P.delta, psi=P.psi, fixC=P.fixC,
                       phi_mode=P.phi_mode, sensitivity=P.sensitivity)
    diagnostics.update(dict(prod_iters=it_prod, prod_gap=gap_prod, prod_converged=conv_prod,
                            **flow_diag_clip))

    rr = RunResult.new(
        run_id=cfg.run_id(year), year=year, package_version=__version__,
        meta=dict(flows_source=("generated" if generated else "observed"),
                  flow_file=(None if generated else str(ap(spec.flows))),
                  flow_source_year=spec.flow_source_year,
                  ttm_file=str(ap(spec.ttm)), ttm_meta=ttm.meta,
                  ttm_tag=spec.ttm_tag,
                  wages_file=str(ap(spec.wages)),
                  floorspace_file=str(ap(spec.floorspace)),
                  floorspace_single_timestamp=fs.single_timestamp,
                  borrow_params_from=spec.borrow_params_from,
                  tag=cfg.tag),
        params=params_used,
        frame=dict(N=fr.N, codes=fr.codes, nazwa=fr.nazwa, powiat=fr.powiat,
                   woj=fr.woj, rodz_class=fr.rodz_class, area_km2=fr.area_km2, pop=fr.pop),
        inputs=dict(comMat=comMat, comMat_observed=comMat_observed, uncondCom=uncond,
                    condCom=cond, L=L, w_n=w_n, v_n=v_n, L_n=L_n, R_n=R_n,
                    tau=tau, dni=dni, Q_n=Q_n,
                    Lw_raw=lab.Lw_n, Rr_raw=lab.Rr_n, w_raw=lab.w_n),
        calibrated=dict(A_n=A_n, tradesh=tradesh, tradeshOwn=tradeshOwn, P_n=P_idx,
                        b_n=b_n, CMA=CMA, CPI=CPI, w_n=w_n, v_n=v_n, real_v=real_v,
                        income=income, expend=expend),
        estimation=dict(gravity=grav),
        diagnostics=diagnostics,
        validation=checks,
    )
    return rr
