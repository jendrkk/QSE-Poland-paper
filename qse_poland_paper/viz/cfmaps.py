"""
viz/cfmaps.py — maps, figures and tables for the road-network GE counterfactual.

One solved exact-hat experiment = base-year fundamentals + a target-year travel-time
matrix (counterfac.network_counterfactual). It returns per-gmina hats (w,v,q,p,r,l) and
N x N hats (lam, pi), all ratios cf/obs. This module renders one experiment as the full
network-effect figure set, and combines two/three experiments into the trade-channel and
cross-baseline difference maps.

Design notes
------------
* Every log-hat map uses the symmetric-diverging Jenks scale (maps.choropleth
  diverging=True), so the caller passes log(hat) as values.
* Native-unit reallocation maps rescale a hat by the observed margin:
  Delta X_n = X_n^obs (x_hat_n - 1); with sum R_n = sum L_n = N these sum to zero.
* Welfare is a scalar here (mobile workers), so there is deliberately no welfare map.
* The "infrastructure" figure uses a tau-derived network-improvement intensity, so it
  needs no road geometry. If a newly-built-segment layer becomes available, swap the
  x-axis for true distance-to-new-road; nothing else changes.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import maps, figures, tables
from .. import counterfac as cf
from ..io.trade import build_dni


# --------------------------------------------------------------------------- #
# Experiment container + constructor
# --------------------------------------------------------------------------- #
@dataclass
class CFExperiment:
    res: dict         # counter_facts output (hats + welfare)
    base: object      # RunResult providing the FIXED fundamentals
    target: object    # RunResult providing the TARGET network (tau)
    with_trade: bool  # whether the trade-cost channel was active
    tag: str          # e.g. "2021fund_2026net" (+"_trade")

    @property
    def year_base(self) -> int:
        return int(self.base.year)

    @property
    def year_net(self) -> int:
        return int(self.target.year)


def build_dni_from_tau(base_run, tau_new):
    """Bilateral trade cost for a new travel-time matrix, matching the baseline
    construction so that dChange = dni_new / base.inputs['dni'] is exact.

    Delegates to ``io.trade.build_dni`` (geometric-mean symmetrisation
    ``t = sqrt(tau * tau.T)``, min over positive entries, default ``force_diag_one=False``),
    which is what ``solve.py`` uses when building the baseline ``dni``.
    """
    psi = float(base_run.params["psi"])
    return build_dni(np.asarray(tau_new, float), psi)


def make_experiment(base_run, target_run, *, with_trade=False) -> CFExperiment:
    """Solve one network counterfactual: base fundamentals + target network."""
    tau_new = np.asarray(target_run.inputs["tau"], float)
    dni_new = build_dni_from_tau(base_run, tau_new) if with_trade else None
    res = cf.network_counterfactual(base_run, tau_new, dni_new=dni_new)
    tag = f"{int(base_run.year)}fund_{int(target_run.year)}net" + ("_trade" if with_trade else "")
    return CFExperiment(res=res, base=base_run, target=target_run,
                        with_trade=with_trade, tag=tag)


# --------------------------------------------------------------------------- #
# Pure objects derived from a run / experiment
# --------------------------------------------------------------------------- #
def pe_cma(tau, w, phi, epsi):
    """Partial-equilibrium commuter market access CMA_n = sum_i tau_ni**(-phi) w_i**eps,
    with wages held at their baseline levels (pure access, no GE response)."""
    tau = np.asarray(tau, float)
    w = np.asarray(w, float)
    return (tau ** (-phi)) @ (w ** epsi)


def own_share(uncond):
    """Own-residence commuting share per residence gmina from an unconditional
    commuting matrix [res, work]."""
    u = np.asarray(uncond, float)
    return np.diag(u) / u.sum(axis=1)


def network_improvement_intensity(exp: CFExperiment):
    """Per-gmina network-improvement intensity from the tau change: the
    commuting-probability-weighted reduction in log travel time from residence n,
    sum_i w_{ni} (log tau^base_{ni} - log tau^new_{ni}), w = conditional commuting
    probabilities from the baseline. Self-contained (no road geometry)."""
    tau_b = np.asarray(exp.base.inputs["tau"], float)
    tau_n = np.asarray(exp.target.inputs["tau"], float)
    lam = np.asarray(exp.base.inputs["uncondCom"], float)
    wcond = lam / lam.sum(axis=1, keepdims=True)
    return np.sum(wcond * (np.log(tau_b) - np.log(tau_n)), axis=1)


# --------------------------------------------------------------------------- #
# Per-experiment map set (6 hats + reallocation + derived scalars) + table
# --------------------------------------------------------------------------- #
_HAT_MAPS = [
    ("l", r"$\log\hat l_n$ (employment reallocation)", "map_l_hat"),
    ("r", r"$\log\hat r_n$ (residence reallocation)", "map_r_hat"),
    ("v", r"$\log\hat v_n$ (residential income access)", "map_v_hat"),
    ("q", r"$\log\hat q_n$ (floor-price capitalisation)", "map_q_hat"),
    ("w", r"$\log\hat w_n$ (wage response)", "map_w_hat"),
    ("p", r"$\log\hat p_n$ (goods price index)", "map_p_hat"),
]


def _summary_table(exp: CFExperiment, outpath):
    res = exp.res
    lr = np.log(np.asarray(res["r"], float))
    ll = np.log(np.asarray(res["l"], float))
    rows = [
        (r"aggregate welfare change (\%)", tables._fmt((res["welf"] - 1) * 100)),
        (r"s.d.\ $\log\hat r_n$ (residence)", tables._fmt(float(np.std(lr)), 4)),
        (r"s.d.\ $\log\hat l_n$ (employment)", tables._fmt(float(np.std(ll)), 4)),
        (r"trade channel active", str(exp.with_trade)),
        (r"converged", str(res["converged"])),
        (r"iterations", tables._fmt(res["iters"])),
    ]
    cap = (f"Road-network GE, {exp.year_base} fundamentals $\\to$ {exp.year_net} network"
           + (" (commuting$+$trade)" if exp.with_trade else " (commuting only)"))
    return tables._write(outpath, tables._table(
        cap, f"tab:cf_{exp.tag}", ["Quantity", "Value"], rows))


def plot_experiment_maps(exp: CFExperiment, gpkg, figdir, tabdir, *, dpi,
                         transparent, fmt, seams=None, made=None):
    """Full per-experiment set: six direct hat maps, native-unit reallocation
    (persons/jobs), own-commute-share change, job-pull vs residence-pull, PE access
    gain, GE-minus-PE gap, and the summary table."""
    made = made if made is not None else []
    figdir = Path(figdir); tabdir = Path(tabdir)
    sub = figdir / f"cf__{exp.tag}"
    sub.mkdir(parents=True, exist_ok=True)
    codes = exp.base.codes
    res = exp.res
    ttl = f"{exp.year_base} fund. $\\to$ {exp.year_net} network"

    if gpkg is not None:
        # (1-6) direct per-gmina hat maps
        for key, label, fname in _HAT_MAPS:
            try:
                made.append(maps.choropleth(
                    gpkg, codes, np.log(np.asarray(res[key], float)),
                    diverging=True, title=f"{label}, {ttl}", label=label,
                    dpi=dpi, transparent=transparent, fmt=fmt, seams=seams,
                    outpath=sub / fname))
            except Exception as e:
                print(f"  [{exp.tag} {fname}] skipped: {e}")

        # native-unit reallocation (sum to zero)
        for key, marg, label, fname in [
                ("r", "R_n", r"$\Delta R_n$ (residents, persons)", "map_R_delta_persons"),
                ("l", "L_n", r"$\Delta L_n$ (jobs, persons)", "map_L_delta_jobs")]:
            try:
                base_margin = np.asarray(exp.base.inputs[marg], float)
                delta = base_margin * (np.asarray(res[key], float) - 1.0)
                made.append(maps.choropleth(
                    gpkg, codes, delta, diverging=True, title=f"{label}, {ttl}",
                    label=label, dpi=dpi, transparent=transparent, fmt=fmt,
                    seams=seams, outpath=sub / fname))
            except Exception as e:
                print(f"  [{exp.tag} {fname}] skipped: {e}")

        # (7) own-commute-share change: new uncond = uncond_obs * lam_hat
        try:
            u_obs = np.asarray(exp.base.inputs["uncondCom"], float)
            u_cf = u_obs * np.asarray(res["lam"], float)
            d_own = own_share(u_cf) - own_share(u_obs)
            made.append(maps.choropleth(
                gpkg, codes, d_own, diverging=True,
                title=f"$\\Delta$ own-commute share, {ttl}",
                label=r"$\Delta$ own-commute share", dpi=dpi,
                transparent=transparent, fmt=fmt, seams=seams,
                outpath=sub / "map_own_share_delta"))
        except Exception as e:
            print(f"  [{exp.tag} own_share] skipped: {e}")

        # (8) job-pull vs residence-pull
        try:
            jp = np.log(np.asarray(res["l"], float)) - np.log(np.asarray(res["r"], float))
            made.append(maps.choropleth(
                gpkg, codes, jp, diverging=True,
                title=f"Job-pull vs residence-pull, {ttl}",
                label=r"$\log(\hat l_n/\hat r_n)$", dpi=dpi,
                transparent=transparent, fmt=fmt, seams=seams,
                outpath=sub / "map_jobpull_respull"))
        except Exception as e:
            print(f"  [{exp.tag} jobpull] skipped: {e}")

        # (9) PE access gain and GE-minus-PE gap
        try:
            phi = float(exp.base.params["phi"]); epsi = float(exp.base.params["epsi"])
            w0 = np.asarray(exp.base.calibrated["w_n"], float)
            tau_b = np.asarray(exp.base.inputs["tau"], float)
            tau_n = np.asarray(exp.target.inputs["tau"], float)
            dlog_cma = (np.log(pe_cma(tau_n, w0, phi, epsi))
                        - np.log(pe_cma(tau_b, w0, phi, epsi)))
            made.append(maps.choropleth(
                gpkg, codes, dlog_cma, diverging=True,
                title=f"PE access gain, {ttl}",
                label=r"$\Delta\log$ CMA$^{PE}_n$", dpi=dpi,
                transparent=transparent, fmt=fmt, seams=seams,
                outpath=sub / "map_cma_pe"))
            gap = np.log(np.asarray(res["v"], float)) - dlog_cma
            made.append(maps.choropleth(
                gpkg, codes, gap, diverging=True,
                title=f"GE$-$PE gap, {ttl}",
                label=r"$\log\hat v_n-\Delta\log$CMA$^{PE}_n$", dpi=dpi,
                transparent=transparent, fmt=fmt, seams=seams,
                outpath=sub / "map_ge_minus_pe_gap"))
        except Exception as e:
            print(f"  [{exp.tag} cma_pe] skipped: {e}")

    # summary table (always)
    try:
        made.append(_summary_table(exp, tabdir / f"cf__{exp.tag}__summary.tex"))
    except Exception as e:
        print(f"  [{exp.tag} summary] skipped: {e}")
    return made


# --------------------------------------------------------------------------- #
# Trade-channel difference maps: (commuting+trade) minus (commuting-only)
# --------------------------------------------------------------------------- #
_TRADE_DIFF = [
    ("v", r"$\Delta\log\hat v_n$ (trade channel)", "map_v_tradediff"),
    ("w", r"$\Delta\log\hat w_n$ (trade channel)", "map_w_tradediff"),
    ("p", r"$\Delta\log\hat p_n$ (trade channel)", "map_p_tradediff"),
    ("r", r"$\Delta\log\hat r_n$ (trade channel)", "map_r_tradediff"),
]


def plot_trade_channel(exp_commute: CFExperiment, exp_trade: CFExperiment, gpkg,
                       figdir, *, dpi, transparent, fmt, seams=None, made=None):
    """Spatial goods-market-access contribution of the SAME network change:
    log(hat_with_trade) - log(hat_commuting_only), per object."""
    made = made if made is not None else []
    if gpkg is None:
        return made
    sub = Path(figdir) / f"trade_channel__{exp_commute.tag}"
    sub.mkdir(parents=True, exist_ok=True)
    codes = exp_commute.base.codes
    ttl = f"{exp_commute.year_base} fund. $\\to$ {exp_commute.year_net} network"
    for key, label, fname in _TRADE_DIFF:
        try:
            d = (np.log(np.asarray(exp_trade.res[key], float))
                 - np.log(np.asarray(exp_commute.res[key], float)))
            made.append(maps.choropleth(
                gpkg, codes, d, diverging=True, title=f"{label}, {ttl}",
                label=label, dpi=dpi, transparent=transparent, fmt=fmt,
                seams=seams, outpath=sub / fname))
        except Exception as e:
            print(f"  [trade_channel {fname}] skipped: {e}")
    return made


# --------------------------------------------------------------------------- #
# Cross-baseline difference map (3-run experiments)
# --------------------------------------------------------------------------- #
def plot_cross_experiment(exp_a: CFExperiment, exp_b: CFExperiment, gpkg, figdir,
                          *, key, title, label, fname, dpi, transparent, fmt,
                          seams=None, made=None):
    """log hat[key] from exp_a minus log hat[key] from exp_b, mapped."""
    made = made if made is not None else []
    if gpkg is None:
        return made
    figdir = Path(figdir); figdir.mkdir(parents=True, exist_ok=True)
    codes = exp_a.base.codes
    try:
        d = (np.log(np.asarray(exp_a.res[key], float))
             - np.log(np.asarray(exp_b.res[key], float)))
        made.append(maps.choropleth(
            gpkg, codes, d, diverging=True, title=title, label=label, dpi=dpi,
            transparent=transparent, fmt=fmt, seams=seams,
            outpath=figdir / fname))
    except Exception as e:
        print(f"  [xcf {fname}] skipped: {e}")
    return made


# --------------------------------------------------------------------------- #
# Infrastructure-response figure (binned, tau-derived intensity)
# --------------------------------------------------------------------------- #
def plot_infra_response(exp: CFExperiment, figdir, *, dpi, transparent, made=None):
    made = made if made is not None else []
    sub = Path(figdir) / f"cf__{exp.tag}"
    sub.mkdir(parents=True, exist_ok=True)
    intensity = network_improvement_intensity(exp)
    xlab = (r"network-improvement intensity "
            r"$\sum_i \lambda^{obs}_{ni}\,\Delta\log\tau^{-1}_{ni}$")
    for key, ylabel, fname in [
            ("v", r"$\hat v_n$ (residential income access)", "infra_response__v_hat"),
            ("r", r"$\hat r_n$ (residence reallocation)", "infra_response__r_hat")]:
        try:
            made.append(figures.binned_response(
                intensity, np.asarray(exp.res[key], float),
                xlabel=xlab, ylabel=ylabel, logy=True,
                weights=np.asarray(exp.base.inputs["R_n"], float),
                title=(f"GE response vs network improvement, "
                       f"{exp.year_base} fund. $\\to$ {exp.year_net} net."),
                outpath=sub / fname, dpi=dpi, transparent=transparent))
        except Exception as e:
            print(f"  [{exp.tag} {fname}] skipped: {e}")
    return made


# --------------------------------------------------------------------------- #
# Partition winners/losers (table + grouped bar)
# --------------------------------------------------------------------------- #
def partition_winners_losers(exp: CFExperiment, gpkg_partitions, figdir, tabdir,
                             *, dpi, transparent, made=None):
    """Mean log v_hat, log r_hat, PE access gain and GE-PE gap by historical
    partition (P/R/A), pop-weighted by residence employment R_n. Writes a LaTeX
    table and a grouped-bar figure. Needs the partitions GeoPackage."""
    made = made if made is not None else []
    from .. import partitions as pt
    codes = exp.base.codes
    try:
        part = pt.load_partition(gpkg_partitions, codes)
    except Exception as e:
        print(f"  [partition winners/losers] skipped (no partition gpkg): {e}")
        return made

    w = np.asarray(exp.base.inputs["R_n"], float)
    lv = np.log(np.asarray(exp.res["v"], float))
    lr = np.log(np.asarray(exp.res["r"], float))
    phi = float(exp.base.params["phi"]); epsi = float(exp.base.params["epsi"])
    w0 = np.asarray(exp.base.calibrated["w_n"], float)
    dlog_cma = (np.log(pe_cma(np.asarray(exp.target.inputs["tau"], float), w0, phi, epsi))
                - np.log(pe_cma(np.asarray(exp.base.inputs["tau"], float), w0, phi, epsi)))
    gap = lv - dlog_cma
    order = ["P", "R", "A"]
    names = {"P": "Prussian", "R": "Russian", "A": "Austrian"}

    def pmean(y, p):
        m = (part == p)
        return float(np.average(y[m], weights=w[m])) if m.any() else float("nan")

    rows = []
    for p in order:
        rows.append((names[p],
                     tables._fmt(pmean(lv, p) * 100),
                     tables._fmt(pmean(lr, p) * 100),
                     tables._fmt(pmean(dlog_cma, p) * 100),
                     tables._fmt(pmean(gap, p) * 100)))
    tab = tables._table(
        (f"Network winners/losers by partition, {exp.year_base} fund.\\ "
         f"$\\to$ {exp.year_net} network (\\%, pop-weighted)"),
        f"tab:partition_cf_{exp.tag}",
        ["Partition", r"$\log\hat v$", r"$\log\hat r$",
         r"$\Delta\log$CMA$^{PE}$", r"GE$-$PE gap"], rows, colspec="lrrrr")
    tab_path = Path(tabdir) / f"partition_cf__{exp.tag}.tex"
    made.append(tables._write(tab_path, tab))

    try:
        v_series = [pmean(lv, p) * 100 for p in order]
        r_series = [pmean(lr, p) * 100 for p in order]
        g_series = [pmean(gap, p) * 100 for p in order]
        made.append(figures.grouped_bar(
            [names[p] for p in order], [v_series, r_series, g_series],
            ylabel=r"mean change (\%)",
            series_labels=[r"$\log\hat v$", r"$\log\hat r$", r"GE$-$PE gap"],
            title=(f"Partition incidence, {exp.year_base} fund. "
                   f"$\\to$ {exp.year_net} network"),
            outpath=Path(figdir) / f"cf__{exp.tag}" / "partition_incidence",
            dpi=dpi, transparent=transparent))
    except Exception as e:
        print(f"  [partition bar] skipped: {e}")
    return made


# --------------------------------------------------------------------------- #
# Top-level dispatcher
# --------------------------------------------------------------------------- #
def _seams(gpkg, gpkg_partitions, codes):
    if gpkg is None or gpkg_partitions is None:
        return None
    try:
        from .. import partitions as pt
        part = pt.load_partition(gpkg_partitions, codes)
        return maps.dissolve_boundaries(gpkg, codes, part)
    except Exception as e:
        print(f"  [partition seams] skipped: {e}")
        return None


def run(runs_sorted, gpkg, outdir, *, gpkg_partitions=None, dpi=300,
        transparent=True, fmt="png"):
    """Dispatch the full network-effect figure set on 2 or 3 runs (sorted ascending
    by year). 1 run -> nothing (a counterfactual needs two networks). Returns the
    list of written artefact paths."""
    made = []
    figdir = Path(outdir) / "figures"
    tabdir = Path(outdir) / "tables"
    figdir.mkdir(parents=True, exist_ok=True)
    tabdir.mkdir(parents=True, exist_ok=True)
    if len(runs_sorted) < 2:
        return made
    seams = _seams(gpkg, gpkg_partitions, runs_sorted[0].codes)
    mk = dict(dpi=dpi, transparent=transparent, fmt=fmt)

    if len(runs_sorted) == 2:
        a, b = runs_sorted
        exp = make_experiment(a, b)
        plot_experiment_maps(exp, gpkg, figdir, tabdir, seams=seams, made=made, **mk)
        plot_infra_response(exp, figdir, dpi=dpi, transparent=transparent, made=made)
        exp_tr = make_experiment(a, b, with_trade=True)
        plot_trade_channel(exp, exp_tr, gpkg, figdir, seams=seams, made=made, **mk)
        if gpkg_partitions is not None:
            partition_winners_losers(exp, gpkg_partitions, figdir, tabdir,
                                     dpi=dpi, transparent=transparent, made=made)
        return made

    # >= 3 runs: use first, second, last as (a, b, c)
    a, b, c = runs_sorted[0], runs_sorted[1], runs_sorted[-1]
    exp_ab = make_experiment(a, b)
    exp_ac = make_experiment(a, c)
    exp_bc = make_experiment(b, c)
    for exp in (exp_ab, exp_ac, exp_bc):
        plot_experiment_maps(exp, gpkg, figdir, tabdir, seams=seams, made=made, **mk)
        plot_infra_response(exp, figdir, dpi=dpi, transparent=transparent, made=made)

    # incidence by starting structure: same target network c, fundamentals a vs b
    plot_cross_experiment(
        exp_ac, exp_bc, gpkg, figdir, key="r",
        title=(f"Incidence by starting structure "
               f"($\\log\\hat r$: {a.year} vs {b.year} fund., {c.year} network)"),
        label=r"$\Delta\log\hat r_n$", fname="xcf__incidence_by_structure",
        seams=seams, made=made, **mk)
    # marginal network increment at fixed fundamentals: base a, networks c vs b
    plot_cross_experiment(
        exp_ac, exp_ab, gpkg, figdir, key="r",
        title=(f"Marginal network increment at {a.year} fund. "
               f"($\\log\\hat r$: {c.year} vs {b.year} network)"),
        label=r"$\Delta\log\hat r_n$", fname="xcf__marginal_increment",
        seams=seams, made=made, **mk)

    # trade channel + partition tables on the primary (clean) experiment b->c
    exp_bc_tr = make_experiment(b, c, with_trade=True)
    plot_trade_channel(exp_bc, exp_bc_tr, gpkg, figdir, seams=seams, made=made, **mk)
    if gpkg_partitions is not None:
        for exp in (exp_ab, exp_ac, exp_bc):
            partition_winners_losers(exp, gpkg_partitions, figdir, tabdir,
                                     dpi=dpi, transparent=transparent, made=made)
    return made
