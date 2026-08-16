"""
viz/tables.py — LaTeX tables (booktabs-style, no external packages required).

Tables draw on both the calibration/estimation stage (gravity) and the model
solution stage (moments of the recovered fundamentals and equilibrium objects).
Each writer returns the path it wrote.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def _fmt(x, nd=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "--"
    if isinstance(x, (int, np.integer)):
        return f"{int(x):,}"
    return f"{x:.{nd}f}"


def _write(path, body):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


def _table(caption, label, header, rows, colspec=None):
    ncol = len(header)
    colspec = colspec or ("l" + "r" * (ncol - 1))
    out = [r"\begin{table}[htbp]", r"\centering",
           rf"\caption{{{caption}}}", rf"\label{{{label}}}",
           rf"\begin{{tabular}}{{{colspec}}}", r"\hline\hline",
           " & ".join(header) + r" \\", r"\hline"]
    for row in rows:
        out.append(" & ".join(str(c) for c in row) + r" \\")
    out += [r"\hline\hline", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(out)


def gravity_table(run, outpath):
    g = run.estimation.get("gravity", {})
    rows = [
        (r"$\varphi=\varepsilon\mu$ (commuting decay)", _fmt(g.get("phi"))),
        (r"std.\ error", _fmt(g.get("se"))),
        (r"$R^2$ (within)", _fmt(g.get("r2"))),
        (r"positive off-diagonal pairs", _fmt(g.get("n_pairs"))),
        (r"source", str(g.get("source", "--")).replace("_", r"\_")),
    ]
    return _write(outpath, _table(
        rf"Commuting gravity, {run.year}", f"tab:gravity_{run.year}",
        ["Quantity", "Value"], rows))


def calibration_table(run, outpath):
    p = run.params
    cal = run.calibrated
    def stat(name, arr, log=True):
        a = np.asarray(arr, float); a = a[np.isfinite(a)]
        la = np.log(a[a > 0]) if log else a
        return (name, _fmt(np.mean(la)), _fmt(np.std(la)),
                _fmt(np.min(la)), _fmt(np.max(la)))
    rows = [stat(r"$\log A_n$ (productivity)", cal["A_n"]),
            stat(r"$\log b_n$ (amenity)", cal["b_n"]),
            stat(r"$\log P_n$ (price index)", cal["P_n"]),
            stat(r"$\log \mathrm{CMA}_n$", cal["CMA"]),
            stat(r"$\log w_n$ (wage)", cal["w_n"]),
            stat(r"$\log v_n$ (res.\ income)", cal["v_n"]),
            stat(r"$\log Q_n$ (floor price)", run.inputs["Q_n"])]
    tbl = _table(
        rf"Recovered fundamentals and equilibrium moments, {run.year}",
        f"tab:calib_{run.year}",
        ["Object", "mean", "s.d.", "min", "max"], rows)
    # parameter block appended
    par = ("\n% parameters: "
           f"alpha={_fmt(p['alpha'])}, epsi={_fmt(p['epsi'])}, mu={_fmt(p['mu'])}, "
           f"phi={_fmt(p['phi'])}, sigma={_fmt(p['sigg'])}, nu={_fmt(p['nu'])}, "
           f"delta={_fmt(p['delta'])}, psi={_fmt(p['psi'])}\n")
    return _write(outpath, tbl + par)


def moments_table(run, outpath):
    d = run.diagnostics
    rows = [
        ("own-commute share", _fmt(d.get("own_commute_share"))),
        ("workplace-margin rel.\ err (pre-IPF)", _fmt(d.get("workplace_margin_rel_err"))),
        ("diagonal clipped units", _fmt(d.get("diagonal_clipped_units"))),
        ("productivity solver iterations", _fmt(d.get("prod_iters"))),
        (r"income$=$expenditure gap", _fmt(d.get("prod_gap"), 8)),
    ]
    return _write(outpath, _table(
        rf"Model diagnostics, {run.year}", f"tab:diag_{run.year}",
        ["Diagnostic", "Value"], rows))


def comparison_table(runs, outpath):
    """One column per run: key parameters and fundamental moments side by side."""
    header = ["Quantity"] + [str(r.year) for r in runs]

    def rowfun(name, fn):
        return [name] + [_fmt(fn(r)) for r in runs]

    def logmean(arr):
        a = np.asarray(arr, float); a = a[np.isfinite(a) & (a > 0)]
        return float(np.mean(np.log(a)))

    def logsd(arr):
        a = np.asarray(arr, float); a = a[np.isfinite(a) & (a > 0)]
        return float(np.std(np.log(a)))

    rows = [
        rowfun(r"$\varphi$", lambda r: r.params["phi"]),
        rowfun(r"$N$", lambda r: r.frame["N"]),
        rowfun(r"own-commute share", lambda r: r.diagnostics.get("own_commute_share")),
        rowfun(r"s.d.\ $\log A_n$", lambda r: logsd(r.calibrated["A_n"])),
        rowfun(r"s.d.\ $\log b_n$", lambda r: logsd(r.calibrated["b_n"])),
        rowfun(r"s.d.\ $\log \mathrm{CMA}_n$", lambda r: logsd(r.calibrated["CMA"])),
        rowfun(r"mean $\log v_n/\mathrm{CPI}$", lambda r: logmean(r.calibrated["real_v"])),
    ]
    return _write(outpath, _table(
        "Cross-run comparison", "tab:compare",
        header, rows, colspec="l" + "r" * len(runs)))
