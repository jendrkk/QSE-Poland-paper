"""
viz/figures.py — non-map figures.

All figures return the saved path (when outpath is given) and use the house
style/sizing. Nothing here assumes a specific run year, so the same functions
serve single-run and comparison contexts.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from . import style


def gravity_fit(comMat, tau, phi, *, title="Commuting gravity", outpath=None,
                dpi=300, transparent=True, nbin=40, residualized=True):
    """Commuting-gravity scatter with the estimated decay slope.

    With residualized=True (default) the axes are the origin- and destination-FE
    residualized log flow and log travel time — i.e. the exact variation the
    two-way fixed-effects estimator uses — so the fitted slope -phi passes
    cleanly through the binned means. This is the faithful picture of the
    estimate. With residualized=False it shows the raw log-flow/log-time cloud.
    """
    n = comMat.shape[0]
    mask = (comMat > 0)
    np.fill_diagonal(mask, False)
    r, c = np.where(mask)
    x = np.log(tau[r, c])
    y = np.log(comMat[r, c])

    if residualized:
        from ..estimate import _absorb_two_way
        y, x = _absorb_two_way(r, c, y, x, n)
        xlab = r"$\log \tau_{ni}$ (origin/dest.\ FE residual)"
        ylab = r"$\log \Lambda_{ni}$ (origin/dest.\ FE residual)"
    else:
        xlab = r"$\log \tau_{ni}$ (travel time, min)"
        ylab = r"$\log \Lambda_{ni}$ (commuters)"

    qs = np.quantile(x, np.linspace(0, 1, nbin + 1))
    qs[-1] += 1e-9
    idx = np.clip(np.digitize(x, qs) - 1, 0, nbin - 1)
    bx = np.array([x[idx == b].mean() for b in range(nbin) if np.any(idx == b)])
    by = np.array([y[idx == b].mean() for b in range(nbin) if np.any(idx == b)])

    fig, ax = plt.subplots(figsize=style.a4_figsize("wide"))
    ax.scatter(x, y, s=2, alpha=0.05, color="#8a8a8a", edgecolors="none", rasterized=True)
    ax.scatter(bx, by, s=18, color="#c1440e", zorder=3, label="bin means")
    xs = np.array([np.quantile(x, 0.001), np.quantile(x, 0.999)])
    b0 = by.mean() + phi * bx.mean()          # residualized means ~ (0,0)
    ax.plot(xs, b0 - phi * xs, color="#1f5c99", lw=1.6,
            label=rf"slope $-\varphi={-phi:.2f}$")
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title(title)
    ax.legend(frameon=False, loc="upper right")
    if outpath is not None:
        return style.save(fig, outpath, dpi=dpi, transparent=transparent)
    return fig


def distribution(values, label, *, title="", logx=True, outpath=None,
                 dpi=300, transparent=True):
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    x = np.log(v[v > 0]) if logx else v
    fig, ax = plt.subplots(figsize=style.a4_figsize("half"))
    ax.hist(x, bins=45, color="#4a78a8", edgecolor="white", linewidth=0.3, density=True)
    ax.axvline(np.median(x), color="#c1440e", lw=1.2, label="median")
    ax.set_xlabel((r"$\log$ " if logx else "") + label)
    ax.set_ylabel("density")
    ax.set_title(title)
    ax.legend(frameon=False)
    if outpath is not None:
        return style.save(fig, outpath, dpi=dpi, transparent=transparent)
    return fig


def lorenz(values, *, label="employment", title="", outpath=None,
           dpi=300, transparent=True):
    v = np.sort(np.asarray(values, float))
    v = v[np.isfinite(v) & (v >= 0)]
    cum = np.cumsum(v) / v.sum()
    p = np.linspace(0, 1, len(cum))
    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    gini = 1 - 2 * _trapz(cum, p)
    fig, ax = plt.subplots(figsize=style.a4_figsize("square"))
    ax.plot(p, cum, color="#1f5c99", lw=1.6, label=rf"{label} (Gini $={gini:.3f}$)")
    ax.plot([0, 1], [0, 1], color="#888888", lw=0.8, ls="--")
    ax.set_xlabel("cumulative share of gminas")
    ax.set_ylabel(f"cumulative share of {label}")
    ax.set_title(title)
    ax.legend(frameon=False, loc="upper left")
    if outpath is not None:
        return style.save(fig, outpath, dpi=dpi, transparent=transparent)
    return fig


def scatter(x, y, xlabel, ylabel, *, title="", logx=True, logy=True,
            outpath=None, dpi=300, transparent=True, s=5):
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if logx:
        ok &= x > 0
    if logy:
        ok &= y > 0
    xx = np.log(x[ok]) if logx else x[ok]
    yy = np.log(y[ok]) if logy else y[ok]
    fig, ax = plt.subplots(figsize=style.a4_figsize("square"))
    ax.scatter(xx, yy, s=s, alpha=0.3, color="#4a4a4a", edgecolors="none", rasterized=True)
    if len(xx) > 2:
        b, a = np.polyfit(xx, yy, 1)
        xs = np.array([xx.min(), xx.max()])
        rho = np.corrcoef(xx, yy)[0, 1]
        ax.plot(xs, a + b * xs, color="#c1440e", lw=1.4,
                label=rf"slope $={b:.2f}$, $\rho={rho:.2f}$")
        ax.legend(frameon=False)
    ax.set_xlabel((r"$\log$ " if logx else "") + xlabel)
    ax.set_ylabel((r"$\log$ " if logy else "") + ylabel)
    ax.set_title(title)
    if outpath is not None:
        return style.save(fig, outpath, dpi=dpi, transparent=transparent)
    return fig


def compare_scatter(x, y, label_x, label_y, *, title="", outpath=None,
                    dpi=300, transparent=True):
    """45-degree scatter of one fundamental across two runs (levels in logs)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    xx, yy = np.log(x[ok]), np.log(y[ok])
    fig, ax = plt.subplots(figsize=style.a4_figsize("square"))
    ax.scatter(xx, yy, s=5, alpha=0.3, color="#4a4a4a", edgecolors="none", rasterized=True)
    lo = min(xx.min(), yy.min()); hi = max(xx.max(), yy.max())
    ax.plot([lo, hi], [lo, hi], color="#888888", lw=0.9, ls="--", label="45$^\\circ$")
    rho = np.corrcoef(xx, yy)[0, 1]
    ax.set_xlabel(label_x); ax.set_ylabel(label_y)
    ax.set_title(title + rf"  ($\rho={rho:.3f}$)")
    ax.legend(frameon=False, loc="upper left")
    if outpath is not None:
        return style.save(fig, outpath, dpi=dpi, transparent=transparent)
    return fig
