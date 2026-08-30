"""
viz/partition_figs.py — figures for the historical-partition analysis.

House style throughout (viz.style): Palatino serif, LaTeX/mathtext, transparent
background, tight bbox. **Every legend sits BELOW the axes, centred and
horizontal** — never inside the plotting area.

All functions take plain arrays / DataFrames (not RunResult), so the same code
serves the paper's render driver and any ad-hoc analysis. Colours reuse the
house accent palette; the three partitions get fixed, colour-blind-safe hues.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from . import style

PART_ORDER = ["P", "R", "A"]
PART_LABEL = {"P": "Prussian", "R": "Russian", "A": "Austrian"}
# fixed, colour-blind-safe; harmonised with the house accents (#1f5c99 / #c1440e)
PART_COLOR = {"P": "#1f5c99", "R": "#c1440e", "A": "#4a8c6f"}
OBJ_LABEL = {"A_n": r"$\log A_n$", "b_n": r"$\log b_n$",
             "CMA": r"$\log \mathrm{CMA}_n$", "real_v": r"$\log v_n/\mathrm{CPI}$"}
SPEC_LABEL = {"raw": "raw", "ttw": "+ dist. Warsaw",
              "woj_ttw": "+ woj. FE + dist. Warsaw"}


def _below(ax, ncol, title=None, y=-0.16):
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, y), ncol=ncol,
              frameon=False, title=title, handletextpad=0.5, columnspacing=1.5)


# --------------------------------------------------------------------------- #
# 1. Partition-gap dot–whisker (the Topic-11 "mean difference" analogue)
# --------------------------------------------------------------------------- #
def gap_dotwhisker(gaps_year, objects=("A_n", "CMA", "real_v"), *,
                   title="", outpath=None, dpi=300, transparent=True):
    """Horizontal dot–whisker of the partition gap (vs Russian base = 0) in each
    recovered object, with 95% CIs. `gaps_year` = gaps[year][spec]."""
    objects = list(objects)
    fig, ax = plt.subplots(figsize=style.a4_figsize("wide"))
    ypos = np.arange(len(objects))[::-1]
    off = 0.16
    for grp, dy in (("P", +off), ("A", -off)):
        xs, xe, ys = [], [], []
        for j, key in enumerate(objects):
            b, se = gaps_year[key][grp]
            xs.append(b); xe.append(1.96 * se); ys.append(ypos[j] + dy)
        ax.errorbar(xs, ys, xerr=xe, fmt="o", ms=5, capsize=2.5, lw=1.2,
                    color=PART_COLOR[grp], label=f"{PART_LABEL[grp]} $-$ Russian")
    ax.axvline(0, color="#888888", lw=0.9, ls="--")
    ax.set_yticks(ypos)
    ax.set_yticklabels([OBJ_LABEL[k] for k in objects])
    ax.set_xlabel("log gap vs Russian partition (base)")
    ax.set_title(title)
    ax.margins(y=0.25)
    _below(ax, 2, y=-0.18)
    if outpath is not None:
        return style.save(fig, outpath, dpi=dpi, transparent=transparent)
    return fig


# --------------------------------------------------------------------------- #
# 2. The Warsaw flip — one object, gap vs base across control specifications
# --------------------------------------------------------------------------- #
def warsaw_flip(gaps_all, year, objects=("CMA", "real_v"),
                specs=("raw", "ttw", "woj_ttw"), *, grp="P",
                title="", outpath=None, dpi=300, transparent=True):
    """Gap (Prussian $-$ Russian by default) in each object as controls are added,
    with 95% CIs — shows the raw market-access deficit flipping to a residual
    surplus once distance to Warsaw is netted out."""
    yr = str(year)
    x = np.arange(len(specs))
    fig, ax = plt.subplots(figsize=style.a4_figsize("wide"))
    colors = {"CMA": "#1f5c99", "real_v": "#c1440e", "A_n": "#4a8c6f",
              "b_n": "#8a6d3b"}
    for obj in objects:
        b = [gaps_all[yr][s][obj][grp][0] for s in specs]
        se = [1.96 * gaps_all[yr][s][obj][grp][1] for s in specs]
        ax.errorbar(x, b, yerr=se, fmt="o-", ms=5, capsize=2.5, lw=1.3,
                    color=colors.get(obj, "#444"), label=OBJ_LABEL[obj])
    ax.axhline(0, color="#888888", lw=0.9, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels([SPEC_LABEL[s] for s in specs])
    ax.set_ylabel(rf"gap ({PART_LABEL[grp]} $-$ Russian)")
    ax.set_title(title)
    ax.margins(x=0.08)
    _below(ax, len(objects), y=-0.22)
    if outpath is not None:
        return style.save(fig, outpath, dpi=dpi, transparent=transparent)
    return fig


# --------------------------------------------------------------------------- #
# 3. Mechanism: market access vs distance to Warsaw, by partition
# --------------------------------------------------------------------------- #
def cma_vs_warsaw(df_year, *, title="", outpath=None, dpi=300, transparent=True):
    """Scatter of log CMA against log travel time to Warsaw, coloured by
    partition, with a per-partition OLS line. Legend below."""
    fig, ax = plt.subplots(figsize=style.a4_figsize("wide"))
    x = np.log(df_year["tt_warsaw"].values + 1.0)
    y = df_year["logCMA"].values
    part = df_year["partition"].values
    for grp in PART_ORDER:
        m = part == grp
        ax.scatter(x[m], y[m], s=5, alpha=0.30, color=PART_COLOR[grp],
                   edgecolors="none", rasterized=True)
    xs = np.array([np.quantile(x, 0.01), np.quantile(x, 0.99)])
    for grp in PART_ORDER:
        m = part == grp
        b, a = np.polyfit(x[m], y[m], 1)
        ax.plot(xs, a + b * xs, color=PART_COLOR[grp], lw=1.7,
                label=rf"{PART_LABEL[grp]} (slope ${b:.2f}$)")
    ax.set_xlabel(r"$\log$ travel time to Warsaw (min)")
    ax.set_ylabel(r"$\log \mathrm{CMA}_n$")
    ax.set_title(title)
    _below(ax, 3, y=-0.20)
    if outpath is not None:
        return style.save(fig, outpath, dpi=dpi, transparent=transparent)
    return fig


# --------------------------------------------------------------------------- #
# 4. Border-imposition welfare cost — grouped bars by year and scenario
# --------------------------------------------------------------------------- #
def border_bars(bw, *, scenarios=None, scen_label=None, title="",
                outpath=None, dpi=300, transparent=True):
    """Grouped bars: welfare cost (%) of re-imposing the partition seam, by year
    and scenario. `bw` = {year: {scenario: welfare_pct}}."""
    years = sorted(bw.keys(), key=lambda s: int(s))
    scenarios = scenarios or list(bw[years[0]].keys())
    scen_label = scen_label or {s: s for s in scenarios}
    palette = ["#4a78a8", "#1f5c99", "#c1440e", "#4a8c6f"]
    fig, ax = plt.subplots(figsize=style.a4_figsize("wide"))
    x = np.arange(len(years)); w = 0.8 / len(scenarios)
    for i, s in enumerate(scenarios):
        vals = [bw[y][s] for y in years]
        ax.bar(x + (i - (len(scenarios) - 1) / 2) * w, vals, w,
               color=palette[i % len(palette)], label=scen_label[s])
    ax.axhline(0, color="#333333", lw=0.7)
    ax.set_xticks(x); ax.set_xticklabels(years)
    ax.set_ylabel(r"welfare change (\%)")
    ax.set_title(title)
    _below(ax, len(scenarios), y=-0.16)
    if outpath is not None:
        return style.save(fig, outpath, dpi=dpi, transparent=transparent)
    return fig


# --------------------------------------------------------------------------- #
# 5. Conditional partition gap over the network vintages
# --------------------------------------------------------------------------- #
def gap_trajectory(gaps_all, obj="CMA", spec="woj_ttw", *,
                   caveat_between=(2011, 2021), title="", outpath=None,
                   dpi=300, transparent=True):
    """Partition gap vs Russian (Prussian and Austrian) across 2011/2021/2026 for
    one object and control spec. Marks the Garmin→OSM network-source break."""
    years = sorted(int(y) for y in gaps_all)
    fig, ax = plt.subplots(figsize=style.a4_figsize("wide"))
    for grp in ("P", "A"):
        b = [gaps_all[str(y)][spec][obj][grp][0] for y in years]
        se = [1.96 * gaps_all[str(y)][spec][obj][grp][1] for y in years]
        ax.errorbar(years, b, yerr=se, fmt="o-", ms=5, capsize=2.5, lw=1.3,
                    color=PART_COLOR[grp], label=rf"{PART_LABEL[grp]} $-$ Russian")
    ax.axhline(0, color="#888888", lw=0.9, ls="--")
    if caveat_between:
        xb = np.mean(caveat_between)
        ax.axvline(xb, color="#bbbbbb", lw=0.8, ls=":")
        ax.text(xb, ax.get_ylim()[1], "  Garmin$\\rightarrow$OSM", fontsize=7,
                color="#999999", va="top", ha="left")
    ax.set_xticks(years)
    ax.set_xlabel("network vintage")
    ax.set_ylabel(rf"conditional gap in {OBJ_LABEL[obj]} ({SPEC_LABEL[spec]})")
    ax.set_title(title)
    _below(ax, 2, y=-0.18)
    if outpath is not None:
        return style.save(fig, outpath, dpi=dpi, transparent=transparent)
    return fig


# --------------------------------------------------------------------------- #
# 6. Distribution of an object by partition (box), optionally residualised
# --------------------------------------------------------------------------- #
def partition_box(df_year, col="logCMA", *, resid_on=None, ylabel=None,
                  title="", outpath=None, dpi=300, transparent=True):
    """Box plot of `col` by partition. If `resid_on` (a column name) is given,
    `col` is first residualised on log(resid_on+1) so the plot shows the
    *conditional* distribution (e.g. CMA net of distance to Warsaw)."""
    y = df_year[col].values.astype(float)
    if resid_on is not None:
        z = np.log(df_year[resid_on].values + 1.0)
        X = np.column_stack([np.ones_like(z), z])
        y = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    fig, ax = plt.subplots(figsize=style.a4_figsize("wide"))
    data = [y[df_year["partition"].values == g] for g in PART_ORDER]
    bp = ax.boxplot(data, positions=range(len(PART_ORDER)), widths=0.55,
                    patch_artist=True, showfliers=False,
                    medianprops=dict(color="#222222", lw=1.2),
                    whiskerprops=dict(color="#555555", lw=0.8),
                    capprops=dict(color="#555555", lw=0.8))
    for patch, g in zip(bp["boxes"], PART_ORDER):
        patch.set_facecolor(PART_COLOR[g]); patch.set_alpha(0.55)
        patch.set_edgecolor("#444444"); patch.set_linewidth(0.7)
    ax.set_xticks(range(len(PART_ORDER)))
    ax.set_xticklabels([PART_LABEL[g] for g in PART_ORDER])
    ax.set_ylabel(ylabel or (col + (r" (resid.)" if resid_on else "")))
    ax.set_title(title)
    if outpath is not None:
        return style.save(fig, outpath, dpi=dpi, transparent=transparent)
    return fig
