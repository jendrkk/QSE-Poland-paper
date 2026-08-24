"""
viz/maps.py — gmina choropleths on the 2021 commune GeoPackage.

Level maps (log A_n, log b_n, ...) use Jenks natural breaks; change maps
(Δlog between two runs) use a symmetric diverging scale centred at zero. A
voivodeship boundary overlay (dissolved from the gmina layer) provides context.
Transparent background, tight bounding box, high DPI — all inherited from
viz.style.save.
"""
from __future__ import annotations

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from . import style
from ..frame import code7

_GDF = None
_STATES = None
_KEYS = ("JPT_KOD_JE", "JPT_KJ_I_1", "teryt7", "gmina_teryt", "TERYT")


def load_communes(gpkg_path, layer="communes", simplify_m=250.0):
    """Load and cache the commune layer, keyed by 7-digit teryt (`_teryt`), with a
    dissolved voivodeship overlay. Geometry is simplified (default 250 m in the
    metric CRS) so national maps render in ~2 s each instead of ~15 s — visually
    indistinguishable at page scale."""
    global _GDF, _STATES
    if _GDF is not None:
        return _GDF, _STATES
    import geopandas as gpd
    try:
        g = gpd.read_file(gpkg_path, layer=layer)
    except Exception:
        g = gpd.read_file(gpkg_path)
    key = next((k for k in _KEYS if k in g.columns), None)
    if key is None:
        raise ValueError(f"no teryt key in commune layer; columns={list(g.columns)[:10]}")
    g["_teryt"] = g[key].map(code7)
    g["_woj"] = g["_teryt"].str[:2]
    if simplify_m:
        try:
            g["geometry"] = g.geometry.simplify(simplify_m, preserve_topology=True)
        except Exception:
            pass
    try:
        states = g.dissolve(by="_woj").reset_index()
    except Exception:
        states = None
    _GDF, _STATES = g, states
    return _GDF, _STATES


def _frame_values(gdf, codes, values):
    m = dict(zip((code7(c) for c in codes), np.asarray(values, float)))
    out = gdf.copy()
    out["_val"] = out["_teryt"].map(m)
    return out


def _sym_jenks_bins(values, k):
    """Symmetric Jenks (natural-breaks) class boundaries centred at zero.

    Classify |value| into ~k/2 natural-breaks classes, then mirror the edges
    across zero. Returns the UserDefined upper-edge boundaries (ascending) or
    None if there is too little spread to classify (caller falls back to a
    continuous scale). Outlier-robust: extremes fall into the outer classes
    rather than stretching the whole colour range.
    """
    v = np.asarray(values, float)
    absv = v[np.isfinite(v) & (v != 0)]
    absv = np.abs(absv)
    kk = max(2, k // 2)
    if absv.size <= kk or np.unique(absv).size <= kk:
        return None
    try:
        import mapclassify
        edges = np.asarray(mapclassify.NaturalBreaks(absv, k=kk).bins, float)
    except Exception:
        edges = np.quantile(absv, np.linspace(1.0 / kk, 1.0, kk))
    edges = np.unique(edges[edges > 0])
    if edges.size == 0:
        return None
    edges[-1] = max(edges[-1], absv.max()) * 1.0001
    return list(-edges[::-1]) + [0.0] + list(edges)


def choropleth(gpkg_path, codes, values, title="", label="", *,
               diverging=False, k=7, cmap=None, dpi=300, transparent=True,
               figsize=None, fmt=None, outpath=None, diverging_continuous=False):
    """Render one choropleth. Returns the saved path (if outpath) or the Figure.

    Level maps (diverging=False) use Jenks natural breaks. Change maps
    (diverging=True) use SYMMETRIC Jenks breaks around zero by default so a few
    outliers cannot inflate the colour scale; pass diverging_continuous=True to
    restore the old continuous TwoSlopeNorm.
    """
    gdf, states = load_communes(gpkg_path)
    g = _frame_values(gdf, codes, values)
    plotted = g[g["_val"].notna()]

    fig, ax = plt.subplots(figsize=figsize or style.a4_figsize("map"))
    gdf.plot(ax=ax, color="#e6e6e6", edgecolor="none", zorder=1)

    if diverging:
        bins = None if diverging_continuous else _sym_jenks_bins(plotted["_val"].values, k)
        if bins is not None:
            plotted.plot(ax=ax, column="_val", cmap=cmap or style.CMAP_DIV,
                         scheme="UserDefined", classification_kwds={"bins": bins},
                         linewidth=0.03, edgecolor="#666666", zorder=2, legend=True,
                         legend_kwds={"loc": "lower left", "fontsize": 6,
                                      "frameon": False, "title": label})
        else:
            vmax = float(np.nanmax(np.abs(plotted["_val"].values)))
            vmax = vmax if vmax > 0 else 1.0
            norm = matplotlib.colors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
            plotted.plot(ax=ax, column="_val", cmap=cmap or style.CMAP_DIV, norm=norm,
                         linewidth=0.03, edgecolor="#666666", zorder=2, legend=True,
                         legend_kwds={"shrink": 0.55, "label": label,
                                      "orientation": "vertical"})
    else:
        scheme = "NaturalBreaks"
        try:
            import mapclassify  # noqa: F401
        except Exception:
            scheme = "Quantiles"
        plotted.plot(ax=ax, column="_val", cmap=cmap or style.CMAP_SEQ,
                     scheme=scheme, k=k, linewidth=0.03, edgecolor="#666666",
                     zorder=2, legend=True,
                     legend_kwds={"loc": "lower left", "fontsize": 6,
                                  "frameon": False, "title": label})
    if states is not None:
        states.boundary.plot(ax=ax, color="#ffffff", linewidth=0.5, zorder=3)

    ax.set_title(title)
    ax.set_axis_off()
    ax.margins(0)
    if outpath is not None:
        return style.save(fig, outpath, dpi=dpi, transparent=transparent, fmt=fmt)
    return fig