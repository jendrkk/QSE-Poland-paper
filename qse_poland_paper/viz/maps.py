"""
viz/maps.py — gmina choropleths on the 2021 commune GeoPackage.

Level maps (log A_n, log b_n, ...) use Jenks natural breaks; change maps
(Δlog between two runs) use a symmetric diverging scale centred at zero. A
voivodeship boundary overlay (dissolved from the gmina layer) provides context.
Transparent background, tight bounding box, high DPI — all inherited from
viz.style.save.

Legends are placed BELOW the map, centred and horizontal (never on the map), so
they never occlude Poland. An optional `seams` overlay draws dissolved
partition (or any grouping) boundaries on top for the partition figures.
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


def _clean_boundary(gdf, group_vals, simplify_m):
    """Dissolve `gdf` by `group_vals` on the geometry AS GIVEN (repairing invalid
    polygons first) and return the group boundary LINES, simplified as whole lines.

    Dissolving BEFORE any per-polygon simplification is the whole point: neighbours
    in the same group then share exact edges, so the union has no interior slivers
    to trace. Simplifying the fills first would leave a gap — hence a stray interior
    fragment — at nearly every shared commune edge. Rows with a null group are
    dropped."""
    g = gdf.copy()
    grp = np.asarray(group_vals, dtype=object)
    keep = np.array([v is not None and v == v for v in grp])   # drop None / NaN
    g = g.loc[keep].copy()
    g["_grp"] = grp[keep]
    try:
        from shapely.validation import make_valid
        g["geometry"] = g.geometry.apply(lambda ge: ge if ge.is_valid else make_valid(ge))
    except Exception:
        g["geometry"] = g.geometry.buffer(0)
    bnd = g.dissolve(by="_grp").boundary
    if simplify_m:
        try:
            bnd = bnd.simplify(simplify_m, preserve_topology=True)
        except Exception:
            pass
    return bnd


def load_communes(gpkg_path, layer="communes", simplify_m=250.0):
    """Load and cache the commune layer, keyed by 7-digit teryt (`_teryt`), with a
    CLEAN dissolved voivodeship boundary overlay (`_STATES`, a GeoSeries of lines).

    The commune FILL geometry is simplified per polygon (default 250 m) for fast
    national rendering, but the voivodeship overlay is dissolved from the ORIGINAL
    geometry first and only then simplified as lines — otherwise the per-polygon
    simplification leaves a gap at every shared commune edge and the woj boundary
    fragments into stray interior lines inside each voivodeship."""
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
    # clean voivodeship overlay from the UNSIMPLIFIED geometry (before the fills
    # are simplified below)
    try:
        states = _clean_boundary(g, g["_woj"].values, simplify_m)
    except Exception:
        states = None
    # per-polygon simplify only the FILL geometry (each commune is outlined, so any
    # sub-pixel gap is hidden; this does not affect the woj/partition overlays)
    if simplify_m:
        try:
            g["geometry"] = g.geometry.simplify(simplify_m, preserve_topology=True)
        except Exception:
            pass
    _GDF, _STATES = g, states
    return _GDF, _STATES


_SEAM_CACHE = {}


def dissolve_boundaries(gpkg_path, codes, groups, *, simplify_m=400.0, layer="communes"):
    """Return a GeoSeries of clean group boundaries (e.g. the P/R/A partition
    seams) for overlaying on any map.

    Reads the ORIGINAL, unsimplified commune geometry (NOT the cached, per-polygon
    simplified layer used for fills): dissolving independently-simplified polygons
    pulls shared edges apart and leaves a gap — hence a boundary fragment — at
    nearly every commune border, which is what fragments a naive seam overlay.
    Here the geometry is repaired (make_valid) and dissolved by group first, so
    shared borders merge exactly, and only the resulting *seam lines* are
    simplified (as whole lines), keeping them crisp. Cached per (path, groups)."""
    import geopandas as gpd
    key = (str(gpkg_path), tuple(np.asarray(groups, dtype=object)), simplify_m)
    if key in _SEAM_CACHE:
        return _SEAM_CACHE[key]
    try:
        g = gpd.read_file(gpkg_path, layer=layer)
    except Exception:
        g = gpd.read_file(gpkg_path)
    tkey = next((k for k in _KEYS if k in g.columns), None)
    g["_teryt"] = g[tkey].map(code7)
    m = dict(zip((code7(c) for c in codes), np.asarray(groups, dtype=object)))
    grp = np.array([m.get(t) for t in g["_teryt"].values], dtype=object)
    bnd = _clean_boundary(g, grp, simplify_m)
    _SEAM_CACHE[key] = bnd
    return bnd


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


# --------------------------------------------------------------------------- #
# Legend-below helpers — the house rule: legend under the map, centred, horizontal
# --------------------------------------------------------------------------- #
def _swatch_legend_kwds(label, ncol):
    """Class-swatch legend, placed below the axes, centred and spread across
    `ncol` columns (horizontal). Never drawn on the map."""
    return {"loc": "upper center", "bbox_to_anchor": (0.5, -0.02), "ncol": ncol,
            "frameon": False, "fontsize": 6, "title": label, "title_fontsize": 7,
            "columnspacing": 1.1, "handletextpad": 0.4, "borderaxespad": 0.0,
            "markerscale": 0.9}


def _cbar_kwds(label):
    """Horizontal colour-bar under the map."""
    return {"orientation": "horizontal", "location": "bottom", "shrink": 0.6,
            "pad": 0.02, "aspect": 32, "label": label}


def choropleth(gpkg_path, codes, values, title="", label="", *,
               diverging=False, k=7, cmap=None, dpi=300, transparent=True,
               figsize=None, fmt=None, outpath=None, diverging_continuous=False,
               legend_ncol=None, seams=None, seam_kwds=None):
    """Render one choropleth. Returns the saved path (if outpath) or the Figure.

    Level maps (diverging=False) use Jenks natural breaks. Change maps
    (diverging=True) use SYMMETRIC Jenks breaks around zero by default so a few
    outliers cannot inflate the colour scale; pass diverging_continuous=True to
    restore the continuous TwoSlopeNorm.

    Legends are ALWAYS below the map, centred and horizontal. `seams` may be a
    GeoSeries of boundaries (e.g. from ``dissolve_boundaries``) to overlay.
    """
    gdf, states = load_communes(gpkg_path)
    g = _frame_values(gdf, codes, values)
    plotted = g[g["_val"].notna()]

    fig, ax = plt.subplots(figsize=figsize or style.a4_figsize("map"))
    gdf.plot(ax=ax, color="#e6e6e6", edgecolor="none", zorder=1)

    if diverging:
        bins = None if diverging_continuous else _sym_jenks_bins(plotted["_val"].values, k)
        if bins is not None:
            nc = legend_ncol or min(len(bins) - 1, 5)
            plotted.plot(ax=ax, column="_val", cmap=cmap or style.CMAP_DIV,
                         scheme="UserDefined", classification_kwds={"bins": bins},
                         linewidth=0.03, edgecolor="#666666", zorder=2, legend=True,
                         legend_kwds=_swatch_legend_kwds(label, nc))
        else:
            vmax = float(np.nanmax(np.abs(plotted["_val"].values)))
            vmax = vmax if vmax > 0 else 1.0
            norm = matplotlib.colors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
            plotted.plot(ax=ax, column="_val", cmap=cmap or style.CMAP_DIV, norm=norm,
                         linewidth=0.03, edgecolor="#666666", zorder=2, legend=True,
                         legend_kwds=_cbar_kwds(label))
    else:
        scheme = "NaturalBreaks"
        try:
            import mapclassify  # noqa: F401
        except Exception:
            scheme = "Quantiles"
        nc = legend_ncol or min(k, 4)
        plotted.plot(ax=ax, column="_val", cmap=cmap or style.CMAP_SEQ,
                     scheme=scheme, k=k, linewidth=0.03, edgecolor="#666666",
                     zorder=2, legend=True,
                     legend_kwds=_swatch_legend_kwds(label, nc))
    if states is not None:
        states.plot(ax=ax, color="#ffffff", linewidth=0.5, zorder=3)
    if seams is not None:
        sk = {"color": "#111111", "linewidth": 0.8, "zorder": 4}
        sk.update(seam_kwds or {})
        seams.plot(ax=ax, **sk)

    ax.set_title(title)
    ax.set_axis_off()
    ax.margins(0)
    if outpath is not None:
        return style.save(fig, outpath, dpi=dpi, transparent=transparent, fmt=fmt)
    return fig


def categorical(gpkg_path, codes, groups, *, colors, order=None, labels=None,
                title="", legend_title="", dpi=300, transparent=True,
                figsize=None, fmt=None, outpath=None):
    """Categorical choropleth (e.g. the P/R/A partition map). One flat colour per
    group; voivodeship overlay; legend below the map, centred, horizontal."""
    import matplotlib.patches as mpatches
    gdf, states = load_communes(gpkg_path)
    m = dict(zip((code7(c) for c in codes), np.asarray(groups, dtype=object)))
    g = gdf.copy()
    g["_grp"] = g["_teryt"].map(m)
    order = order or list(colors)
    fig, ax = plt.subplots(figsize=figsize or style.a4_figsize("map"))
    gdf.plot(ax=ax, color="#e6e6e6", edgecolor="none", zorder=1)
    for grp in order:
        sub = g[g["_grp"] == grp]
        if len(sub):
            sub.plot(ax=ax, color=colors[grp], linewidth=0.03,
                     edgecolor="#ffffff", zorder=2)
    if states is not None:
        states.plot(ax=ax, color="#ffffff", linewidth=0.5, zorder=3)
    labels = labels or {gk: gk for gk in order}
    handles = [mpatches.Patch(facecolor=colors[gk], edgecolor="none",
                              label=labels[gk]) for gk in order]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02),
              ncol=len(order), frameon=False, title=legend_title,
              handletextpad=0.5, columnspacing=1.5)
    ax.set_title(title)
    ax.set_axis_off()
    ax.margins(0)
    if outpath is not None:
        return style.save(fig, outpath, dpi=dpi, transparent=transparent, fmt=fmt)
    return fig
