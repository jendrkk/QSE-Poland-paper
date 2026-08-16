"""
viz/style.py — plotting style, sizing and save helpers.

House style (matching the project notebooks): serif Palatino typography, LaTeX
rendering, transparent background, tight bounding box, high adjustable DPI, figure
sizes optimised for an A4 page.

LaTeX is a hard requirement of the house style, so usetex is attempted first; if
the local TeX cannot render (missing distribution/packages), the module falls
back to matplotlib's mathtext with a Palatino-like serif and records that it did,
so a run never crashes for want of LaTeX.
"""
from __future__ import annotations

import shutil
import subprocess
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402


# A4 geometry (inches) and a usable text width for a typical LaTeX article
A4_W, A4_H = 8.27, 11.69
TEXT_W = 6.3          # ~ \textwidth at default margins

_STATE = {"usetex": None}

_PALATINO = ["Palatino", "TeX Gyre Pagella", "URW Palladio L", "Palatino Linotype",
             "P052", "DejaVu Serif", "serif"]

# neutral, colour-blind-safe scales (no fixed group colours)
CMAP_SEQ = "viridis"
CMAP_DIV = "RdBu_r"


def _latex_works(preamble: str) -> bool:
    """Actually render a tiny figure WITH usetex enabled and the given preamble;
    return True only if the TeX toolchain produces output without error."""
    if not shutil.which("latex"):
        return False
    saved = {k: plt.rcParams[k] for k in ("text.usetex", "text.latex.preamble")}
    try:
        plt.rcParams["text.usetex"] = True
        plt.rcParams["text.latex.preamble"] = preamble
        fig = plt.figure()
        fig.text(0.5, 0.5, r"$\alpha_{n}$ test")
        tmp = Path("/tmp/_qse_latex_probe.png")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            fig.savefig(tmp, dpi=50)
        plt.close(fig)
        tmp.unlink(missing_ok=True)
        return True
    except Exception:
        plt.close("all")
        return False
    finally:
        plt.rcParams.update(saved)


def use_style(usetex: bool | None = None, base_fontsize: float = 11.0) -> bool:
    """Install the house rcParams. usetex=None -> auto-detect. Returns the
    effective usetex flag (also cached on the module)."""
    avail = {f.name for f in font_manager.fontManager.ttflist}
    serif = [f for f in _PALATINO if f in avail] or ["serif"]

    _preamble = (r"\usepackage[T1]{fontenc}\usepackage{amsmath}\usepackage{amssymb}"
                 r"\usepackage{mathpazo}")
    if usetex is None or usetex:
        # verify the full house preamble renders; if not, retry without mathpazo
        if not _latex_works(_preamble):
            _preamble_basic = (r"\usepackage[T1]{fontenc}\usepackage{amsmath}"
                               r"\usepackage{amssymb}")
            usetex = _latex_works(_preamble_basic)
            if usetex:
                _preamble = _preamble_basic
        else:
            usetex = True

    rc = {
        "font.family": "serif",
        "font.serif": serif,
        "font.size": base_fontsize,
        "axes.titlesize": base_fontsize + 1,
        "axes.labelsize": base_fontsize,
        "legend.fontsize": base_fontsize - 2,
        "xtick.labelsize": base_fontsize - 2,
        "ytick.labelsize": base_fontsize - 2,
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.7,
        "savefig.transparent": True,
        "figure.dpi": 150,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
    }
    if usetex:
        rc["text.usetex"] = True
        rc["text.latex.preamble"] = _preamble
    else:
        rc["text.usetex"] = False
        rc["mathtext.fontset"] = "dejavuserif"
    plt.rcParams.update(rc)
    _STATE["usetex"] = bool(usetex)
    return bool(usetex)


def a4_figsize(kind: str = "wide") -> tuple[float, float]:
    """Figure sizes tuned for an A4 page.
    kind: 'wide' (full text width, 4:3-ish), 'half', 'square', 'map', 'tall'."""
    return {
        "wide": (TEXT_W, TEXT_W * 0.62),
        "half": (TEXT_W / 2, TEXT_W / 2 * 0.85),
        "square": (TEXT_W * 0.8, TEXT_W * 0.8),
        "map": (TEXT_W, TEXT_W * 1.02),        # Poland is ~square
        "map_pair": (TEXT_W, TEXT_W * 0.55),
        "tall": (TEXT_W, TEXT_W * 1.15),
    }.get(kind, (TEXT_W, TEXT_W * 0.62))


def save(fig, path, dpi: int = 300, transparent: bool = True, fmt: str | None = None):
    """Save with tight bbox, transparent background and adjustable DPI."""
    path = Path(path)
    if fmt:
        path = path.with_suffix("." + fmt.lstrip("."))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, transparent=transparent,
                bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return path


def math(s: str) -> str:
    """Wrap a label so it renders whether or not usetex is active."""
    return s
