"""
qse_poland_paper.viz — visualization back-end.

style : LaTeX/Palatino rcParams, A4-optimised figure sizing, transparent/tight/
        high-dpi save helpers.
maps  : gmina choropleths on the 2021 commune GeoPackage (Jenks levels, diverging
        changes), with voivodeship overlay.
figures : non-map figures (gravity fit, distributions, Lorenz, scatters, cross-run
          comparison panels).
tables : LaTeX tables (gravity, calibration summary, moments, cross-run comparison).

The visualization runner (run_viz.py) consumes RunResult pickles and drives these.
"""
from __future__ import annotations

from . import style, maps, figures, tables

__all__ = ["style", "maps", "figures", "tables"]
