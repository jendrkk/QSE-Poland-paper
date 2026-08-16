"""
qse_poland_paper — baseline MRRH (Monte–Redding–Rossi-Hansberg 2018) toolkit for
the Polish quantitative-spatial project.

Back-end ("src") for calibrating and solving the MRRH model on the universe of
~2,477 Polish gminas, for several yearly cross-sections (2011, 2021, 2026), with
NO rail and NO congestion in the baseline. The verified model core is ported from
the Topic-11 reference implementation (professor's MATLAB `*TK.m` and the students'
`mrrh_pipeline`), which were checked line-for-line against SW2020 eqs. 10/12 and
Codebook §A.2.

Public entry points:
    calibrate_year(cfg, year) -> RunResult      (qse_poland_paper.solve)
    counter_facts(...)                          (qse_poland_paper.counterfac)
    RunResult                                   (qse_poland_paper.result)

The package is deliberately organised around named extension seams (rail TTM,
mode choice, congestion, partition-border analysis) documented in the design plan;
none of those are implemented in the baseline, but the interfaces are in place.
"""
from __future__ import annotations

__version__ = "0.1.0"
SCHEMA_VERSION = 1

from . import config, frame, quantify, counterfac, validate  # noqa: E402
from .result import RunResult  # noqa: E402

__all__ = [
    "config",
    "frame",
    "quantify",
    "counterfac",
    "validate",
    "RunResult",
    "__version__",
    "SCHEMA_VERSION",
]
