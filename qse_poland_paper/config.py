"""
config.py — typed configuration objects and the YAML loader.

The single source of truth for a baseline run is a YAML file (see
``config/baseline.yaml``). It is parsed into the dataclasses below, validated,
and copied verbatim into each run folder for reproducibility. No configuration
value is hard-coded elsewhere in the package; every structural parameter is a
documented, override-able knob.

Structural parameters — provenance of the defaults
--------------------------------------------------
alpha  expenditure share on TRADABLES (1-alpha = floor-space/housing share).
       Default 0.70 (German SW2020 value; user's stated lean). Housing share is
       genuinely uncertain for Poland: EU COICOP-04 (housing) ≈ 23.6% of
       consumption (Eurostat, 2024), Poland below the EU average with an
       unusually large utilities component, so the *pure floor-space* share net
       of utilities is not cleanly pinned. Report the alpha sensitivity grid.
epsi   Fréchet shape (commuting/location dispersion). Default 4.60 (SW2020).
       Only phi = epsi*mu is identified from the commuting gradient, so epsi is
       calibrated and mu is the residual. Report an epsi grid.
mu     travel-time elasticity of commuting cost. Residual: mu = phi/epsi when
       phi is estimated in-sample; used directly only if phi_mode='fixed'.
sigg   CES elasticity across varieties (trade elasticity sigma-1). Default 4
       (Broda–Weinstein 2004; SW2020). Not estimable subnationally.
nu     agglomeration elasticity in productivity. Default 0.05
       (Combes–Gobillon–Puga consensus 0.02–0.05).
delta  housing-supply elasticity. Default 0.38 (SW2020). Estimable for Poland
       with a Saiz-type geographic instrument (future work).
psi    trade-cost distance elasticity, used only when dni is built from tau/
       distance (d_ni = (tau/min tau)^psi). Default 0.42 (Head–Mayer).
fixC   price-index normalisation. Fixed at 1.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml


# --------------------------------------------------------------------------- #
# Structural parameters
# --------------------------------------------------------------------------- #
@dataclass
class Params:
    alpha: float = 0.70
    epsi: float = 4.60
    mu: float = 0.47
    sigg: float = 4.00
    nu: float = 0.05
    delta: float = 0.38
    psi: float = 0.42
    fixC: float = 1.00
    # phi = epsi*mu (commuting decay). If phi_mode='estimate', phi comes from the
    # in-sample gravity regression and mu is overwritten to phi/epsi. If 'fixed',
    # phi = epsi*mu from the values above.
    phi_mode: str = "estimate"
    phi_fixed: float | None = None  # only used if phi_mode='fixed' and set explicitly
    # own-commute diagonal imputation (used only if a TTM diagonal must be rebuilt;
    # the delivered TTMs already carry an imputed diagonal, so off by default)
    reimpute_tau_diagonal: bool = False
    own_time_c: float = 2.0 / 3.0
    own_time_speed_kmh: float = 30.0
    own_time_floor_min: float = 3.0
    # documentation-only sensitivity grids (surfaced in manifests / tables)
    sensitivity: dict[str, list[float]] = field(default_factory=lambda: {
        "alpha": [0.70, 0.75, 0.80],
        "epsi": [3.0, 4.0, 4.6, 6.0],
        "sigg": [3.0, 4.0, 6.0],
        "nu": [0.02, 0.05],
        "own_time_speed_kmh": [25.0, 30.0, 40.0],
    })

    @property
    def phi(self) -> float:
        """Commuting decay used in tau**(-phi). Only meaningful for phi_mode='fixed'."""
        if self.phi_mode == "fixed" and self.phi_fixed is not None:
            return float(self.phi_fixed)
        return float(self.epsi * self.mu)


# --------------------------------------------------------------------------- #
# Per-year data specification
# --------------------------------------------------------------------------- #
@dataclass
class YearSpec:
    year: int
    ttm: str                        # path to the .npz travel-time matrix
    wages: str                      # path to labor_tidy_<year>.csv
    floorspace: str                 # path to gmina floor-space price index csv
    flows: str | None = None        # path to raw bilateral matrix; None => generate
    flow_source_year: int | None = None   # TERYT vintage of the flows file (for parser)
    borrow_params_from: int | None = None  # borrow estimated phi from another year
    ttm_tag: str = ""               # short label for the run id (e.g. "osm", "garmin")


# --------------------------------------------------------------------------- #
# Solver knobs
# --------------------------------------------------------------------------- #
@dataclass
class Solver:
    prod_maxiter: int = 5000
    prod_relax: float = 0.25
    prod_precision: int = 6         # rounding digits for |income-expenditure| test
    prod_tol: float = 1e-8          # fallback abs tol if precision test not reached
    flowgen_maxiter: int = 2000
    flowgen_tol: float = 1e-3
    cf_tol: float = 1e-4
    cf_maxiter: int = 100_000
    cf_relax: float = 0.25
    # gravity
    gravity_include_diagonal: bool = False
    gravity_fe_maxiter: int = 200
    gravity_fe_tol: float = 1e-10
    strict_invariants: bool = True  # refuse to save a run that fails a hard invariant
    # IPF-rake observed commuting matrices to the trusted labour margins (R_n, L_n).
    # phi-invariant; fixes census-censoring/vintage inconsistencies. On by default.
    reconcile_margins: bool = True
    ipf_maxiter: int = 200
    ipf_tol: float = 1e-9


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
@dataclass
class Paths:
    repo_root: str | None = None    # resolved at runtime; default = dir containing config/
    crosswalk: str = "data/processed/labour_market/teryt_gmina_crosswalk_2021.csv"
    shapefile: str = "data/processed/shapefiles/communes_2021.gpkg"
    runs_dir: str = "runs"

    def resolve(self, root: Path) -> "Paths":
        out = copy.deepcopy(self)
        out.repo_root = str(root)
        return out

    def abspath(self, rel: str) -> Path:
        rel = str(rel)
        p = Path(rel)
        if p.is_absolute():
            return p
        return Path(self.repo_root) / rel


# --------------------------------------------------------------------------- #
# Top-level run configuration
# --------------------------------------------------------------------------- #
@dataclass
class RunConfig:
    params: Params
    years: dict[int, YearSpec]
    solver: Solver
    paths: Paths
    tag: str = "baseline"
    raw: dict[str, Any] = field(default_factory=dict)  # verbatim parsed YAML

    # ------------------------------------------------------------------ #
    @classmethod
    def from_yaml(cls, path: str | Path, repo_root: str | Path | None = None) -> "RunConfig":
        path = Path(path)
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

        params = Params(**(raw.get("params") or {}))

        years: dict[int, YearSpec] = {}
        for y, spec in (raw.get("years") or {}).items():
            yi = int(y)
            years[yi] = YearSpec(year=yi, **{k: v for k, v in (spec or {}).items()})

        solver = Solver(**(raw.get("solver") or {}))
        paths = Paths(**(raw.get("paths") or {}))

        # resolve repo root: explicit arg > YAML value > parent of the config dir
        if repo_root is not None:
            root = Path(repo_root)
        elif paths.repo_root:
            root = Path(paths.repo_root)
        else:
            root = path.resolve().parent.parent   # <root>/config/baseline.yaml
        paths = paths.resolve(root)

        cfg = cls(params=params, years=years, solver=solver, paths=paths,
                  tag=str(raw.get("tag", raw.get("run", {}).get("tag", "baseline"))),
                  raw=raw)
        cfg.validate()
        return cfg

    # ------------------------------------------------------------------ #
    def validate(self) -> None:
        assert 0.0 < self.params.alpha < 1.0, "alpha must be in (0,1)"
        assert self.params.epsi > 1.0, "epsi must exceed 1"
        assert self.params.sigg > 1.0, "sigg must exceed 1"
        assert self.params.phi_mode in ("estimate", "fixed")
        root = Path(self.paths.repo_root)
        assert root.exists(), f"repo_root does not exist: {root}"
        for y, spec in self.years.items():
            if spec.borrow_params_from is not None:
                assert spec.borrow_params_from in self.years, (
                    f"year {y} borrows params from {spec.borrow_params_from}, "
                    "which is not configured")

    def run_id(self, year: int) -> str:
        spec = self.years[year]
        tag = spec.ttm_tag or "road"
        return f"{year}_{tag}_{self.tag}"

    def to_dict(self) -> dict:
        return dict(
            tag=self.tag,
            params=asdict(self.params),
            years={y: asdict(s) for y, s in self.years.items()},
            solver=asdict(self.solver),
            paths=asdict(self.paths),
        )
