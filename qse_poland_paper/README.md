# `qse_poland_paper` — MRRH baseline toolkit

Back-end ("src") for calibrating and solving the Monte–Redding–Rossi-Hansberg
(2018) quantitative spatial model on the ~2,477 Polish gminas, for several yearly
cross-sections (2011, 2021, 2026), **no rail, no congestion** in the baseline.

The verified model core (inversion and exact-hat-algebra counterfactual) is
ported from the Topic-11 reference implementation (the professor's MATLAB
`*TK.m` and the students' `mrrh_pipeline`), checked line-for-line against SW2020
eqs. 10/12 and Codebook §A.2, and re-homed into an extensible package driven by a
central YAML config and a thin orchestrator.

## Layout

```
qse_poland_paper/          back-end package
  config.py                typed config + YAML loader
  frame.py                 canonical 2021 gmina frame; TERYT helpers
  io/                      data loaders (labour, flows, ttm, floorspace, trade)
  quantify.py              productivity/trade inversion, amenity residual  (verified)
  flowgen.py               gravity flow generation for the no-flows year   (verified)
  estimate.py              commuting-gravity phi (two-way FE); future seams
  counterfac.py            exact-hat-algebra solver + road-network helper   (verified)
  costs.py                 commuting-cost aggregator (rail/congestion seams)
  solve.py                 calibrate_year(): one year end-to-end -> RunResult
  result.py                RunResult: the single self-describing run artefact
  validate.py              model invariants
  viz/                     style, maps, figures, tables
config/baseline.yaml       central configuration (parameters, per-year data, solver)
run_baseline.py            orchestrator (front-end): writes runs/<run_id>/
run_viz.py                 visualization runner (1 run, or comparison of >=2)
```

## Quick start

```bash
pip install -r requirements.txt          # numpy, pandas, pyyaml, geopandas, mapclassify,
                                          # matplotlib, openpyxl, xlrd

# calibrate & solve all configured years -> runs/<year>_<ttm>_baseline/run.pkl
python run_baseline.py --config config/baseline.yaml

# visualise one run (figures + maps + LaTeX tables into its run folder)
python run_viz.py runs/2021_osm_baseline/run.pkl

# compare two runs (adds delta maps + the road-network GE counterfactual)
python run_viz.py runs/2011_garmin_baseline/run.pkl runs/2021_osm_baseline/run.pkl \
    --dpi 300 --format pdf            # --no-transparent / --no-usetex available
```

Each run writes a single pickled `RunResult` (`run.pkl`) holding **every**
parameter and result array, plus `manifest.json`, `config.resolved.yaml` and a
log. `run.pkl` is the authoritative artefact; load it with
`qse_poland_paper.result.RunResult.load(path)`.

## What the baseline produces, per year

Recovered fundamentals `A_n` (productivity) and `b_n` (amenity); trade shares
`pi_ni` and the tradable price index `P_n`; commuter market access `CMA_n`; the
assembled commuting matrix (observed for 2011/2021, generated for 2026) and its
margins; the estimated commuting decay `phi = eps*mu`; and derived objects (real
residence income, CPI). Every quantity is stored for visualization.

## Correctness

Under identity forcing the counterfactual solver returns welfare `= 1` and every
hat `= 1` to machine precision — the primary analytic check. Each run also
records the invariant suite (`Sigma L_n = Sigma R_n = N`, `uncondCom.sum = 1`,
income `=` expenditure, `geomean(b_n) = 1`, `A_n > 0`); the orchestrator refuses
to save a run that fails a hard invariant.

## Extensibility (designed-for, not implemented)

Rail via a second TTM through `costs.modal_aggregate` (`tau -> tilde tau`,
everything downstream unchanged); mode-share identification via
`estimate.modal_params`; congestion via `costs.congestion`; partition-border
analysis via `frame` running-variable fields and `counterfac` forcings; freight
modes via `io/trade`. Each is a named seam, not a rewrite.

## Known limitations (current baseline)

- Floor-space price `Q_n` is a single 2021 hedonic index, reused for all years
  (per-year swap-in is a one-line config change once the indices exist).
- `alpha` and other structural parameters are documented, sourced config knobs
  with sensitivity grids; `phi` is estimated in-sample every year (2026 borrows
  2021's `phi`).
- Observed flow matrices are IPF-raked to the trusted labour margins (phi-
  invariant) to fix census-censoring / TERYT-vintage inconsistencies.
