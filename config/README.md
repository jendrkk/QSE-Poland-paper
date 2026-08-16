# `config/` — central configuration

`baseline.yaml` is the single source of truth for a baseline run. The orchestrator
(`run_baseline.py`) reads it, passes it to the toolkit, and copies the resolved
version into every run folder for reproducibility.

## Blocks

- **`params`** — structural parameters. Each default carries a literature source in
  a comment (see `qse_poland_paper/config.py` for full provenance). `phi = eps*mu`
  is estimated in-sample (`phi_mode: estimate`); `alpha`, `epsi`, `sigg`, `nu`,
  `delta`, `psi` are calibrated knobs with `sensitivity` grids. Change a value here
  and every run picks it up — nothing is hard-coded in the package.
- **`years`** — per-year data specification: the travel-time matrix (`ttm`), wages
  (`wages`), floor-space index (`floorspace`), and the bilateral flow matrix
  (`flows`). Set `flows: null` to generate flows from the margins + travel times
  (2026); `borrow_params_from` names the year whose estimated `phi` to reuse.
  `ttm_tag` becomes part of the run id: `runs/<year>_<ttm_tag>_<tag>/`.
- **`solver`** — tolerances and gates (productivity fixed point, counterfactual
  fixed point, gravity FE, IPF margin reconciliation, `strict_invariants`).
- **`paths`** — repository-relative locations (crosswalk, shapefile, `runs/`).
  `repo_root` is resolved at runtime (default: the folder containing `config/`);
  override with `run_baseline.py --repo-root /path/to/repo`.

## Adding a scenario

To evaluate a different road network, add a year entry pointing `ttm` at the new
matrix and give it a distinct `ttm_tag`; solve it, then compare against a baseline
run with `run_viz.py run_a.pkl run_b.pkl` (which computes the general-equilibrium
welfare and reallocation of the network change).
