# scripts/labour_market/wages

Builds the gmina-level **wage** and **employment** inputs for the MRRH model:
mean wages and employment by place of *work* and *residence*, harmonised onto the
**2021 TERYT gmina frame**, for **2011 / 2021 / ~2026**.

Full rationale: [`METHODOLOGY_labour_market_wages.md`](./METHODOLOGY_labour_market_wages.md).

## Run

```bash
python scripts/labour_market/wages/run_labour.py
```

Writes to `data/processed/labour_market/`:
`labor_tidy_2011.csv`, `labor_tidy_2021.csv`, `labor_tidy_2026.csv`,
`teryt_gmina_crosswalk_2021.csv`, `labor_build_diagnostics.json`.

Useful flags:

```bash
--recent-window 6            # trailing months for the recent cross-section (default 12)
--struct-shrinkage 0.75      # shrink the transferred within-powiat wage gradient
--benchmark-recent           # rake the recent P4609 gmina wage to the P2497 anchor
--residence-emp-method flows_then_rake|workage_rake|off
--ostrowice-split            # split the dissolved gmina across two absorbers
--output-dir PATH  --log-file PATH
```

## Modules

| file | role |
|---|---|
| `config.py` | paths, cross-sections, crosswalk special cases, tunables |
| `teryt.py` | 2021-anchored gmina crosswalk + harmonisation onto the frame |
| `gus.py` | GUS BDL wide-CSV parsers (wages, employment, working-age pop, census income) |
| `flows.py` | parse + harmonise the bilateral commuting matrices; per-gmina margins |
| `estimate.py` | within-powiat wage transfer + exact benchmark; residence-employment recovery |
| `run_labour.py` | orchestrator (CLI, logging, validation, outputs) |

## Optional data to complete the residence side (2011/2021)

Drop **P3357** (NSP 2011) and **P4488** (NSP 2021) — powiat population by main
source of income ("utrzymujący się z pracy") — into `data/raw/labour_market/`.
They are auto-detected and used to level-calibrate residence employment. Without
them residence employment is still produced (register+flows-recovered) but is not
level-calibrated; the run logs a warning.

## Note on employment comparability

Employment **levels are not comparable across the three years**: 2011/2021 use the
narrower P2172 concept (≈9 M), the recent window uses the broad P4508/P4280 concept
(≈15 M). MRRH normalises each cross-section, so single-year calibration is
unaffected; do not read cross-year employment growth off these tables.
