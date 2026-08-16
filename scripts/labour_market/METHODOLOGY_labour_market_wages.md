# Labour-market inputs for the MRRH model — methodology

Gmina-level mean **wages** and **employment counts**, by place of *work* and place
of *residence*, harmonised onto the **2021 TERYT gmina frame**, for three
cross-sections: **2011**, **2021**, and a recent **~2026** window. These are the
`labor_tidy_<year>.csv` tables consumed by `Topic_11/mrrh_pipeline/dataio.py`.

Pipeline: `scripts/labour_market/wages/` (orchestrator `run_labour.py`). This
document records every construction choice and its justification.

---

## 1. What the MRRH model actually needs

Read against the code the model runs (`dataio.load()`), not against intuition:

- The pipeline consumes **one** wage vector, `w_n = median_income_workplace`
  (mean-normalised). The residence wage `v_n` is **derived internally** as
  `v_n = condCom @ w_n` and is *never read from data*. Residence wage is a model
  input only under the §1.5 inversion fallback (`w = Λ̃⁻¹v`), which is not needed
  here because workplace wages are observed. → **workplace wage is required for
  all years; residence wage is auxiliary** (validation / robustness), produced
  only where a clean source exists (the recent window).
- Employment margins `L_n` (workplace) and `R_n` (residence) are the column and
  row sums of the commuting matrix. But both Polish commuting matrices are
  **off-diagonal only** (2011: no diagonal rows; 2021: censored at flows < 3),
  so the within-gmina diagonal `λ_nn` — 50–80 % of commuter mass, the object that
  drives amenity recovery `b_n` — is missing and must be reconstructed from an
  **independent employment margin**. → **both employment concepts are needed**;
  register workplace employment is the pivot that makes the matrix usable.

The German reference `labor_tidy.csv` carries exactly these four columns
(`median_income_workplace`, `median_income_residence`, `employment_workplace`,
`employment_residence`); we reproduce that schema.

---

## 2. Spatial frame and the TERYT crosswalk

Everything is expressed on the **2021 gmina frame** (2 477 whole gminas, the
6-digit `WWPPGG` code; RODZ 4/5 sub-parts of a type-3 gmina are dropped to avoid
double counting). This makes wages, employment, population and the bilateral
**commuting flows** share one identical unit set — the compatibility requirement.

At 6-digit level the frame is nearly static. From the three TERC snapshots and
the two change files, with **no genuine areal splits/merges recorded** (all
`Wyodrebniono`/`Wlaczono` fields empty), every inter-vintage difference is a
documented 1:1 recode, a whole-gmina merger, one dissolution, or a post-2021
carve-out:

| Source code | Name | 2021 target | Event |
|---|---|---|---|
| `022109` | Wałbrzych (gm. miejska) | `026501` | regained city-county status, 2013 |
| `080910` | Zielona Góra (gm. wiejska) | `086201` | merged into the city, 2015 |
| `320304` | Ostrowice | `320302` (Drawsko Pom.) | dissolved 2019; primary absorber |
| `120713` | Szczawa | `120705` (Kamienica) | carved out 2025 |
| `200216` | Grabówka | `200209` (Supraśl) | carved out 2025 |

All other codes map to themselves. Because the six mover codes are disjoint
across vintages and none collides with a 2021 code, a single **universal**
crosswalk (identity ∪ the five special maps) resolves *any* input file regardless
of which TERYT vintage GUS applied on export — so we never have to detect a
file's vintage. Since true splits are absent, the map is **many-to-one onto 2021**
and **no bilateral flow is ever split**, which is exactly what keeps it consistent
with the commuting matrices; `--ostrowice-split` optionally distributes the one
dissolved gmina across two absorbers (72/28 by population). Extensive quantities
(employment, flows, population) are **summed** under folding; intensive quantities
(wages) are **employment-weighted-averaged**. The crosswalk is emitted as
`teryt_gmina_crosswalk_2021.csv` for the flows/tt pipelines to consume verbatim.

---

## 3. Wages: powiat → gmina disaggregation

> **"Median" is a schema label, not the statistic.** Both P2497 and P4609 report
> *przeciętne* (arithmetic **mean**) gross wages, and we use the mean throughout
> for cross-year consistency (only the mean exists at powiat for 2011/2021). The
> output column is nonetheless named `median_income_workplace` because that is
> the exact field `dataio.load()` reads. Verified: the employment-weighted mean
> of P4609 gmina wages reproduces P2497 at powiat (ratio ≈ 0.94, sd 0.045 across
> 379 powiats) — same concept, the ~6 % gap being the products' different
> employment universes. We therefore take the **level** from P2497 and only the
> within-powiat **structure** from P4609, which is robust to that gap.

Gmina workplace wages exist only from 2024 (P4609, GUS *Rozkład wynagrodzeń*);
for 2011 and 2021 only the powiat aggregate (P2497) is observed. The powiat mean
**is** the employment-weighted mean of its gmina means, so the disaggregation is
a weighted structure-transfer that reproduces that identity exactly.

1. **Modern structure.** From the recent gmina cross-section compute each gmina's
   within-powiat relative wage
   `r_g = w_g^recent / (Σ_{j∈p} s_j^recent w_j^recent)`,
   `s_j = emp_j / Σ emp`. This carries the **full** observed within-powiat
   structure — including the gmina-specific component, *not merely what a few
   covariates explain*. (An earlier covariate-only model transferred only the
   systematic part, whose gradient is small, and so reproduced just ~43 % of the
   observed within-powiat dispersion — visible as spurious "powiat-border"
   discontinuities. The ratio transfer restores it to ~85 %.)
2. **Transfer with mild shrinkage.** `r_g^λ`, `λ` = `--struct-shrinkage`
   (default 0.90; 1.0 transfers the full modern dispersion, <1 guards against
   propagating noise from small, volatile gminas over the 10–13-year horizon).
3. **Exact benchmark with historical weights.**
   `w_{g,t} = W_{p,t} · r_g^λ / ( Σ_{j∈p} s_{j,t} · r_j^λ )`,
   `s_{j,t} = emp_{j,t} / Σ emp`, using year-*t* workplace employment as weights.
   By construction the employment-weighted gmina mean reproduces P2497 **to
   machine precision** (≈4e-16), while the within-powiat *dispersion and ranking*
   equal the modern observed structure (Spearman rank corr 2026↔2021 = 1.0).

Gminas whose modern wage is fully confidentiality-suppressed have no observed
`r_g` and receive the powiat-neutral `r ≈ 1` (hierarchical fill); their level is
still pinned by the powiat control.

The recent-year workplace wage is taken **directly** from P4609 (trailing
`--recent-window` months, default 12, ending at the freshest fully-populated
month, Feb 2026); `--benchmark-recent` optionally rakes it to the P2497 2025
anchor (off by default, since P4609 is gmina-level truth). Residence wage is
produced for the recent window only (P4609, `wg miejsca zamieszkania`); for
2011/2021 it has no powiat anchor and is left NaN — which is all the model would
use, since it derives `v_n` internally.

**Key assumption.** The within-powiat wage *structure* (the ranking and relative
spread of gmina means) is stable back to 2011/2021; the *level* is pinned to the
observed powiat mean each year. λ bounds how much of the modern structure is
carried. This is the defensible position when the fine-grained series begins only
recently, and it is exactly the employment-weighted logic the powiat mean implies.

**Suppression.** GUS encodes confidential wages (single-dominant-employer gminas
— mining, military) as `0,00`; these are read as missing. When a gmina's recent
window is fully suppressed it falls back to its own longer history, then to a
hierarchical powiat→woj→national prior; per-gmina provenance is recorded in
`wage_workplace_source` (`P4609_window` / `P4609_own_history` / `P4609_hier_impute`
/ `P2497_disaggregated`). In the current data 2 465 gminas are direct-window,
3 own-history, 9 hierarchically imputed.

**Key assumption.** The within-powiat wage *structure* (which gmina types/sizes
pay above their powiat mean) is stable in relative terms 2011→2024; the *level*
is pinned to observed powiat totals each year. This is the standard defensible
position when the fine-grained series only begins recently, and the shrinkage
parameter bounds its influence.

---

## 4. Employment

**Workplace** (`employment_workplace`): register `pracujący` at gmina — P2172 for
2011/2021, P4508 (2025) for the recent window. Within-gmina temporal
interpolation fills GUS-suppressed year cells; the same imputed series is used
both as the wage-rake weight and as the output value, so the benchmark holds
exactly. Imputed cells are flagged (`emp_work_imputed`).

**Residence** (`employment_residence`):
- *Recent window*: directly from P4280 (residence `pracujący`, trailing-window
  mean).
- *2011 / 2021*: recovered from the accounting identity with the harmonised
  flows, `R_n = max(L_n − inflows_n, 0) + outflows_n` — a data-driven gmina shape
  from the actual commuting structure, **not** an assumption that residence
  employment tracks working-age population. Where the census income-earner
  control (P3357/P4488, powiat "utrzymujący się z pracy") is present it is raked
  within powiat to that total, correcting the register's undercount of small-firm
  and farm employment (the east/rural-correlated bias); working-age population
  (P3457/P4362) is the fallback shape. `res_source` records the route.
  *When the census control files are absent the residence employment is still
  produced (register+flows-recovered) but is not level-calibrated —* the run logs
  a warning, and dropping P3357/P4488 into `data/raw/labour_market/` upgrades it
  automatically.

### ⚠ Employment definitional break across cross-sections

The gmina employment universe **changes between GUS products**: P2172 (2011,
2021) reports ≈ 8.7 M / 9.8 M nationally (the narrower BDL concept, excluding
entities ≤ 9 employees and individual farms), whereas P4508/P4280 (2026) report
≈ 15.1 M (the broad `pracujący`). Because MRRH normalises each cross-section's
`L_n`, `R_n` to mean 1, this does **not** contaminate any single-year calibration,
but employment **levels are not comparable across the three years**. Do not read
2011→2026 employment growth off these tables. (Harmonising the concept would
require a consistent long gmina series that GUS does not publish.)

---

## 5. Missing values

Every gmina receives a workplace wage and workplace employment by construction
(the powiat anchor + rake leave no gaps). Residual gaps in covariates and in the
recent direct wages are filled hierarchically (powiat → voivodeship → national
median), and thin cells borrow strength through the shrinkage step. All
imputation routes are logged and flagged per gmina.

---

## 6. Validation gates (enforced in `run_labour.py`, STAGE 5)

- Frame completeness: exactly 2 477 unique gminas per year.
- No missing workplace wage or workplace employment.
- **Exact benchmark**: employment-weighted within-powiat gmina wage reproduces
  P2497 (`max_rel_err < 1e-6`; observed ≈ 3e-16).
- Crosswalk weights sum to 1 per source code; every target ∈ 2021 frame.
- Flow gminas ⊆ 2021 frame (compatibility with the commuting matrices).

`labor_build_diagnostics.json` records the wage method and λ, provenance counts,
census-control presence, and the full validation report.

---

## 7. Outputs

```
data/processed/labour_market/
  labor_tidy_2011.csv   labor_tidy_2021.csv   labor_tidy_2026.csv
  teryt_gmina_crosswalk_2021.csv
  labor_build_diagnostics.json
```

Columns: `region_id` (6-digit 2021 gmina), `teryt7`, `nazwa`, `powiat`,
`rodz_class`, `median_income_workplace`, `median_income_residence`,
`employment_workplace`, `employment_residence`, `wage_workplace_source`,
`res_source`, `emp_work_imputed`.

---

## 8. Data sources

| File | GUS symbol | Level | Coverage | Role |
|---|---|---|---|---|
| P2497 | mean wage, workplace | powiat | 2002–2025 | wage anchor (control total) |
| P4609 | mean wage, both concepts | gmina | monthly 2024–2026 | recent wage + transfer-model training |
| P2172 | employment, workplace | gmina | 2010–2021 | `L_n` 2011/2021; wage covariate |
| P4508 | employment, workplace | gmina | 2023–2025 | `L_n` recent |
| P4280 | employment, residence | gmina | monthly 2022–2026 | `R_n` recent |
| P3457 / P4362 | working-age population | gmina | 2011 / 2021 | residence-emp fallback shape |
| P3357 / P4488 | population by source of income | powiat | 2011 / 2021 | residence-emp control (auto-detected) |
| bilateral 2011 / 2021 | commuting matrix | gmina | 2011 / 2021 | diagonal recovery for `R_n` |
| TERC ×3 + change ×2 | administrative register | — | 2011/2021/2026 | crosswalk |
