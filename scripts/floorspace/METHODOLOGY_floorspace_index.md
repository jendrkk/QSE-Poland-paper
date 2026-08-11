# A Static (2021) Gmina-Level Hedonic Residential Floorspace Price Index for Poland

**Purpose.** Produce one quality-adjusted price of 1 m² of residential *living space* for every 2021 gmina (≈2,477 units), with a single, comparable interpretation across urban, rural, and urban–rural gminas, for calibrating an MRRH-type quantitative spatial model. The index pools **all** floorspace-bearing RCN transaction microdata (flats + houses), corrects and disaggregates it with official GUS powiat-level transaction prices, and reflects the 2021 price level.

This document specifies exactly which information is drawn from each input file, the cleaning rules, the spatial assignment, the econometric model, and the output. It is the design contract for the code in `scripts/floorspace/`.

---

## 0. Summary of the empirical situation (why the design is what it is)

Findings from a full profiling of the three RCN GeoPackages, the two GUS files, the TERYT tables, and `communes_2021.gpkg`:

1. **`teryt` in RCN is a 4-digit *powiat* code** (uniform across all three layers, 371–379 distinct powiats). It does **not** encode gmina. Gmina must come from a **spatial join** of the transaction geometry to gmina polygons. Geometry is present on 100 % of rows, valid, in **EPSG:2180**.

2. **The three layers are one transaction decomposed**, linked by `tran_lokalny_id_iip` (66 % of `budynki` and 37 % of `dzialki` transactions also appear in `lokale`). This creates double-counting risk that the pipeline resolves by defining a single unit of observation per layer role.

3. **Only two layers carry usable floor area:**
   - `lokale` (flat sales, POINT): `lok_pow_uzyt` present on **99.8 %** of the 2.13 M residential rows. This is the workhorse (~1.46 M clean arm's-length flats).
   - `budynki` (building polygons): usable floor area `bud_pow_uzyt` exists for only **7 % of built-up residential house sales** (~50–97 k nationally). This is the *only* source of genuine house floorspace prices; it is scarce but essential for rural coverage.
   - `dzialki` (land parcels) has **no building floor area at all** → it cannot price floorspace directly. It is used for (a) *land-price* covariates that help predict thin gminas and (b) built-up/undeveloped classification.

4. **GUS `P3787`/`P3788` are the officially-cleaned powiat aggregates of the *same* RCN registry, restricted to `lokale mieszkalne` (flats in multi-dwelling buildings; single-family houses excluded).** Proof: our cleaned RCN `lokale` 2021 powiat medians reproduce GUS median almost exactly — **corr = 0.982**, median ratio **RCN/GUS = 1.005** (p10 0.98, p90 1.08) over 162 common powiats. Consequences:
   - GUS is a **concept-matched anchor for the flat component**; the house component is tied in through a hedonic property-type differential.
   - GUS is not independent of RCN flats (same source), but it still adds value: official stratified cleaning, complete powiat/year coverage, an external validated *level*, and a clean temporal path. We therefore use it as the **prior mean / auxiliary covariate** in a small-area model rather than as independent evidence.
   - The near-perfect reproduction is a **built-in validation gate** for the cleaning pipeline.

5. **Concentration, not raw quality, is the binding constraint.** Top-10 powiats hold 43 % of flat transactions; the bottom 50 % of powiats hold 4.5 %; 65 powiats have < 200 flat sales. At gmina resolution many rural gminas have few or zero flat sales. This is why (a) houses/land are pooled in and (b) a hierarchical shrinkage-to-GUS estimator is used.

6. **Data-quality problems are material and must be filtered** (magnitudes on the clean flat subset, all years, n≈1.71 M): ~5.3 % zero/negative unit prices (developer records with the price only in the bundle field), ~7 % below 1,000 zł/m², extreme upper outliers (area mis-entry → price/m² in the millions; mean 19,960 vs p99.9 34,891), ~921 k NULL ownership shares plus explicit fractional-share deals, corrupted dates (year 0006–9200), and ~106 k duplicate groups.

---

## 1. Inputs and exactly what is taken from each

All paths are relative to the repository root `QSE_Poland_paper/`.

### 1.1 `data/raw/floorspace/lokale.gpkg` — flats (primary)
POINT geometry, EPSG:2180, 2,593,413 rows. Fields used:

| Field | Use |
|---|---|
| `geom` (POINT) | gmina assignment via spatial join |
| `lok_funkcja` | keep `mieszkalna`; drop `garaz`, `inne`, `handlowoUslugowa`, `biurowa`, `produkcyjna` |
| `lok_pow_uzyt` | floor area (m²) — denominator of price/m² and hedonic size regressor |
| `lok_cena_brutto` → `nier_cena_brutto` → `tran_cena_brutto` | unit price by **coalescing cascade**; `tran` only used when the transaction is single-unit (see §2.2) |
| `lok_liczba_izb` | rooms (structural control) |
| `lok_nr_kond` | storey/floor (structural control; negatives = basement) |
| `tran_rodzaj_rynku` | primary/secondary market dummy (`pierwotny`/`wtorny`) |
| `tran_rodzaj_trans` | keep `wolnyRynek` only |
| `tran_sprzedajacy` | flag/exclude public sellers (`jednostkaSamorzaduTerytorialnego`, `skarbPanstwa`) as robustness |
| `nier_udzial` | ownership share; keep `1/1` (or NULL when single-unit); drop fractions |
| `dok_data` | transaction year → year FE; date-sanity filter |
| `tran_lokalny_id_iip`, `lok_id_lokalu` | transaction/unit identity → dedup and single-unit detection |
| `teryt` | powiat code → cross-check against geometry-derived powiat |

### 1.2 `data/raw/floorspace/budynki.gpkg` — houses (rural extension)
MULTIPOLYGON, EPSG:2180, 6,233,436 rows. We keep **only genuine whole-house sales with floor area**:
`nier_rodzaj = 'nieruchomoscGruntowaZabudowana'` (house on its plot) **and** `bud_rodzaj = 'mieszkalny'` **and** `bud_pow_uzyt` non-null (~97 k rows; ~50 k after quality filters). Fields:

| Field | Use |
|---|---|
| `geom` (polygon) → representative point | gmina assignment |
| `bud_pow_uzyt` | house usable floor area (m²) |
| `bud_cena_brutto` → `nier_cena_brutto` → `tran_cena_brutto` | price cascade (single-unit rule) |
| `bud_rodzaj`, `nier_rodzaj` | residential/house filter |
| `tran_rodzaj_trans`, `tran_rodzaj_rynku`, `nier_udzial`, `tran_sprzedajacy`, `dok_data`, `tran_lokalny_id_iip` | same roles as in `lokale` |
| `nier_pow_gruntu` | plot area (house-specific control; captures land component) |

Rows with `nier_rodzaj = 'nieruchomoscLokalowa'` in `budynki` are the building footprints of flat sales already represented in `lokale` and are **discarded** (double-counting).

### 1.3 `data/raw/floorspace/dzialki.gpkg` — land (covariates only)
MULTIPOLYGON, EPSG:2180, 9,868,027 rows. **Not** a floorspace source. We derive gmina-level **auxiliary covariates**:

- **Residential land price**: from `nier_rodzaj = 'nieruchomoscGruntowaNiezabudowana'` (undeveloped) with `dzi_przezn_wmpzp`/`dzi_sposob_uzyt` indicating residential zoning (`budownictwoMieszkanioweJedno/Wielorodzinne`, `gruntyZabudowaneIZurbanizowane`), `tran_rodzaj_trans='wolnyRynek'`, price `nier_cena_brutto`/`dzi_cena_brutto` over `dzi_pow_ewid` (100 % populated). Aggregate to a robust gmina **median land zł/m²**.
- **Built-up share / transaction intensity**: counts of built-up vs undeveloped residential parcels per gmina (a development-pressure proxy).

These feed the small-area model as predictors of the gmina price level, materially helping gminas with few/zero house or flat sales (land transactions are far more numerous in rural areas).

### 1.4 `data/raw/floorspace/GUS_P3787_median_…csv` and `GUS_P3788_mean_…csv`
Semicolon-delimited BDL exports, UTF-8, 397 data rows (POLSKA + 16 voivodeships + ~380 powiats), header dimensioned **market × size-class × year**:
- market ∈ {`ogółem`, `rynek pierwotny`, `rynek wtórny`}
- size ∈ {`ogółem`, `do 40 m2`, `od 40,1 do 60 m2`, `od 60,1 do 80 m2`, `od 80,1 m2`}
- year ∈ {2010,…,2024}
- code `WWPPGGG` with `GGG=000` for powiat aggregates (powiat = first four digits, matches RCN `teryt`).

**Zeros are suppression → treated as missing.** We extract:
- **Anchor**: `median, ogółem, ogółem, 2021` per powiat (primary); `mean, …, 2021` as robustness/auxiliary.
- **Composition**: primary/secondary and size-class 2021 values (to sanity-check the hedonic market and size coefficients).
- **Temporal series 2010–2024** per powiat: used to (a) validate the national hedonic year FE against official local price paths and (b) optionally deflate under the alternative temporal scheme.

### 1.5 `data/processed/shapefiles/communes_2021.gpkg`
2,477 gmina polygons, **EPSG:3035**, key fields `JPT_KOD_JE` (7-digit TERYT), `pop`, `centroid`, `weighted_centroid`. This defines the target spatial units and provides population (an SAE covariate and an aggregation weight). Reprojected to EPSG:2180 for the join (reprojecting 2,477 polygons is far cheaper than millions of points).

### 1.6 `data/raw/teryt/TERC_Urzedowy_2021-01-01.csv` (+ 2011 + change XML)
`WOJ;POW;GMI;RODZ;NAZWA;…`. Provides gmina **names** and **RODZ type** (1 miejska / 2 wiejska / 3 miejsko-wiejska / 4 miasto / 5 obszar wiejski). RODZ is a structural covariate (housing-stock composition differs sharply by type) and drives the urban-flats / rural-houses logic. The 2011 table + `TERC…zmiany….xml` are used only if any historical TERYT reconciliation is required (RCN spans years with boundary changes); the 2021 vintage is authoritative for the static index.

---

## 2. Stage 1 — Micro data construction (per layer)

### 2.1 Numeric parsing
All numeric fields are TEXT. Parse defensively: strip spaces, normalise decimal comma → dot, cast to float; non-parseable → NaN.

### 2.2 Price definition (avoids the main mechanical bias)
Unit price = first non-null of (`{lok|bud}_cena_brutto`, `nier_cena_brutto`, `tran_cena_brutto`), **except** `tran_cena_brutto` is used only when the transaction is single-unit (`tran_lokalny_id_iip` maps to exactly one retained row). Rationale: `tran_cena_brutto` is the whole-bundle price and is repeated across all rows of a multi-item deal (e.g. flat + 2 garages), so dividing it by one unit's area over-prices. All prices are gross (brutto); primary-market gross includes VAT — absorbed by the market dummy.

### 2.3 Transaction filters (applied to `lokale` and `budynki`)
- `tran_rodzaj_trans = 'wolnyRynek'` (drops bonifikata/tender/foreclosure/non-tender municipal sales — e.g. `sprzedazZBonifikata` are 80–95 % below market).
- Ownership: `nier_udzial ∈ {'1/1'}` or NULL when single-unit; drop explicit fractions and malformed `'/'`.
- Residential function/type filters (§1.1, §1.2).
- Date sanity: `2006 ≤ year ≤ 2026` (configurable); year from `dok_data[:4]`.
- **Deduplicate** on (`tran_lokalny_id_iip`, `lok_id_lokalu`/`bud_id_budynku`), then on (unit id, date, price).

### 2.4 Outlier trimming (two tiers)
1. **Hard sanity gate** (domain bounds): price > 0; area ∈ [15, 400] m² for flats, [25, 1000] m² for houses; price/m² ∈ [1,000, 50,000] zł. Removes the pathological ~6–8 %.
2. **Robust within-stratum trim**: within `powiat × property-type × market × year` strata, drop observations with |z| > 4 on log price/m² using the median/MAD (1.4826·MAD) robust scale; strata with < ~30 obs fall back to `powiat × property-type`. Removes ~1–1.5 % extreme tails without distorting central mass. Trimming thresholds are CLI-configurable.

### 2.5 Geometry → representative point
`lokale` are already points. For `budynki`/`dzialki` we take a **representative point on surface** (guaranteed interior) of the polygon. Points remain in EPSG:2180.

Output of Stage 1: a tidy micro table `micro` with columns
`{price, area, log_ppm2, property_type∈{flat,house}, market, rooms, floor, plot_area, year, x2180, y2180, powiat_teryt, weight}` — flats and houses stacked, plus a separate gmina-level `land` table from `dzialki`.

---

## 3. Stage 2 — Spatial assignment to gmina

1. Load `communes_2021.gpkg`, reproject 3035 → 2180.
2. Point-in-polygon join of `micro` points to gmina polygons using a prepared-geometry STRtree, parallelised over CPU cores (joblib/loky), mirroring `commune_centroids.py`. Each transaction receives `gmina_teryt` (`JPT_KOD_JE`).
3. **Validation gate**: geometry-derived powiat (`gmina_teryt[:4]`, mapped through TERC) must equal the record's RCN `teryt`; the mismatch rate is logged and mismatches are flagged/optionally dropped (geocoding/boundary error indicator).
4. Aggregate the `dzialki` land table to gmina medians in the same way (representative points → gmina).

Result: every retained transaction has a 2021 gmina; gmina-level land covariates are attached.

---

## 4. Stage 3 — National hedonic regression (the "direct" estimator)

Pooled OLS on all retained flats + houses:

```
log(price_i / area_i) = α
      + f(log area_i)                     # size gradient (log area + quadratic)
      + β_house · 1[house]_i              # property-type differential (ties houses to flats)
      + β_house · (house-specific terms)  # log plot area, interacted with 1[house]
      + β_mkt · 1[primary]_i              # market level shift
      + β_rooms · rooms_i + β_floor · floor_i
      + δ_t   (year fixed effects, δ_2021 ≡ 0)      # STATIC-2021 normalisation
      + γ_g   (gmina fixed effects)                  # quality-adjusted gmina level
      + ε_i
```

- **Identification of the static 2021 index.** Year FE with the 2021 reference make `γ_g` a time-invariant, quality-adjusted gmina price level expressed at 2021. Pooling all years with a common national `δ_t` imposes that *relative* cross-gmina prices are time-stable and price growth is proportional nationwide. This is the baseline the user selected ("first with hedonic year FE, use all data"). Robustness knobs relax it: `--time-fe {year, woj_year}` (voivodeship×year trends) and `--year-min/--year-max` (window restriction, e.g. 2016–2024).
- **Estimation.** High-dimensional two-way FE (≈2,477 gmina + ≤21 year dummies + ~10 controls) solved as a sparse linear system: build the sparse design `X` (scipy.sparse), form the normal equations `XᵀX` (dense ≈2,510², trivially invertible) with a tiny ridge for numerical stability, solve for (β, γ, δ). This is exact, dependency-light, and multicore via BLAS. No `pyfixest`/`lfe` dependency required.
- **Direct estimate per gmina.** `θ̂_g = α + γ_g + β·x̄_ref` = log price/m² of a reference (sample-mean-characteristics) dwelling in gmina g at 2021, with sampling variance `D_g` from the FE block of `σ̂²(XᵀX)⁻¹` (heteroskedasticity-robust; optionally clustered by gmina). Gminas with **zero** retained transactions have no `θ̂_g` (handled in Stage 4). The overall level (α) is only pinned in Stage 4 by GUS, so the arbitrary reference-composition choice is immaterial.
- **Diagnostics.** R², coefficient signs/magnitudes (expect concave size gradient, primary > secondary by ~10–15 %, house < flat per m²), and comparison of the estimated `δ_t` path to the GUS POLSKA 2010–2024 series.

---

## 5. Stage 4 — Small-area model: hierarchical shrinkage to the GUS prior

This is where sparse gminas are corrected with GUS and where zero-data gminas get an imputed value. It is a **two-level Fay–Herriot area-level model** (gmina nested in powiat), i.e. parametric empirical/hierarchical Bayes — matching the requested "hierarchical Bayesian prior."

**Area model** (one observation per gmina g in powiat p):
```
θ̂_g = θ_g + e_g,              e_g ~ N(0, D_g)        # sampling error, D_g from Stage 3
θ_g = xᵀ_g η + u_p + v_g,      u_p ~ N(0, τ²), v_g ~ N(0, A)
```
with area covariates `x_g`:
- `log GUS_median_powiat,2021` (the dominant predictor; coefficient ≈ 1 expected),
- RODZ gmina type (urban/rural/urban–rural),
- `log gmina residential land price` (from `dzialki`),
- log population / population density (`communes_2021`),
- gmina flat-share of transactions (composition control linking the flats-only GUS to the unified target).

**Behaviour by data richness:**
- Data-rich gmina (small `D_g`): posterior ≈ direct estimate `θ̂_g`.
- Thin gmina (large `D_g`): posterior shrinks toward the model mean `xᵀ_g η + u_p`, i.e. toward the **GUS powiat level** adjusted for the gmina's type/land/population.
- **Zero-data gmina**: `θ̃_g = xᵀ_g η + û_p` — a pure prediction from GUS + covariates. Every one of the 2,477 gminas therefore receives a defined value with an uncertainty band.

**Estimation.** Default: empirical Bayes — variance components (τ², A) by REML (Fisher scoring / EM), then BLUP/posterior means and MSE (Prasad–Rao). This needs only numpy/scipy and runs in seconds on 2,477 areas. Optional `--bayes` flag: full MCMC (numpyro/NUTS) on the same area model for exact posterior credible intervals (cheap at area level), behind a guarded import.

**GUS mean vs median.** Median GUS is the primary anchor (robust, matches the robust hedonic). GUS mean enters as an additional covariate / robustness anchor; the size-class and primary/secondary GUS cells validate the hedonic composition coefficients.

**Optional exact powiat benchmarking** (`--benchmark-to-gus`): after shrinkage, multiplicatively rescale gmina levels within each powiat so that the transaction- (or population-) weighted mean of gmina indices equals the GUS powiat 2021 value exactly. This enforces hard consistency with the official statistic on top of the soft Bayesian pull; off by default (the Bayesian anchor already ties levels), available for calibration coherence.

---

## 6. Stage 5 — Output

- `index_g = exp(θ̃_g)` — **zł per m² of residential living space, 2021, quality-adjusted**, for all 2,477 gminas, with posterior SD / credible interval.
- Companion columns: number of retained flat and house transactions, effective sample, shrinkage weight `A/(A+D_g)` (transparency on how much is RCN vs GUS/model), data-source composition, powiat GUS value, validation flags.
- Deliverables:
  - `data/processed/floorspace/gmina_floorspace_index_2021.parquet` and `.csv` (keyed by `JPT_KOD_JE`),
  - `data/processed/floorspace/gmina_floorspace_index_2021.gpkg` (index joined to commune polygons for mapping),
  - a run log and a diagnostics bundle (GUS-vs-index scatter, coverage, coefficient table, year-FE-vs-GUS path).

All GeoPackage writes go through a local scratch dir then move (NFS-safe), following `commune_centroids.py`.

---

## 7. Assumptions, risks, and how the design mitigates them

1. **Proportional national price growth (year-FE pooling).** Baseline assumption; mitigated by the `woj_year` option and window restriction, and validated against GUS local series.
2. **Flats-only GUS anchoring a flats+houses target.** Mitigated by the hedonic property-type differential (houses priced on the same scale as flats before anchoring) and by including gmina flat-share/RODZ in the area model. Residual house/flat level gaps are absorbed into `η`.
3. **House floorspace scarcity (~50 k).** Houses inform the *national* property-type and size gradients (well-identified from pooled data) and the rural area-model covariates; they are not required to be dense in every gmina.
4. **GUS ≠ independent of RCN flats.** Acknowledged; GUS is used as a prior/auxiliary and validation, not as independent measurement. The 0.982 reproduction is a cleaning check, not double counting.
5. **Sparsity is the true limit.** Rural gmina values are effectively GUS-powiat + covariate predictions; the shrinkage weight column makes this explicit and honest for the paper.
6. **Boundary/temporal TERYT changes** across the RCN span are handled by assigning gmina from 2021 polygons directly (geometry, not historical codes), sidestepping code-vintage issues.

---

## 8. How to run

```
python scripts/floorspace/run_index.py \
    --year-min 2006 --year-max 2026 \
    --time-fe year \
    --workers $(nproc) \
    [--bayes] [--benchmark-to-gus] \
    [--sample-frac 0.02]        # fast smoke test
```
Defaults resolve every path from the repository layout. `--sample-frac` runs the whole pipeline on a random subset for a quick end-to-end check before the full HPC run. See module docstrings for the full parameter list.
```
```
