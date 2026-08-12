# A Static (2021) Gmina-Level Hedonic Residential Floorspace Price Index for Poland

**Purpose.** One quality-adjusted price of 1 m² of residential *living space* for every 2021 gmina (≈2,477), with a single interpretation across urban, rural, and urban–rural gminas, for calibrating an MRRH-type quantitative spatial model. The index pools **all** floorspace-bearing transaction microdata — flats (`lokale`) and houses (via BDOT10k-estimated usable area) — nets out plot-land value so it measures enclosed living space, disaggregates and corrects it with official GUS powiat prices, and reflects the 2021 price level.

This document is the design contract for the code in `scripts/floorspace/`. **v2** adds: BDOT10k-based house usable area (13× more house observations), a marginal rural-house route via parcels, explicit plot-land netting, and an explicit market covariate.

---

## 0. Empirical situation (why the design is what it is)

1. **`teryt` in RCN is a 4-digit *powiat* code** across all layers. Gmina must come from a **spatial join** of the transaction geometry (100 % present, EPSG:2180) to 2021 gmina polygons.

2. **The RCN layers are one transaction decomposed**, linked by `tran_lokalny_id_iip` (66 % of `budynki` and 37 % of `dzialki` transactions also appear in `lokale`). Deduplication is therefore always by **transaction id**, never by building/parcel.

3. **GUS `P3787`/`P3788` are the officially-cleaned powiat aggregates of the same RCN registry, restricted to flats** (`lokale mieszkalne`; single-family houses excluded). Our cleaned RCN `lokale` 2021 powiat medians reproduce GUS median at **corr 0.982, ratio 1.005**. GUS is thus a concept-matched anchor for the *flat* level and a validation gate; the house level is tied to flats through the hedonic property-type differential.

4. **BDOT10k enrichment unlocks house floorspace.** `budynki_bdot10k.gpkg` adds, per building, a footprint (from geometry), a BDOT10k floor count, and `usable_area_est_m2 = footprint × floors × ~0.73` (usable/gross factor). For built-up residential house sales this raises usable-area coverage from **19 % → 98 %**, i.e. **~50 k → ~456 k** clean arm's-length house transactions (overlap-matched). This is the single biggest improvement to rural coverage.

5. **Concentration, not raw quality, is the binding constraint.** Top-10 powiats hold 43 % of flat sales; 65 powiats have < 200. Houses + hierarchical shrinkage to GUS fill the rural gap.

6. **Self-constructed data must be audited** (see §4B). The BDOT10k join has a `match_type` in {overlap 90 %, nearest 8 %, unmatched 2 %}; `nearest` matches systematically grab small outbuildings (median footprint 24 m²) and are excluded. BDOT floor counts show artefact spikes (5, 11) and tall "houses" (whole apartment blocks); floors are capped. Parcel areas in `dzialki` are ~9 % ha-coded — geometry area is used instead.

---

## 1. Inputs and exactly what is taken from each

### 1.1 `lokale.gpkg` — flats (primary)
POINT, EPSG:2180, 2.59 M rows. Both markets are kept (market is a covariate, not a filter). Fields: `geom` (→ gmina), `lok_funkcja='mieszkalna'`, `lok_pow_uzyt` (area), price cascade `lok→nier→tran` (tran only when single-unit), `lok_liczba_izb` (rooms), `lok_nr_kond` (flat storey), `tran_rodzaj_rynku` (→ market ∈ {primary, secondary, unknown}), `tran_rodzaj_trans='wolnyRynek'`, `nier_udzial` (1/1 or NULL-single), `dok_data` (year), `tran_lokalny_id_iip`/`lok_id_lokalu` (dedup), `teryt` (powiat cross-check).

### 1.2 `budynki_bdot10k.gpkg` — houses (PRIMARY, v2)
MULTIPOLYGON, EPSG:2180, 6.23 M rows = original `budynki` + BDOT10k columns. We keep **whole-house sales**: `nier_rodzaj='nieruchomoscGruntowaZabudowana'`, `is_residential=1`, `match_type='overlap'`, `match_overlap_frac ≥ 0.30`, `usable_area_est_m2` present, `bdot_floors ∈ [1,5]`, `tran_rodzaj_trans='wolnyRynek'`, full ownership. Per **transaction** (`tran_lokalny_id_iip`): usable area = Σ residential-building `usable_area_est_m2`; building floors = max; plot = `nier_pow_gruntu` (ha-corrected against footprint); price = property price (`nier`, else single-unit `tran`); representative point = centroid of the largest building.

### 1.3 `dzialki.gpkg` — land + marginal houses
MULTIPOLYGON, EPSG:2180, 9.87 M rows; four `nier_rodzaj` kinds (lokalowa, gruntowaZabudowana, gruntowaNiezabudowana, budynkowa). Used for:
- **Undeveloped residential-land price surface**: `gruntowaNiezabudowana` with residential zoning (`dzi_przezn_wmpzp LIKE %budownictwoMieszkaniowe%`) or `dzi_sposob_uzyt='gruntyZabudowaneIZurbanizowane'`, arm's-length, price `nier→dzi`, area from **polygon geometry** (m², avoids the ~9 % ha errors in `dzi_pow_ewid`).
- **Marginal rural house transactions**: `gruntowaZabudowana` priced sales **not already in the budynki house set**, joined to the building bridge for usable area; plot area from the parcel polygon.

### 1.4 `budynki_dzialki_bdot10k.gpkg` — building↔parcel bridge (v2)
POLYGON, 1.41 M rows: BDOT10k buildings mapped to parcels, with `dzi_id_dzialki`, `dzi_tran_lokalny_id_iip`, `usable_area_est_m2`, `is_residential`, `bdot_floors`. Provides residential usable area per parcel (summed across multiple buildings — 40 % of built-up parcels have >1 building) for the marginal route. 91 % of its buildings are shared with `budynki_bdot10k` (same BDOT register) — hence the transaction-id deduplication.

### 1.5 GUS `P3787` (median) / `P3788` (mean)
Powiat × market × size-class × year (2010–2024). Anchor = `median, total, ogółem, 2021`; mean and size/market cells are auxiliary/validation. Zeros = suppression → NaN. Code powiat = first 4 digits.

### 1.6 `communes_2021.gpkg` (EPSG:3035; 2,477 gminas, `JPT_KOD_JE`, `pop`) and `TERC_Urzedowy_2021` (gmina names, RODZ type). Communes are reprojected to 2180 for the join.

---

## 2. Micro construction (per source)

**Parsing.** All numeric fields are TEXT; parsed defensively (space-strip, decimal-comma → dot).

**Price.** Unit price = first non-null of the component/property/transaction fields, with the whole-transaction field used only for single-unit deals (avoids dividing a bundled price by one unit's area).

**Flats** (`build_flats`): filters `mieszkalna`, `wolnyRynek`, ownership; hard gates price>0, area∈[15,400], year∈[2006,2026], ppm2∈[1 000, 50 000]; dedup by (tran,unit) and (unit,date,price). Output `ppm2 = price/area`.

**Houses** (`build_houses_primary` + `build_houses_marginal`): assembled per transaction (§1.2, §1.3), deduplicated across the two routes by `tran_lokalny_id_iip` (budynki primary; dzialki adds only new ids — ~85 k marginal). Usable area is the summed residential BDOT10k estimate. `ppm2` is filled **after** land netting (§3).

**Land** (`build_land`): undeveloped residential-land points with `land_ppm2 = price / geometry_area`; gate [1, 20 000] zł/m²; ha-coded rows flagged via recorded/geometry ratio.

---

## 3. Spatial assignment, land surface, and land netting

1. **Gmina assignment.** Reproject communes 3035→2180; parallel within-join of flat/house/land points to gmina polygons; validate geometry-derived powiat against RCN `teryt`.
2. **Local land-price surface** (`build_land_surface`): hierarchical median undeveloped-land zł/m² per gmina, filled **gmina → powiat → voivodeship → national** so every gmina has a value. This doubles as the SAE `log_land` covariate.
3. **House land-netting** (`net_land`): the index must price *enclosed living space*, not garden/plot land, so
   `structure_price = P_total − min(p_land_local × plot_area, cap × P_total)`, `price/m² = structure_price / usable_area`,
   with `cap = 0.60` and plot imputed by powiat-median where missing. Methods: `subtract` (default), `regress` (keep P/usable, add log-plot as a house covariate), `none`. Post-netting gate ppm2 ∈ [500, 40 000].

Rationale: netting the plot at local raw-land prices removes the value of the plot as land while retaining the location premium embedded in the structure — comparable to a flat's price/m², and to the GUS flat anchor after the property-type differential.

---

## 4. Hedonic (national, pooled flats + houses)

```
log(price/m²)_i = α
   + f(log area)                      # size gradient (log area + quadratic)
   + β_house·1[house]                 # property-type level (ties houses to flats)
   + β_prim·1[primary] + β_unk·1[unknown]     # explicit market covariate (secondary = ref)
   + rooms (+miss)                    # flats
   + storey (+miss)                   # flat storey
   + bld_floors (+miss)               # BDOT10k building floors (houses)
   [+ log_plot_house  if netting='regress']
   + δ_t (year FE, δ_2021 ≡ 0)        # STATIC-2021 normalisation (all reference-year cells dropped)
   + γ_g (gmina FE)                   # quality-adjusted gmina level = direct estimate θ̂_g
   + ε_i
```
Controls are centred, so `γ_g` = expected log(price/m²) of a mean-characteristics dwelling in gmina g at 2021, with HC1 sampling variance `D_g`. Solved as a sparse two-way FE normal-equations system (no external FE library). Options: `--time-fe {year, woj_year}`, `--year-min/--year-max`. Diagnostics compare the δ_t path to the GUS national series.

---

## 4B. Data-quality audit — self-constructed BDOT10k / parcel data

Because usable area and parcel area are self-constructed, filters are applied **before** use and their attrition is logged (each builder reports rows in/out):

- **Match reliability**: keep `match_type='overlap'` (drop `nearest`: outbuildings, median footprint 24 m², 2.2× inflated price/m²); require `match_overlap_frac ≥ 0.30`.
- **Floors**: `bdot_floors ∈ [1,5]` — excludes the artefact spikes (5, 11 are BDOT default-fills that mostly vanish under the overlap filter) and tall buildings (≥6 → whole apartment blocks / mismatches, price/m² jumps to ~16 000).
- **Usable area**: ∈ [25, 1000] m²; robust within-stratum MAD trim on log price/m².
- **Parcel/plot area**: use **polygon geometry** (m²), not `dzi_pow_ewid`/`nier_pow_gruntu` (~9–13 % ha-coded); a plot smaller than its own footprint is ha-corrected (×10 000); land ratio flag drops residual ha rows.
- **Land netting guards**: land value capped at 60 % of price; post-netting price/m² bounds; source down-weighting (`qweight`: overlap-frac for primary houses, 0.7 for marginal).

The audit is not optional cosmetics: it is what makes the 456 k + 85 k self-constructed house observations trustworthy without being over-conservative (the overlap+floors gates retain ~0.5 M of ~0.66 M candidate house transactions).

---

## 5. Small-area model — hierarchical shrinkage to the GUS prior

Two-level Fay–Herriot (parametric empirical/hierarchical Bayes) on the gmina direct estimates:
```
θ̂_g = θ_g + e_g,  e_g ~ N(0, D_g);   θ_g = x_g'η + v_g,  v_g ~ N(0, A)
```
with `x_g` = { log GUS_median_powiat,2021 (prior mean, β≈1), RODZ gmina type, log gmina land price (dzialki), log population, gmina flat-share }. REML/moment estimation of (η, A); posterior `θ̃_g = x_g'η + A/(A+D_g)·(θ̂_g − x_g'η)`. Data-rich gminas ≈ direct estimate; thin gminas shrink to the GUS-anchored prediction; zero-RCN gminas take the prediction outright. `--bayes` runs full MCMC on the same area model. `--benchmark-to-gus` optionally rescales within-powiat weighted means to GUS exactly.

---

## 6. Outputs

`index_zl_m2 = exp(θ̃_g)` — 2021 quality-adjusted zł/m² of living space for all 2,477 gminas, with posterior SD, shrinkage weight (RCN vs GUS/model), n_flats / n_houses, GUS anchor, local land price + fallback level, and validation flags. Written as `data/processed/floorspace/gmina_floorspace_index_2021.{csv,parquet,gpkg}` plus a diagnostics JSON (hedonic coefficients incl. market/floors/house, year-FE path, source composition, land-level counts). GeoPackage writes are NFS-safe (local scratch → move).

---

## 7. Assumptions and risks

1. **Proportional national price growth** (year-FE pooling) — relaxed via `woj_year`/window; validated vs GUS.
2. **Flats-only GUS anchoring a flats+houses target** — absorbed by the property-type differential and gmina flat-share covariate.
3. **Self-constructed usable area** — `usable = footprint × floors × 0.73`; error controlled by overlap+floors+area gates and robust trimming; residual noise down-weighted, not trusted unconditionally.
4. **Land netting** — raw-land netting of the plot is a standard residual method; capped and imputation-guarded; `regress`/`none` alternatives provided for robustness.
5. **Sparsity remains the true limit** — rural gmina values are effectively GUS-powiat + covariate predictions; the shrinkage-weight and land-level columns make this explicit.

---

## 8. How to run

```
python scripts/floorspace/run_index.py --workers $(nproc)          # full run (HPC)
python scripts/floorspace/run_index.py --sample-frac 0.02          # fast smoke test
python scripts/floorspace/run_index.py --house-land-netting regress   # plot as covariate
python scripts/floorspace/run_index.py --house-max-floors 4 --no-dzialki-marginal
python scripts/floorspace/run_index.py --year-min 2016 --time-fe woj_year --benchmark-to-gus
```
All paths resolve from the repository layout. Key flags: `--house-land-netting {subtract,regress,none}`, `--house-max-floors`, `--house-use-nearest`, `--no-dzialki-marginal`, `--time-fe`, `--year-min/--year-max`, `--anchor {median,mean}`, `--benchmark-to-gus`, `--bayes`, `--sample-frac`.
```
```
