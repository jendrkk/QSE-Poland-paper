# Three-Epoch (2011 / 2021 / 2026) Gmina-Level Residential Floorspace Price Index

**Purpose.** Replace the single static 2021 RFPI vector — currently reused verbatim for every MRRH model year — with **three internally-consistent, full gmina vectors** `P_{g,2011}`, `P_{g,2021}`, `P_{g,2026}`, each carrying (i) the *correct price level* of its model year and (ii) an *era-appropriate within-powiat spatial pattern*, while degrading gracefully to a persistent pattern wherever the microdata are thin. This document is the design contract for `scripts/floorspace/multiyear.py` and the multi-year branch of `run_index.py`. It supersedes nothing in `METHODOLOGY_floorspace_index.md`; the static path is preserved unchanged.

---

## 0. The decomposition that governs everything

For a model year `t`, write the target in logs as

```
log P_{g,t}  =  L_{p(g),t}            (powiat-t price LEVEL, the temporal dimension)
             +  s_{g,t}               (within-powiat gmina DEVIATION, the spatial dimension)
```

with the identifying normalisation `Σ_{g∈p} w_{g,t} · s_{g,t} = 0` (housing-stock-weighted deviations sum to zero inside each powiat). The two terms have **completely different best data sources**, and this is the single most important fact for the redesign:

| Term | What it is | Best source | Availability at 2011 / 2021 / 2026 |
|---|---|---|---|
| `L_{p,t}` | powiat mean/median zł/m² in year `t` | **GUS P3787** (cleaned official RCN aggregate, flats) | **direct 2011 ✓, direct 2021 ✓, 2026 = 2024 + short extrapolation** |
| `s_{g,t}` | which gminas are dear/cheap relative to their powiat | **RCN hedonic** gmina FE | rich 2021/2026; thin-but-usable 2011 (see §2) |

The distortion you observed from reusing the 2021 vector is **overwhelmingly a level error**, not a pattern error: applying 2021 zł/m² to 2011 (roughly *half* the nominal level) and to 2026 (well above 2024) misprices every gmina by the national/powiat drift, which for PL 2011→2026 is the dominant moving part. Getting `L_{p,t}` right per year fixes ~80–90 % of the problem with essentially zero modelling risk, because **GUS already publishes the 2011 and 2021 powiat levels**. Letting `s_{g,t}` drift across epochs is a genuine but *second-order* refinement, worth doing only where the data support it.

This reframes your plan: **you do not need to extrapolate the 2011 level from a 2021→2026 trend at all** — GUS gives it to you directly. The RCN work for 2011 is only about recovering the *spatial texture*, which is a much smaller and better-posed problem.

**Empirical confirmation (from the actual files).** GUS `P3787` national powiat-median is **2 726 zł/m² in 2011**, **4 115 in 2021**, **5 884 in 2024**, with a 2011 median present for **357 of 380 powiats** — i.e. the 2011 level is directly and near-completely observed. Reusing the 2021 vector therefore *over*prices 2011 by ≈ 4115/2726 ≈ **1.51×** and *under*prices 2026 (vs 2024) by ≈ **1.4×+**. That level error, applied to every gmina, dwarfs any plausible drift in the within-powiat pattern — which is exactly why the redesign puts the level on GUS-per-year and treats the pattern as the smaller correction. Housing stock (`P2166`, types 1/2/3) reconciles to the national dwelling total **exactly** (13 587 440 in 2011) once the urban-rural 4/5 split rows are excluded.

---

## 1. Critical assessment of the proposed plan

Your instinct (batch by era, estimate within-powiat variation per batch, fill gaps with GUS, borrow across time for 2011) is directionally correct. Six specific points change or sharpen it.

**1.1 GUS price data is powiat-only; housing stock is gmina-level.** Verified code granularity of the five GUS files:

- `P3787` median, `P3788` mean, `P3783` count-sold, `P3785` area-sold → **1 national + 16 woj + 380 powiat rows, no gminas**, market × size-class × year **2010–2024**.
- `P2166` total housing stock → **national + woj + 380 powiat + 4 057 gmina rows**, dwellings (`mieszkania`) and rooms (`izby`), years **2010–2025**.

Consequence: GUS can never supply a *within-powiat* gmina price signal — that must come from RCN. GUS supplies the **level** (`P3787`, per powiat per year) and the **weights/denominator** (`P2166`, per gmina per year). Your "weight by total housing stock vs transaction counts" question resolves cleanly on this basis — see §4.

**1.2 The 2011 batch is thinner but NOT empty — and it is directly estimable.** Actual cleaned-eligible flat counts by year (residential, `wolnyRynek`, whole-ownership; pre-price-gate) from `lokale.gpkg`:

```
2006  7 953   2011 30 733   2016 55 041   2021 161 652
2007 12 703   2012 33 374   2017 68 171   2022 155 776
2008 16 969   2013 38 451   2018 78 439   2023 200 673
2009 20 871   2014 41 914   2019 77 998   2024 216 300
2010 26 699   2015 45 656   2020 86 885   2025 234 470   2026 80 396 (partial)
```

A 2006–2016 window holds **≈355 k** flats (≈230 k after price cleaning) — *more than enough* to estimate the national hedonic slopes and to give direct gmina effects for the several hundred data-rich (mostly urban) gminas around 2011. So the 2011 spatial pattern should be **anchored on real ~2011 microdata where it exists**, and only *shrunk toward the persistent/2021 pattern where it does not** — not blindly extrapolated from the 2021→2026 trend. Backward-extrapolating a 5-year (2021→2026) relative-pattern trend across a 10–15-year gap is the most fragile element of the original plan and is demoted to an optional robustness check.

**1.3 Separate per-batch regressions vs one set of characteristic prices.** Re-estimating the full hedonic on each batch is fine for 2021/2026 and acceptable for 2011 (≈230 k obs identify ~11 structural coefficients easily). The only real sparsity is at the *gmina* level, which the small-area model already handles. We therefore keep per-epoch regressions as the robust default, and add `--share-controls` to fix one common set of characteristic prices (size gradient, house dummy, market premium, floors) estimated on the pooled sample and reused per epoch — marginally more efficient and makes the three epochs strictly comparable in "quality-adjusted to the *same* reference dwelling" terms. Recommended for the production run.

**1.4 Non-`wolnyRynek` transactions: keep excluding them.** The registry's other `tran_rodzaj_trans` classes are non-arm's-length by construction: `sprzedazZBonifikata` (municipal sales to sitting tenants at **statutory 80–95 % discounts**), `sprzedazBezprzetargowa` (non-tender, discounted), `...WPostepowaniuEgzekucyjnym` (foreclosure). Folding these in — even with a transaction-type dummy — fails, because the discount is *large, heterogeneous (80–95 % is a 4× spread) and endogenous* to location and property unobservables, so a single additive dummy cannot net it out; it would bias exactly the within-powiat pattern we are trying to measure. The one borderline-defensible class is `sprzedazPrzetargowa` (open competitive tender, often at/above market); it is exposed as `--extra-trans przetargowa` with its own dummy for a **robustness check only**, not the main spec. Net verdict: the coverage gap at 2011 is far better closed by houses + a wider window + shrinkage than by admitting discounted sales.

**1.5 House↔parcel matching tolerance: yes, relax it — selectively and down-weighted.** The current gate (`match_type='overlap'` **and** `match_overlap_frac ≥ 0.30`, dropping all `nearest`) is right for the *level* but is over-conservative for *rural coverage*, which is exactly where the 2011 vector is weakest. Two safe relaxations, both **down-weighted, not trusted equally**:
  - lower the overlap floor to ~0.15–0.20 (recovers partially-overlapping footprints) with `qweight = overlap_frac`;
  - admit `nearest` matches **only** behind a footprint floor (≥ 40–50 m², which removes the 24 m² outbuilding artefacts that inflate price/m² 2.2×) and a max match distance, at `qweight ≈ 0.5`.

  **Prerequisite (latent gap in the current code):** `qweight` is computed and carried in `MICRO_COLS` but is **never used** — `fit_hedonic` runs unweighted OLS. Coverage-with-down-weighting is only meaningful once `qweight` enters as a regression weight. The rewrite adds optional WLS (`--wls`, default off to preserve current numbers). Exposed via `--house-min-overlap`, `--house-use-nearest`, `--house-nearest-min-footprint`, `--house-nearest-max-dist`.

**1.6 Other silently-excluded useful data.** Nothing large beyond the above. The flat gates (`area∈[15,400]`, `ppm2∈[1000,50000]`) and ownership handling are sound. `lok_pow_przyn` (accessory area, 36 %) is too sparse to require. The genuinely wasted signal is temporal: pooling everything into one 2021 vector *discards the entire cross-epoch variation* — which is what this redesign recovers.

---

## 2. The optimal approach (SOTA: time-varying local hedonic + per-epoch SAE + official per-year anchor)

For each epoch `τ ∈ {2011, 2021, 2026}` with micro window `W_τ` and target year `a_τ`:

**Stage A — characteristic prices.** Either re-fit the pooled hedonic on `micro ∩ W_τ` (default), or fix one common control vector from the full sample (`--share-controls`). Structural controls, dwelling types, land-netting, and robust trimming are **exactly as in the static methodology** (`§2–§4` there) — this redesign changes *what is pooled and how it is anchored*, not the micro construction.

**Stage B — direct within-epoch gmina effects.** Regress `log(ppm2)` on centred controls + **gmina FE** + **year FE (ref = a_τ)** on `W_τ`. The gmina FE is the direct estimate `θ̂_{g,τ}` = quality-adjusted log zł/m² of a mean-characteristics dwelling in gmina `g` at year `a_τ` *within the epoch's level normalisation*, with HC1 sampling variance `D_{g,τ}`. (Only gminas with ≥ `min_txn_direct` obs in `W_τ` get a direct estimate.)

**Stage C — per-year GUS anchor `L_{p,a_τ}`.**
  - 2011, 2021 → `P3787` powiat median for that exact year, `ogółem/ogółem` (imputed woj→national where a powiat cell is suppressed), as today but for the epoch year rather than fixed 2021.
  - 2026 (beyond GUS's 2024 horizon) → `L_{p,2026} = L_{p,2024} × ρ`, where `ρ` is the RCN national median-price growth 2024→2026 (robustly estimated from the cleaned micro, or overridable via `--gus-extrapolation`; optionally per-woj). Exposed and logged.

**Stage D — per-epoch Fay–Herriot shrinkage.** The existing area model, run once per epoch: prior mean `x_g'η` with `x_g = {log L_{p,a_τ}, log land price, log population, gmina type}`. **Cross-epoch borrowing for 2011:** after the 2021 vector is computed, its within-powiat deviation `ŝ_{g,2021} = θ̃_{g,2021} − (stock-weighted powiat mean)` is added as an **extra prior covariate** in the 2011 area model. Thin 2011 gminas then shrink toward *their own 2021 relative position* (loading ≈ 1 if the pattern is persistent, estimated by GLS), not merely toward the powiat mean; data-rich 2011 gminas keep their direct 2011 estimate. Zero-data gminas take the prediction outright. This makes the 2011 vector a principled interpolation between "real 2011 texture" and "2021 texture re-levelled to 2011," which is the best obtainable given the sparsity.

**Stage E — exact per-year level reconciliation.** Within each powiat, rescale so the **housing-stock-weighted** (P2166, year `a_τ`) mean of gmina levels equals `L_{p,a_τ}` (§4). This pins each vector to the correct official per-year level and makes `s_{g,τ}` a clean within-powiat object. Default **on** in multi-year mode (it is the mechanism that fixes the level distortion); `--no-benchmark` to disable.

**Output.** `P_{g,τ} = exp(θ̃_{g,τ})` for all 2 477 gminas × 3 epochs, plus posterior SD, shrinkage weight, source flag, n_flats/n_houses, GUS anchor, land level. Epochs are computed **in order 2021 → 2026 → 2011** (2011 consumes the 2021 pattern).

Why this is the right frontier: it is a **time-varying hedonic index with shared characteristic prices** (Sirmans/Hill hedonic-index practice), **per-period small-area estimation** (Fay–Herriot / empirical Bayes) with a **cross-period smoothness prior**, and **benchmarking to official statistics** (calibration/raking to GUS). It puts each dimension on the data that identify it — level on GUS's authoritative per-year powiat series (2011 included), spatial texture on RCN — instead of asking one 2021 cross-section to stand in for fifteen years of both.

---

## 3. Why not the alternatives

- **Pooled panel with gmina×epoch interaction terms in one regression** (`γ_g + λ_{g,τ}`): elegant and slightly more efficient, but a larger rewrite, heavier memory (≈2 477×3 extra sparse columns), and extracting per-epoch `D_{g,τ}` for the downstream SAE is fiddlier. The per-epoch-run design reuses the tested static machinery almost verbatim and is easier to validate. `--share-controls` recovers most of the efficiency gain.
- **Pure "reuse 2021 pattern, re-level to GUS per year"** (no per-epoch hedonic): this is exactly the robust *fallback* our design collapses to when an epoch has no microdata. We keep it available (it is what a zero-data 2011 gmina receives) but do better where data allow.
- **Backward trend extrapolation to 2011** (original plan): retained only as `--pattern-trend` robustness; not the default.

---

## 4. The weighting question, resolved

Two distinct roles:

1. **Level reconciliation (Stage E).** Target = GUS powiat level; we rescale the gmina vector so its *weighted mean* hits it. The MRRH model consumes a **stock** price (housing services flow from the standing stock, not the transaction flow), and a gmina's weight in its powiat's average should reflect *how much housing is there*, not *how much happened to trade*. ⇒ **weight by P2166 gmina housing stock of year `a_τ`** (default). Transaction-count weighting (RCN counts per gmina) is offered as `--benchmark-weight txn` for robustness / for reproducing GUS's transaction-median more literally.
2. **2011 powiat→gmina projection** for zero-data gminas: same P2166 stock weights, year 2011 — available and appropriate.

`P3783`/`P3785` (powiat count/area sold) are auxiliary: usable for size-composition reweighting of the anchor and for coverage diagnostics, not required for the core.

---

## 5. What changes in the code (additive; static path preserved)

- `config.py` — paths for `P2166/P3783/P3785`; `EPOCHS` default `{2011:(2006,2016), 2021:(2017,2022), 2026:(2023,2026)}` (contiguous partition; override with `--epoch-windows`); multi-year defaults.
- `gus.py` — `load_housing_stock()` (gmina×year dwellings from P2166); `anchor_for_year(long, year, extrapolation)` wrapping the existing `powiat_anchor` with the 2026 extrapolation.
- `estimate.py` — `fit_hedonic(..., fixed_controls=, weights=)` (shared characteristic prices + optional WLS); `within_powiat_pattern()`; `benchmark_to_gus(..., weights=, gus_by_powiat=)` generalised to any year & weight vector.
- `multiyear.py` — orchestrator: build pooled micro once → per-epoch Stages A–E → write per-epoch + combined outputs.
- `run_index.py` — `--multi-year`, `--model-years`, `--epoch-windows`, `--share-controls`, `--benchmark-weight`, `--gus-extrapolation`, `--wls`, house-tolerance flags; branch to `multiyear.run(...)` when `--multi-year`, else the existing static `main`.

Outputs: `gmina_floorspace_index_{2011,2021,2026}.{csv,parquet,gpkg}` + `gmina_floorspace_index_multiyear.csv` (wide: `index_2011/2021/2026`) + one diagnostics JSON per epoch.

---

## 6. Assumptions & residual risks

1. **Level dominates; GUS is authoritative per year.** If GUS powiat cells are suppressed (small powiats, early years) they impute woj→national — flagged per powiat/year.
2. **2026 level is extrapolated** two years beyond GUS via an RCN growth factor — the only non-official level input; logged and overridable.
3. **Spatial-pattern persistence** underlies 2011 borrowing; mitigated by using real 2011 micro where present and by the estimated (not imposed) loading on the 2021 pattern.
4. **Flats-only GUS anchoring a flats+houses target** — unchanged from static design; absorbed by the house dummy and gmina flat-share.
5. **Within-window pattern drift** — the 2011 window's mass centroid (~2013) sits after 2011; the level is re-fixed to GUS-2011 regardless, and the window is narrowable (`--epoch-windows`) if a purer 2011 texture is wanted at the cost of gmina coverage.
