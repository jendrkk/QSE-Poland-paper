#!/usr/bin/env python3
"""
multiyear.py
============

Three-epoch (2011 / 2021 / 2026) gmina-level residential floorspace price index.

Instead of one static 2021 vector reused for every MRRH model year, this module
produces a FULL gmina RFPI vector for each model year, decomposing

    log P_{g,t} = L_{p(g),t}   (powiat-t LEVEL, from GUS -- the temporal signal)
                + s_{g,t}       (within-powiat DEVIATION, from RCN -- the spatial signal)

Each dimension is estimated from the data that identify it:

* LEVEL  -> GUS P3787 powiat price of the *model year itself* (2011 and 2021 are
  published directly; 2026 is the 2024 cross-section grown by the RCN national
  trend). This alone removes the bulk of the distortion from reusing 2021 prices.
* PATTERN -> a per-epoch hedonic gmina fixed effect on the micro window for that
  era, shrunk (Fay-Herriot) toward the GUS-anchored prior; the data-poor 2011
  cross-section additionally borrows the 2021 within-powiat pattern as a prior.

The heavy micro construction (read + clean + gmina-assign + land-net) runs ONCE
(``run_index.build_pooled_micro``); only stages 4-7 repeat per epoch.

Design contract: ``scripts/floorspace/METHODOLOGY_multiyear_RFPI.md``.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

import config as C
import gus as gusmod
import teryt as tercmod
import estimate

LOGGER = logging.getLogger("floorspace.multiyear")


# --------------------------------------------------------------------------- #
# CLI helpers
# --------------------------------------------------------------------------- #
def _parse_model_years(args):
    if args.model_years:
        return [int(y) for y in str(args.model_years).replace(" ", "").split(",") if y]
    return list(C.MODEL_YEARS)


def _parse_epoch_windows(args, model_years):
    windows = dict(C.EPOCH_WINDOWS)
    if args.epoch_windows:
        for tok in str(args.epoch_windows).replace(" ", "").split(","):
            if not tok:
                continue
            yr, rng = tok.split(":")
            lo, hi = rng.split("-")
            windows[int(yr)] = (int(lo), int(hi))
    out = {}
    for my in model_years:
        if my in windows:
            out[my] = windows[my]
        else:
            raise SystemExit(f"No epoch window for model year {my}; pass --epoch-windows.")
    return out


def _solve_order(model_years):
    ordered = [y for y in C.EPOCH_SOLVE_ORDER if y in model_years]
    ordered += [y for y in model_years if y not in ordered]
    return ordered


def _rcn_growth_factor(micro, base_year, target_year, override):
    """Multiplicative RCN median-ppm2 factor base_year -> target_year (>=base).

    Used to grow the last GUS cross-section (2024) to a model year beyond the GUS
    horizon. ``override`` (if given) is returned verbatim.
    """
    if override is not None:
        return float(override)
    med = micro.groupby("year")["ppm2"].median()
    if base_year in med.index and target_year in med.index and med[base_year] > 0:
        return float(med[target_year] / med[base_year])
    # fall back to a compounded recent annual growth
    yrs = [y for y in med.index if base_year - 3 <= y <= base_year]
    if len(yrs) >= 2 and med[min(yrs)] > 0:
        g = (med[max(yrs)] / med[min(yrs)]) ** (1.0 / (max(yrs) - min(yrs)))
    else:
        g = 1.0
    factor = float(g ** max(target_year - base_year, 0))
    LOGGER.warning("RCN growth %d->%d not directly observed; compounded g=%.4f -> factor=%.4f",
                   base_year, target_year, g, factor)
    return factor


# --------------------------------------------------------------------------- #
# One epoch
# --------------------------------------------------------------------------- #
def _run_epoch(model_year, window, micro, communes_pop, communes, land_surface,
               gus_long, terc, stock, args, cfg, mcfg, fixed_controls, prior_pattern):
    lo, hi = window
    sub = micro[(micro["year"] >= lo) & (micro["year"] <= hi)].copy()
    ref = int(min(max(model_year, lo), hi))            # normalise inside window
    LOGGER.info("EPOCH %d: window %d-%d, %s micro rows, ref-year %d",
                model_year, lo, hi, f"{len(sub):,}", ref)

    # ---- hedonic: per-gmina direct estimate --------------------------------
    direct, hinfo = estimate.fit_hedonic(
        sub, time_fe=mcfg.time_fe, ridge=mcfg.ridge, ref_year=ref,
        plot_regressor=(cfg.house_land_netting == "regress"),
        fixed_controls=fixed_controls,
        weights=(sub["qweight"].to_numpy(float) if args.wls else None),
    )

    # ---- per-year GUS anchor (extrapolated beyond 2024) --------------------
    if model_year > C.GUS_LAST_YEAR:
        growth = _rcn_growth_factor(micro, C.GUS_LAST_YEAR, model_year, args.gus_extrapolation)
    else:
        growth = None
    anchor = gusmod.anchor_for_year(gus_long, model_year, market="total",
                                    last_year=C.GUS_LAST_YEAR, growth=growth)
    if args.anchor == "mean":
        anchor = anchor.assign(gus_median=anchor["gus_mean"])
    covariates = _build_covariates(communes_pop, land_surface, anchor, terc)

    # ---- cross-epoch pattern prior (2011 borrows 2021) ---------------------
    cov_cols = ["log_gus", "log_land", "log_pop", "urban", "urban_rural"]
    if prior_pattern is not None:
        covariates = covariates.merge(prior_pattern, on="gmina_teryt", how="left")
        covariates["prior_pattern"] = covariates["prior_pattern"].fillna(0.0)
        cov_cols = ["log_gus", "prior_pattern", "log_land", "log_pop", "urban", "urban_rural"]
        LOGGER.info("EPOCH %d: using %d-epoch within-powiat pattern as extra prior",
                    model_year, C.EPOCH_PATTERN_PRIOR.get(model_year))

    # ---- Fay-Herriot shrinkage --------------------------------------------
    index = estimate.fay_herriot(direct, covariates, cov_cols, anchor_col="log_gus")
    index = index.merge(covariates[["gmina_teryt", "powiat", "pop", "rodz_class", "gus_median"]],
                        on="gmina_teryt", how="left")
    index = index.merge(terc[["teryt7", "nazwa"]].rename(columns={"teryt7": "gmina_teryt"}),
                        on="gmina_teryt", how="left")
    src = (sub.assign(kind=np.where(sub["source"] == "flat", "n_flats", "n_houses"))
              .groupby(["gmina_teryt", "kind"]).size().unstack(fill_value=0).reset_index())
    index = index.merge(src, on="gmina_teryt", how="left")
    for c in ("n_flats", "n_houses"):
        if c in index:
            index[c] = index[c].fillna(0).astype(int)
    index = index.merge(land_surface[["gmina_teryt", "land_level", "land_ppm2"]],
                        on="gmina_teryt", how="left")

    # ---- level reconciliation (exact within-powiat benchmark to GUS) -------
    stock_w = index["gmina_teryt"].map(
        gusmod.housing_stock_weights(stock, model_year)).to_numpy(float)
    if args.benchmark_weight == "stock":
        bw = stock_w
    elif args.benchmark_weight == "txn":
        bw = index["n_obs"].to_numpy(float)
    else:
        bw = index["pop"].to_numpy(float)
    if not args.no_benchmark:
        gus_by_powiat = anchor.set_index("powiat")["gus_median"].to_dict()
        index = estimate.benchmark_to_gus(index, weights=bw, gus_by_powiat=gus_by_powiat)

    # within-powiat pattern (stock-weighted) to seed later epochs
    pat = estimate.within_powiat_pattern(index, weights=stock_w)

    diag = {
        "model_year": model_year, "window": [lo, hi], "ref_year": ref,
        "n_micro": int(hinfo["n"]),
        "n_flats": int((sub["source"] == "flat").sum()),
        "n_houses": int((sub["source"] != "flat").sum()),
        "n_gminas_direct": int(hinfo["n_gminas"]),
        "n_gminas_pure_gus": int((~index["has_rcn"]).sum()),
        "gus_growth_from_last_year": (None if growth is None else float(growth)),
        "hedonic_r2": hinfo["r2"], "hedonic_sigma2": hinfo["sigma2"],
        "control_coef": {k: float(v) for k, v in hinfo["control_coef"].items()},
        "time_ref": hinfo["time_ref"],
        "median_index_zl_m2": float(np.nanmedian(index["index_zl_m2"])),
        "median_shrinkage_to_data": float(np.nanmedian(index["shrinkage_to_data"])),
        "used_pattern_prior": prior_pattern is not None,
        "benchmark_weight": args.benchmark_weight,
        "benchmarked": (not args.no_benchmark),
    }
    return index, pat, diag


def _build_covariates(communes, land_cov, gus_anchor, terc):
    """Thin wrapper around run_index.build_gmina_covariates (imported lazily to
    avoid a circular import at module load)."""
    import run_index as R
    return R.build_gmina_covariates(communes, land_cov, gus_anchor, terc)


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def run(args, cfg, mcfg):
    import run_index as R
    t0 = time.time()
    model_years = _parse_model_years(args)
    windows = _parse_epoch_windows(args, model_years)
    LOGGER.info("MULTI-YEAR mode: model years %s, windows %s",
                model_years, {k: windows[k] for k in model_years})

    micro, communes, land_surface = R.build_pooled_micro(cfg, args.workers, args.sample_frac)
    communes_pop = communes.drop(columns="geometry")
    gus_long = gusmod.load_gus_long(C.GUS_MEDIAN_CSV, C.GUS_MEAN_CSV)
    terc = tercmod.load_terc(C.TERC_2021_CSV)
    stock = gusmod.load_housing_stock(C.GUS_HOUSING_STOCK_CSV)

    # optional shared characteristic prices (estimated once on the full sample)
    fixed_controls = None
    if args.share_controls:
        LOGGER.info("Estimating shared characteristic prices on the full sample")
        _, info0 = estimate.fit_hedonic(
            micro, time_fe=mcfg.time_fe, ridge=mcfg.ridge, ref_year=C.REFERENCE_YEAR,
            plot_regressor=(cfg.house_land_netting == "regress"),
            weights=(micro["qweight"].to_numpy(float) if args.wls else None),
        )
        fixed_controls = {"control_coef": info0["control_coef"],
                          "control_means": info0["control_means"]}

    outdir = args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)

    patterns = {}
    results = {}
    diags = {}
    for my in _solve_order(model_years):
        prior_src = C.EPOCH_PATTERN_PRIOR.get(my)
        prior_pattern = patterns.get(prior_src) if prior_src is not None else None
        index, pat, diag = _run_epoch(
            my, windows[my], micro, communes_pop, communes, land_surface,
            gus_long, terc, stock, args, cfg, mcfg, fixed_controls, prior_pattern)
        patterns[my] = pat
        results[my] = index
        diags[my] = diag
        _write_epoch(index, communes, my, outdir, args)
        LOGGER.info("EPOCH %d done: median index=%.0f zl/m2 (%d RCN, %d pure GUS)",
                    my, np.nanmedian(index["index_zl_m2"]),
                    int(index["has_rcn"].sum()), int((~index["has_rcn"]).sum()))

    _write_combined(results, model_years, outdir)
    (outdir / f"{C.MULTIYEAR_COMBINED_STEM}_diagnostics.json").write_text(
        json.dumps({str(k): diags[k] for k in model_years}, indent=2, ensure_ascii=False))
    LOGGER.info("MULTI-YEAR DONE: %d vectors in %.1fs", len(model_years), time.time() - t0)
    print(str(outdir / f"{C.MULTIYEAR_COMBINED_STEM}.csv"))
    return 0


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def _write_epoch(index, communes, model_year, outdir: Path, args):
    import run_index as R
    ordered = [
        "gmina_teryt", "nazwa", "powiat", "rodz_class", "index_zl_m2",
        "theta_tilde", "theta_sd", "shrinkage_to_data", "has_rcn", "n_obs",
        "n_flats", "n_houses", "gus_median", "land_ppm2", "land_level", "pop",
    ]
    ordered = [c for c in ordered if c in index.columns]
    tab = index[ordered].sort_values("gmina_teryt")
    stem = f"{C.MULTIYEAR_OUTPUT_PREFIX}_{model_year}"
    tab.to_csv(outdir / f"{stem}.csv", index=False)
    try:
        tab.to_parquet(outdir / f"{stem}.parquet", index=False)
    except Exception as e:
        LOGGER.warning("parquet write skipped (%s)", e)
    gdf = communes.merge(tab, on="gmina_teryt", how="left")
    R.write_gpkg_nfs_safe(gdf, outdir / f"{stem}.gpkg", tmp_dir=args.tmp_dir)


def _write_combined(results, model_years, outdir: Path):
    base = None
    for my in model_years:
        col = f"index_{my}"
        sub = results[my][["gmina_teryt", "index_zl_m2"]].rename(columns={"index_zl_m2": col})
        base = sub if base is None else base.merge(sub, on="gmina_teryt", how="outer")
    # attach names/powiat from the last epoch for readability
    meta = results[model_years[-1]][["gmina_teryt", "nazwa", "powiat", "rodz_class"]]
    base = meta.merge(base, on="gmina_teryt", how="right").sort_values("gmina_teryt")
    base.to_csv(outdir / f"{C.MULTIYEAR_COMBINED_STEM}.csv", index=False)
    try:
        base.to_parquet(outdir / f"{C.MULTIYEAR_COMBINED_STEM}.parquet", index=False)
    except Exception as e:
        LOGGER.warning("combined parquet write skipped (%s)", e)
