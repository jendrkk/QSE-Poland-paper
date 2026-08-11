#!/usr/bin/env python3
"""
run_index.py
============

Centralised runner for the static (2021) gmina-level hedonic residential
floorspace price index.

Pipeline
--------
1.  Read + clean RCN micro data (flats from ``lokale``, houses from ``budynki``)
    and undeveloped residential land (``dzialki``)                    [rcn.py]
2.  Assign every transaction to a 2021 gmina by spatial join           [spatial.py]
3.  Build gmina covariates: population, gmina type, land price, and the
    GUS powiat 2021 anchor                                            [gus/teryt]
4.  National hedonic with gmina + year FE -> per-gmina direct estimate  [estimate.py]
5.  Fay-Herriot small-area model: shrink toward the GUS-anchored prior  [estimate.py]
6.  (optional) exact within-powiat benchmark to GUS
7.  Write the index (parquet + csv + gpkg) and a diagnostics bundle.

Design and data provenance are documented in
``scripts/floorspace/METHODOLOGY_floorspace_index.md``.

Usage
-----
    python scripts/floorspace/run_index.py --workers $(nproc)
    python scripts/floorspace/run_index.py --sample-frac 0.02      # smoke test
    python scripts/floorspace/run_index.py --year-min 2016 --time-fe woj_year
    python scripts/floorspace/run_index.py --benchmark-to-gus --bayes

All paths default to the QSE_Poland_paper repository layout.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

import config as C
from config import CleaningConfig, ModelConfig
import rcn
import gus as gusmod
import teryt as tercmod
import spatial
import estimate

LOGGER = logging.getLogger("floorspace.run")


# --------------------------------------------------------------------------- #
# Covariate assembly
# --------------------------------------------------------------------------- #
def build_gmina_covariates(communes, land_cov, gus_anchor, terc):
    """One row per gmina (all 2,477) with model covariates.

    Columns produced: gmina_teryt, powiat, pop, log_pop, rodz_class,
    urban, urban_rural, log_land, gus_median, log_gus.
    """
    base = pd.DataFrame({"gmina_teryt": communes["gmina_teryt"].astype(str)})
    if "pop" in communes:
        base["pop"] = communes["pop"].to_numpy()
    else:
        base["pop"] = np.nan
    base["powiat"] = base["gmina_teryt"].str[:4]

    # gmina type
    base["rodz_class"] = base["gmina_teryt"].map(tercmod.rodz_class_from_code)
    base["urban"] = (base["rodz_class"] == "urban").astype(float)
    base["urban_rural"] = (base["rodz_class"] == "urban_rural").astype(float)

    # land price covariate
    base = base.merge(land_cov[["gmina_teryt", "log_land"]], on="gmina_teryt", how="left")

    # GUS powiat anchor
    base = base.merge(gus_anchor.rename(columns={"powiat": "powiat"}), on="powiat", how="left")

    # transforms + imputation of the GUS anchor (voivodeship then national median)
    base["log_pop"] = np.log(np.clip(base["pop"].to_numpy(float), 1, None))
    base["woj"] = base["gmina_teryt"].str[:2]
    gm = base["gus_median"].astype(float)
    woj_med = base.groupby("woj")["gus_median"].transform("median")
    gm = gm.fillna(woj_med).fillna(base["gus_median"].median())
    base["gus_median_imp"] = gm
    base["log_gus"] = np.log(base["gus_median_imp"])
    # land imputation: powiat then national median of log_land
    ll = base["log_land"]
    base["log_land"] = ll.fillna(base.groupby("powiat")["log_land"].transform("median")).fillna(ll.median())
    LOGGER.info("Gmina covariates assembled: %d gminas (GUS anchor present for %d powiats before impute)",
                len(base), gus_anchor["gus_median"].notna().sum())
    return base


# --------------------------------------------------------------------------- #
# NFS-safe GeoPackage writing (mirrors commune_centroids.py)
# --------------------------------------------------------------------------- #
def write_gpkg_nfs_safe(gdf, output: Path, layer="floorspace_index", tmp_dir=None):
    output.parent.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix="floorspace_", dir=str(tmp_dir) if tmp_dir else None))
    work = scratch / output.name
    try:
        gdf.to_file(work, driver="GPKG", layer=layer)
        if output.exists():
            output.unlink()
        shutil.move(str(work), str(output))
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    p.add_argument("--year-min", type=int, default=CleaningConfig.year_min)
    p.add_argument("--year-max", type=int, default=CleaningConfig.year_max)
    p.add_argument("--time-fe", choices=["year", "woj_year"], default=ModelConfig.time_fe)
    p.add_argument("--anchor", choices=["median", "mean"], default=ModelConfig.anchor,
                   help="GUS statistic used as the prior mean.")
    p.add_argument("--drop-public-sellers", action="store_true")
    p.add_argument("--benchmark-to-gus", action="store_true",
                   help="Exact within-powiat rescale so weighted gmina mean = GUS.")
    p.add_argument("--bayes", action="store_true", help="Full MCMC area model (needs numpyro).")
    p.add_argument("--sample-frac", type=float, default=None,
                   help="Random fraction of transactions for a fast end-to-end test.")
    p.add_argument("--output-dir", type=Path, default=C.OUTPUT_DIR)
    p.add_argument("--tmp-dir", type=Path, default=None, help="Local scratch for GPKG (NFS).")
    p.add_argument("--log-file", type=Path, default=None)
    return p.parse_args(argv)


def setup_logging(log_file):
    handlers = [logging.StreamHandler(sys.stderr)]
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
    )


def main(argv=None):
    args = parse_args(argv)
    setup_logging(args.log_file)
    t0 = time.time()

    cfg = CleaningConfig(
        year_min=args.year_min, year_max=args.year_max,
        drop_public_sellers=args.drop_public_sellers,
    )
    mcfg = ModelConfig(
        time_fe=args.time_fe, bayes=args.bayes,
        benchmark_to_gus=args.benchmark_to_gus, anchor=args.anchor,
    )

    # ---- 1. micro + land ------------------------------------------------- #
    LOGGER.info("STAGE 1  reading + cleaning RCN micro data")
    micro = rcn.build_micro(cfg, workers=args.workers, sample_frac=args.sample_frac)
    land = rcn.build_land(cfg, workers=args.workers, sample_frac=args.sample_frac)

    # ---- 2. spatial join ------------------------------------------------- #
    LOGGER.info("STAGE 2  spatial assignment to 2021 gminas")
    communes = spatial.load_communes(C.COMMUNES_GPKG)
    micro = spatial.assign_gmina(micro, communes, workers=args.workers)
    micro = micro[micro["gmina_teryt"].notna()].copy()
    land_cov = spatial.gmina_land_covariate(land, communes, workers=args.workers)

    # ---- 3. covariates --------------------------------------------------- #
    LOGGER.info("STAGE 3  GUS anchor + gmina covariates")
    gus_long = gusmod.load_gus_long(C.GUS_MEDIAN_CSV, C.GUS_MEAN_CSV)
    gus_anchor = gusmod.powiat_anchor(gus_long, C.REFERENCE_YEAR)
    if args.anchor == "mean":
        gus_anchor = gus_anchor.assign(gus_median=gus_anchor["gus_mean"])
    terc = tercmod.load_terc(C.TERC_2021_CSV)
    communes_pop = communes.drop(columns="geometry")
    covariates = build_gmina_covariates(communes_pop, land_cov, gus_anchor, terc)

    # ---- 4. hedonic ------------------------------------------------------ #
    LOGGER.info("STAGE 4  national hedonic (gmina + %s FE)", args.time_fe)
    direct, hinfo = estimate.fit_hedonic(
        micro, time_fe=args.time_fe, ridge=mcfg.ridge, ref_year=C.REFERENCE_YEAR
    )

    # ---- 5. Fay-Herriot -------------------------------------------------- #
    LOGGER.info("STAGE 5  Fay-Herriot shrinkage to GUS prior")
    cov_cols = ["log_gus", "log_land", "log_pop", "urban", "urban_rural"]
    index = estimate.fay_herriot(direct, covariates, cov_cols, anchor_col="log_gus")
    index = index.merge(covariates[["gmina_teryt", "powiat", "pop", "rodz_class", "gus_median"]],
                        on="gmina_teryt", how="left")
    index = index.merge(terc[["teryt7", "nazwa"]].rename(columns={"teryt7": "gmina_teryt"}),
                        on="gmina_teryt", how="left")

    # ---- 6. optional benchmark ------------------------------------------ #
    if args.benchmark_to_gus:
        LOGGER.info("STAGE 6  exact within-powiat benchmark to GUS")
        index = estimate.benchmark_to_gus(index, gus_anchor, weight_col="pop")

    # ---- 7. outputs ------------------------------------------------------ #
    LOGGER.info("STAGE 7  writing outputs")
    outdir = args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)
    ordered = [
        "gmina_teryt", "nazwa", "powiat", "rodz_class", "index_zl_m2",
        "theta_tilde", "theta_sd", "shrinkage_to_data", "has_rcn", "n_obs",
        "gus_median", "pop",
    ]
    ordered = [c for c in ordered if c in index.columns]
    tab = index[ordered].sort_values("gmina_teryt")
    csv_path = outdir / f"{C.OUTPUT_STEM}.csv"
    pq_path = outdir / f"{C.OUTPUT_STEM}.parquet"
    tab.to_csv(csv_path, index=False)
    try:
        tab.to_parquet(pq_path, index=False)
    except Exception as e:  # pyarrow may be absent
        LOGGER.warning("parquet write skipped (%s)", e)

    # gpkg with geometry
    gdf = communes.merge(tab, on="gmina_teryt", how="left")
    gpkg_path = outdir / f"{C.OUTPUT_STEM}.gpkg"
    write_gpkg_nfs_safe(gdf, gpkg_path, tmp_dir=args.tmp_dir)

    # diagnostics
    diag = {
        "n_micro": int(hinfo["n"]),
        "n_gminas_direct": int(hinfo["n_gminas"]),
        "n_gminas_total": int(len(index)),
        "n_gminas_pure_gus": int((~index["has_rcn"]).sum()),
        "hedonic_r2": hinfo["r2"],
        "hedonic_sigma2": hinfo["sigma2"],
        "control_coef": {k: float(v) for k, v in hinfo["control_coef"].items()},
        "time_ref": hinfo["time_ref"],
        "time_coef": {k: float(v) for k, v in hinfo["time_coef"].items()},
        "median_shrinkage_to_data": float(np.nanmedian(index["shrinkage_to_data"])),
        "index_median_zl_m2": float(np.nanmedian(index["index_zl_m2"])),
        "config": {"year_min": cfg.year_min, "year_max": cfg.year_max,
                    "time_fe": mcfg.time_fe, "anchor": mcfg.anchor,
                    "benchmark_to_gus": mcfg.benchmark_to_gus,
                    "sample_frac": args.sample_frac},
    }
    (outdir / f"{C.OUTPUT_STEM}_diagnostics.json").write_text(json.dumps(diag, indent=2, ensure_ascii=False))

    LOGGER.info("DONE: %d gminas (%d from RCN, %d pure GUS/model). index median=%.0f zl/m2. %.1fs total.",
                len(index), int(index["has_rcn"].sum()), int((~index["has_rcn"]).sum()),
                np.nanmedian(index["index_zl_m2"]), time.time() - t0)
    print(str(csv_path))
    print(str(gpkg_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
