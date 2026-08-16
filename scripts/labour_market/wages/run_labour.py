#!/usr/bin/env python3
"""
run_labour.py
=============

End-to-end builder for the gmina-level labour-market inputs to the MRRH model:
mean wages and employment counts by place of **work** and place of
**residence**, harmonised onto the 2021 TERYT gmina frame, for three
cross-sections (2011, 2021, recent ~2026).

Outputs (``data/processed/labour_market/``)
    labor_tidy_2011.csv, labor_tidy_2021.csv, labor_tidy_2026.csv
        region_id, teryt7, nazwa, powiat, rodz_class,
        median_income_workplace, median_income_residence,
        employment_workplace, employment_residence, + provenance flags
    teryt_gmina_crosswalk_2021.csv       the shared 2021-anchored crosswalk
    labor_build_diagnostics.json         run diagnostics + validation report

Method: ``METHODOLOGY_labour_market_wages.md``. Structure mirrors
``scripts/floorspace/run_index.py`` (STAGE logging, config dataclasses, CLI).

Usage
-----
    python scripts/labour_market/wages/run_labour.py
    python scripts/labour_market/wages/run_labour.py --recent-window 6
    python scripts/labour_market/wages/run_labour.py --benchmark-recent --ostrowice-split
    python scripts/labour_market/wages/run_labour.py --residence-emp-method workage_rake
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as C
from config import LabourConfig, ModelConfig
import teryt
import gus
import flows
import estimate

LOGGER = logging.getLogger("labour.run")


# --------------------------------------------------------------------------- #
# Covariate assembly
# --------------------------------------------------------------------------- #
def covariate_frame(emp_work_harm, ref21):
    """[code6, emp_work, log_emp_work, urban, urban_rural] for all 2021 gminas."""
    base = ref21[["code6", "rodz_class"]].copy()
    base = base.merge(emp_work_harm.rename(columns={"emp": "emp_work"}), on="code6", how="left")
    base["emp_work"] = estimate.impute_hierarchical(base, "emp_work").clip(lower=1.0)
    base["log_emp_work"] = np.log(base["emp_work"])
    base["urban"] = (base["rodz_class"] == "urban").astype(float)
    base["urban_rural"] = (base["rodz_class"] == "urban_rural").astype(float)
    return base


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--recent-window", type=int, default=C.RECENT_WINDOW_MONTHS,
                   help="Trailing months averaged for the recent cross-section.")
    p.add_argument("--struct-shrinkage", type=float, default=LabourConfig.struct_shrinkage,
                   help="Shrinkage (0..1) on the covariate-predicted within-powiat wage deviation.")
    p.add_argument("--benchmark-recent", action="store_true",
                   help="Rake the recent gmina wage to the P2497 powiat anchor "
                        "(default off: P4609 is gmina-level truth).")
    p.add_argument("--residence-emp-method", choices=["flows_then_rake", "workage_rake", "off"],
                   default=LabourConfig.residence_emp_method)
    p.add_argument("--ostrowice-split", action="store_true",
                   help="Split dissolved Ostrowice across two absorbers instead of primary-only.")
    p.add_argument("--output-dir", type=Path, default=C.OUTPUT_DIR)
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
        datefmt="%H:%M:%S", handlers=handlers,
    )


def _find(glob_pat):
    hits = sorted(glob.glob(str(C.LM / glob_pat)))
    return Path(hits[0]) if hits else None


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv=None):
    args = parse_args(argv)
    setup_logging(args.log_file)
    t0 = time.time()
    cfg = LabourConfig(recent_window_months=args.recent_window,
                       struct_shrinkage=args.struct_shrinkage,
                       benchmark_recent_to_powiat=args.benchmark_recent,
                       residence_emp_method=args.residence_emp_method,
                       ostrowice_split=args.ostrowice_split)
    mcfg = ModelConfig()

    # ---- 1. crosswalk ---------------------------------------------------- #
    LOGGER.info("STAGE 1  building 2021-anchored TERYT crosswalk")
    universal, labelled, ref21 = teryt.build_crosswalk(cfg, C)
    ref21 = ref21.assign(teryt7=ref21["code6"] + ref21["rodz"].astype(str))
    FRAME = ref21["code6"]

    # ---- 2. parse raw sources ------------------------------------------- #
    LOGGER.info("STAGE 2  parsing GUS sources")
    wage_pow = gus.load_wage_powiat_yearly(C.P2497_WAGE_POWIAT)          # [powiat, year, wage]
    wage_gm = gus.load_wage_gmina_monthly(C.P4609_WAGE_GMINA)            # [code6, concept, y, m, wage]
    emp2172 = gus.load_emp_work_yearly(C.P2172_EMP_WORK)                 # [code6, year, emp]
    emp4508 = gus.load_emp_work_yearly(C.P4508_EMP_WORK)
    emp_res_m = gus.load_emp_residence_monthly(C.P4280_EMP_RES)          # [code6, y, m, emp]
    workage = {2011: gus.load_working_age(C.P3457_WORKAGE_2011, 2011),
               2021: gus.load_working_age(C.P4362_WORKAGE_2021, 2021)}
    census = {}
    for yr, pat in [(2011, C.CENSUS_INCOME_2011_GLOB), (2021, C.CENSUS_INCOME_2021_GLOB)]:
        f = _find(pat)
        census[yr] = gus.load_census_income_powiat(f) if f else None
        LOGGER.info("census income control %d: %s", yr, f.name if f else "NOT FOUND (uncalibrated)")

    def emp_work_year(y):
        # temporal within-gmina interpolation fills GUS-suppressed year cells
        src = emp2172 if y <= 2021 else emp4508
        yy = y if y <= 2021 else 2025
        piv = src.pivot_table(index="code6", columns="year", values="emp")
        piv = piv.interpolate(axis=1, limit_direction="both")
        s = (piv[yy] if yy in piv.columns else piv.iloc[:, -1]).rename("emp").reset_index()
        return teryt.harmonise(s, "code6", "emp", universal, how="sum")

    # ---- 3. recent cross-section + transfer model ----------------------- #
    LOGGER.info("STAGE 3  recent cross-section + within-powiat transfer model")
    end, win = cfg.recent_end, cfg.recent_window_months
    emp_work_recent_raw = emp4508[emp4508["year"] == 2025][["code6", "emp"]]

    def build_recent_wage(concept):
        """Recent gmina wage on the 2021 frame with provenance + imputation."""
        rw = estimate.recent_wage(wage_gm, concept, end, win)          # [code6, wage, wsrc]
        rw = rw.merge(emp_work_recent_raw, on="code6", how="left")
        rw["emp"] = rw["emp"].fillna(1.0)
        h = teryt.harmonise(rw, "code6", "wage", universal, how="wmean", weight_col="emp")
        h = ref21[["code6"]].merge(h, on="code6", how="left")           # full frame
        src_map = dict(zip(rw["code6"], rw["wsrc"]))
        h["wsrc"] = h["code6"].map(src_map).fillna("missing")
        miss = h["wage"].isna()
        if miss.any():
            h["wage"] = estimate.impute_hierarchical(h, "wage")
            h.loc[miss, "wsrc"] = "hier_impute"
        return h

    rec_w_work_h = build_recent_wage("workplace")
    rec_w_res_h = build_recent_wage("residence")
    recent_wsrc = dict(zip(rec_w_work_h["code6"], rec_w_work_h["wsrc"]))
    cov_recent = covariate_frame(emp_work_year(2026), ref21)
    # fit the transfer model only on genuinely observed gminas
    obs = rec_w_work_h[rec_w_work_h["wsrc"].isin(["window", "own_history"])][["code6", "wage"]]
    fit_df = cov_recent.merge(obs, on="code6", how="inner")
    beta, resid_sd = estimate.fit_transfer_model(fit_df, mcfg.covariates, ridge=cfg.ridge)

    # ---- 4. per-year build ---------------------------------------------- #
    LOGGER.info("STAGE 4  per-year assembly")
    outputs = {}
    for y in C.TARGET_YEARS:
        ew = emp_work_year(y)                                            # [code6, emp]
        cov = covariate_frame(ew, ref21)

        if y == C.RECENT_LABEL_YEAR:
            wage_work = (rec_w_work_h.rename(columns={"wage": "wage_hat"})[["code6", "wage_hat"]]
                         .assign(has_direct=True))
            wage_src = "P4609_direct"
            if cfg.benchmark_recent_to_powiat:
                pw = wage_pow[wage_pow["year"] == 2025].set_index("powiat")["wage"]
                ww = wage_work.merge(ew, on="code6", how="left")
                ww["powiat"] = ww["code6"].str[:4]
                ww["emp"] = ww["emp"].fillna(1.0).clip(lower=1.0)
                num = (ww["wage_hat"] * ww["emp"]).groupby(ww["powiat"]).transform("sum")
                den = ww["emp"].groupby(ww["powiat"]).transform("sum")
                anchor = ww["powiat"].map(pw)
                factor = (anchor / (num / den)).where(anchor.notna(), 1.0)
                ww["wage_hat"] = ww["wage_hat"] * factor
                wage_work = ww[["code6", "wage_hat", "has_direct"]]
                wage_src = "P4609_benchmarked_P2497"
            wage_res = rec_w_res_h[["code6", "wage"]].rename(columns={"wage": "med_res"})
        else:
            pw = wage_pow[wage_pow["year"] == y][["powiat", "wage"]]
            gcov = cov.copy()
            wage_work = estimate.disaggregate_wage(
                y, pw, gcov, beta, mcfg.covariates, "emp_work",
                cfg.struct_shrinkage, cfg.impute_hierarchy)
            wage_res = pd.DataFrame({"code6": FRAME, "med_res": np.nan})
            wage_src = "P2497_disaggregated"

        # residence employment
        if cfg.residence_emp_method == "off":
            emp_res = pd.DataFrame({"code6": FRAME, "emp_res": np.nan, "res_source": "off"})
        elif y == C.RECENT_LABEL_YEAR:
            # residence employment recent: trailing-window mean of P4280
            rr = emp_res_m.copy()
            rr["idx"] = rr["year"] * 12 + (rr["month"] - 1)
            ey, em = end
            hi = ey * 12 + (em - 1)
            rr = rr[(rr["idx"] >= hi - (win - 1)) & (rr["idx"] <= hi)]
            rr = rr.groupby("code6", as_index=False)["emp"].mean()
            emp_res = teryt.harmonise(rr, "code6", "emp", universal, how="sum") \
                .rename(columns={"emp": "emp_res"}).assign(res_source="P4280_direct")
        else:
            fm = flows.load_flow_margins(C.FLOWS_2011 if y == 2011 else C.FLOWS_2021, y, universal)
            wa = teryt.harmonise(workage[y], "code6", "workage", universal, how="sum")
            emp_res = estimate.residence_employment(
                ew, fm, wa, census[y], cfg.impute_hierarchy, method=cfg.residence_emp_method)

        # assemble tidy table
        tab = ref21[["code6", "teryt7", "nazwa", "powiat", "rodz_class"]].copy()
        tab = tab.merge(wage_work.rename(columns={"wage_hat": "median_income_workplace"}),
                        on="code6", how="left")
        tab = tab.merge(wage_res.rename(columns={"med_res": "median_income_residence"}),
                        on="code6", how="left")
        # employment_workplace = imputed emp_work (same series the rake weights use,
        # so the benchmark holds exactly); flag the imputed cells.
        raw = ew.set_index("code6")["emp"]
        tab["emp_work_imputed"] = ~tab["code6"].map(raw).notna()
        tab = tab.merge(cov[["code6", "emp_work"]].rename(columns={"emp_work": "employment_workplace"}),
                        on="code6", how="left")
        tab = tab.merge(emp_res[["code6", "emp_res", "res_source"]]
                        .rename(columns={"emp_res": "employment_residence"}),
                        on="code6", how="left")
        if y == C.RECENT_LABEL_YEAR:
            tab["wage_workplace_source"] = "P4609_" + tab["code6"].map(recent_wsrc).fillna("direct")
        else:
            tab["wage_workplace_source"] = wage_src
        tab = tab.rename(columns={"code6": "region_id"})
        outputs[y] = tab

    # ---- 5. validation -------------------------------------------------- #
    LOGGER.info("STAGE 5  validation gates")
    report = validate(outputs, wage_pow, ref21)

    # ---- 6. write ------------------------------------------------------- #
    LOGGER.info("STAGE 6  writing outputs")
    outdir = args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)
    written = []
    cols = ["region_id", "teryt7", "nazwa", "powiat", "rodz_class",
            "median_income_workplace", "median_income_residence",
            "employment_workplace", "employment_residence",
            "wage_workplace_source", "res_source", "emp_work_imputed"]
    for y, tab in outputs.items():
        path = outdir / f"{C.OUTPUT_STEM}_{y}.csv"
        tab[cols].sort_values("region_id").to_csv(path, index=False)
        written.append(str(path))
    cw_path = outdir / f"{C.CROSSWALK_STEM}.csv"
    labelled.sort_values(["src_year", "src_code"]).to_csv(cw_path, index=False)
    written.append(str(cw_path))

    diag = {
        "target_frame_gminas": int(len(ref21)),
        "recent_window_months": win, "recent_end": list(end),
        "transfer_beta": {k: float(v) for k, v in beta.items()},
        "transfer_resid_sd": resid_sd,
        "census_control_present": {str(k): (v is not None) for k, v in census.items()},
        "validation": report,
        "config": {"struct_shrinkage": cfg.struct_shrinkage,
                   "benchmark_recent": cfg.benchmark_recent_to_powiat,
                   "residence_emp_method": cfg.residence_emp_method,
                   "ostrowice_split": cfg.ostrowice_split},
    }
    (outdir / "labor_build_diagnostics.json").write_text(
        json.dumps(diag, indent=2, ensure_ascii=False))

    LOGGER.info("DONE: %d gminas x %d years. validation_passed=%s. %.1fs.",
                len(ref21), len(outputs), report["all_passed"], time.time() - t0)
    for w in written:
        print(w)
    return 0 if report["all_passed"] else 2


# --------------------------------------------------------------------------- #
# Validation gates
# --------------------------------------------------------------------------- #
def validate(outputs, wage_pow, ref21):
    checks = {}
    N = len(ref21)
    for y, tab in outputs.items():
        rep = {}
        rep["n_gminas"] = int(len(tab))
        rep["frame_complete"] = bool(len(tab) == N and tab["region_id"].is_unique)
        rep["workplace_wage_no_missing"] = int(tab["median_income_workplace"].isna().sum())
        rep["workplace_emp_no_missing"] = int(tab["employment_workplace"].isna().sum())
        # within-powiat weighted mean reproduces P2497 (disaggregated years)
        if tab["wage_workplace_source"].iloc[0].startswith("P2497"):
            pw = wage_pow[wage_pow["year"] == y].set_index("powiat")["wage"]
            g = tab.dropna(subset=["median_income_workplace", "employment_workplace"]).copy()
            wm = (g.assign(num=g["median_income_workplace"] * g["employment_workplace"])
                    .groupby("powiat").agg(num=("num", "sum"), den=("employment_workplace", "sum")))
            wm["recovered"] = wm["num"] / wm["den"]
            wm["anchor"] = wm.index.map(pw)
            wm = wm.dropna(subset=["anchor"])
            rel = np.abs(wm["recovered"] / wm["anchor"] - 1.0)
            rep["benchmark_max_rel_err"] = float(rel.max())
            rep["benchmark_ok"] = bool(rel.max() < 1e-6)
        checks[str(y)] = rep
    all_passed = all(
        r["frame_complete"] and r["workplace_wage_no_missing"] == 0
        and r.get("benchmark_ok", True)
        for r in checks.values())
    checks["all_passed"] = bool(all_passed)
    for y, r in checks.items():
        if y != "all_passed":
            LOGGER.info("  validate %s: complete=%s wage_missing=%s benchmark_ok=%s",
                        y, r["frame_complete"], r["workplace_wage_no_missing"],
                        r.get("benchmark_ok", "n/a"))
    return checks


if __name__ == "__main__":
    raise SystemExit(main())
