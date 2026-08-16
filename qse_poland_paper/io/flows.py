"""
io/flows.py — assemble the bilateral commuting matrix comMat[res, work].

The raw GUS matrices (2011 tax register .xls; 2021 census .xlsx) are
**off-diagonal only** and expressed in the source-year TERYT vintage, using the
split RODZ 4/5 (miasto / obszar wiejski) codes for urban-rural gminas. Two
harmonisations are therefore required before the matrix is usable:

1. TERYT vintage  -> 2021 frame, via the universal 6-digit crosswalk
   (``teryt_gmina_crosswalk_2021.csv``). Codes are padded to 7 digits first
   (``code7``) so voivodeships 02-09, stored int-stripped in the 2011 file, are
   not mis-parsed.
2. RODZ 4/5 split -> whole gmina, absorbed automatically because the crosswalk
   and the frame both live in 6-digit whole-gmina space (``code6``).

The within-gmina diagonal is not observed; it is recovered from the residence
margin: ``own_n = max(R_n - outflows_n, 0)`` (residents of n not working outside
n). This is done in ``solve.build_observables`` where R_n is available. Here we
return the off-diagonal matrix (frame-ordered) plus its margins and diagnostics.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..frame import Frame, code6, code7, whole6


def ipf_reconcile(comMat, row_target, col_target, tau=None,
                  gravity_floor_phi=2.0, floor_scale=1e-6,
                  maxiter=200, tol=1e-9):
    """Iterative proportional fitting of a commuting matrix to trusted margins.

    Rakes ``comMat`` so its row sums match ``row_target`` (residence employment,
    R_n) and its column sums match ``col_target`` (workplace employment, L_n),
    while preserving the observed bilateral interaction structure. Because the
    two-way fixed-effects commuting gravity absorbs all row/column scaling, this
    reconciliation leaves the estimated commuting decay phi **invariant** — it
    only fixes census-censoring/vintage inconsistencies in the margins (e.g. the
    11 zero-workplace gminas and 198 out>resident gminas in the 2011 tax file).

    Structural-zero cells receive a tiny distance-decaying seed (``floor_scale *
    tau**(-gravity_floor_phi)``) so gminas with an all-zero observed row/column
    can still reach their target. Targets are rescaled to a common total.
    """
    M = np.asarray(comMat, dtype=float).copy()
    if tau is not None:
        floor = floor_scale * (tau ** (-gravity_floor_phi))
        np.fill_diagonal(floor, floor_scale)     # own-cell floor
        M = M + floor
    else:
        M = M + floor_scale
    rt = np.asarray(row_target, float).copy()
    ct = np.asarray(col_target, float).copy()
    T = 0.5 * (rt.sum() + ct.sum())              # common total for a closed system
    rt = rt * (T / rt.sum())
    ct = ct * (T / ct.sum())

    it = 0
    for it in range(maxiter):
        r = M.sum(axis=1)
        M *= (rt / np.where(r > 0, r, 1.0))[:, None]
        c = M.sum(axis=0)
        M *= (ct / np.where(c > 0, c, 1.0))[None, :]
        err = max(float(np.max(np.abs(M.sum(1) - rt))),
                  float(np.max(np.abs(M.sum(0) - ct))))
        if err < tol * T:
            break
    return M, dict(ipf_iters=it, ipf_final_err=float(err), common_total=float(T))


@dataclass
class Flows:
    off_diag: np.ndarray          # (N,N) off-diagonal flows, frame order, diagonal = 0
    outflows: np.ndarray          # (N,) row sums of off_diag
    inflows: np.ndarray           # (N,) col sums of off_diag
    source: str
    source_year: int
    diagnostics: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Raw parsers (mirror scripts/labour_market/flows.py, hardened)
# --------------------------------------------------------------------------- #
def _parse_2011(path) -> pd.DataFrame:
    df = pd.read_excel(path, engine="xlrd", header=0).iloc[:, :3]
    df.columns = ["res", "work", "count"]
    return df


def _parse_2021(path) -> pd.DataFrame:
    m = pd.read_excel(path, engine="openpyxl",
                      sheet_name="Macierz przepływów", header=None)
    m = m.iloc[1:, [0, 2, 4]].copy()
    m.columns = ["res", "work", "count"]
    return m


def _parse(path, source_year: int) -> pd.DataFrame:
    if int(source_year) == 2011:
        return _parse_2011(path)
    return _parse_2021(path)


# --------------------------------------------------------------------------- #
# Crosswalk (universal 6-digit map onto the 2021 frame)
# --------------------------------------------------------------------------- #
def load_crosswalk_map(crosswalk_csv) -> dict[str, list[tuple[str, float]]]:
    """src_code6 -> [(dst_code6, weight), ...], weights summing to 1.

    The labelled crosswalk carries one row per (src_year, src_code, dst_code).
    Because the mover codes are disjoint across vintages and none collides with a
    2021 code (see scripts/labour_market/teryt.py), a single universal map is
    unambiguous.
    """
    cw = pd.read_csv(crosswalk_csv, dtype=str)
    # crosswalk codes are already 6-digit whole-gmina keys -> normalise with whole6,
    # NOT code6 (which would pad to 7 and shift the voivodeship).
    cw["src"] = cw["src_code"].map(whole6)
    cw["dst"] = cw["dst_code"].map(whole6)
    cw["w"] = cw["weight"].astype(float)
    cw = cw.drop_duplicates(["src", "dst"])
    out: dict[str, list[tuple[str, float]]] = {}
    for src, grp in cw.groupby("src"):
        tot = grp["w"].sum()
        out[src] = [(d, w / tot) for d, w in zip(grp["dst"], grp["w"])]
    return out


def load_flows(path, source_year: int, fr: Frame, crosswalk_csv) -> Flows:
    cwmap = load_crosswalk_map(crosswalk_csv)
    code6_to_pos = {code6(c): i for i, c in enumerate(fr.codes)}
    if len(code6_to_pos) != fr.N:
        raise ValueError("frame code6 keys are not unique — cannot join flows")

    raw = _parse(path, source_year).dropna(subset=["res", "work"])
    raw["res6"] = raw["res"].map(code6)
    raw["work6"] = raw["work"].map(code6)
    raw["count"] = pd.to_numeric(raw["count"], errors="coerce").fillna(0.0)

    M = np.zeros((fr.N, fr.N), dtype=float)
    total_in, total_kept, n_unmapped = 0.0, 0.0, 0
    unmapped_codes: set[str] = set()

    for res6, work6, cnt in zip(raw["res6"], raw["work6"], raw["count"]):
        total_in += cnt
        rmap = cwmap.get(res6)
        wmap = cwmap.get(work6)
        if rmap is None or wmap is None:
            n_unmapped += 1
            if rmap is None:
                unmapped_codes.add(res6)
            if wmap is None:
                unmapped_codes.add(work6)
            continue
        for rd, rw in rmap:
            ri = code6_to_pos.get(rd)
            if ri is None:
                continue
            for wd, ww in wmap:
                wi = code6_to_pos.get(wd)
                if wi is None:
                    continue
                M[ri, wi] += cnt * rw * ww
        total_kept += cnt

    fold_diag = float(np.trace(M))         # mass folded onto the diagonal by RODZ/merge
    np.fill_diagonal(M, 0.0)               # keep only off-diagonal; diagonal recovered later
    outflows = M.sum(axis=1)
    inflows = M.sum(axis=0)

    diag = dict(
        source_rows=int(len(raw)),
        total_commuters_in=total_in,
        total_commuters_kept=total_kept,
        unmapped_rows=int(n_unmapped),
        unmapped_codes=sorted(unmapped_codes)[:20],
        folded_to_diagonal=fold_diag,
        offdiag_total=float(M.sum()),
    )
    if n_unmapped:
        raise ValueError(f"{n_unmapped} flow rows had codes absent from the crosswalk; "
                         f"examples: {sorted(unmapped_codes)[:10]}")
    return Flows(off_diag=M, outflows=outflows, inflows=inflows,
                 source=str(path), source_year=int(source_year), diagnostics=diag)
