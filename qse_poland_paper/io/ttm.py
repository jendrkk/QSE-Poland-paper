"""
io/ttm.py — bilateral road travel-time matrix from a .npz bundle.

The delivered bundles are {matrix (N×N float32, minutes, diagonal already imputed
via the √area own-time convention, fully finite), ids (7-digit teryt strings),
meta (JSON)}. We re-index the matrix onto the frame order and return minutes as
float64.

Diagonal re-imputation is available (params.reimpute_tau_diagonal) for the
own-commute speed-sensitivity check the paper requires, but is OFF by default
because the delivered diagonal is already imputed.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from ..frame import Frame, code7


@dataclass
class TTM:
    tau: np.ndarray        # (N,N) minutes, frame-ordered, finite
    meta: dict
    source: str


def load_ttm(npz_path, fr: Frame, params=None) -> TTM:
    z = np.load(npz_path, allow_pickle=True)
    mat = np.asarray(z["matrix"], dtype=float)
    ids = [code7(x) for x in z["ids"]]
    meta = {}
    if "meta" in z.files:
        try:
            meta = json.loads(str(z["meta"]))
        except Exception:
            meta = {}

    if len(ids) != fr.N:
        raise ValueError(f"TTM has {len(ids)} ids, frame has {fr.N}")
    pos = {c: i for i, c in enumerate(ids)}
    missing = [c for c in fr.codes if c not in pos]
    if missing:
        raise ValueError(f"TTM missing {len(missing)} frame codes, e.g. {missing[:5]}")
    order = np.array([pos[c] for c in fr.codes], dtype=int)
    tau = mat[np.ix_(order, order)].copy()

    if not np.all(np.isfinite(tau)):
        raise ValueError("TTM contains non-finite entries after alignment")

    # Floor non-positive off-diagonal travel times. Isolated routing artefacts
    # (e.g. two commune centroids that snapped to the same node) produce a 0 that
    # would break log(tau) in the gravity and dni**(1-sigma) in the inversion.
    off = ~np.eye(fr.N, dtype=bool)
    bad = off & (tau <= 0)
    n_floored = int(bad.sum())
    if n_floored:
        pos_min = float(tau[off & (tau > 0)].min())
        tau[bad] = pos_min
    meta = dict(meta)
    meta["n_floored_offdiag"] = n_floored
    # guard the diagonal too (own-time floor)
    d = np.diag(tau).copy()
    if np.any(d <= 0):
        floor = getattr(params, "own_time_floor_min", 3.0) if params else 3.0
        d[d <= 0] = floor
        np.fill_diagonal(tau, d)

    # optional diagonal re-imputation (speed-sensitivity check)
    if params is not None and getattr(params, "reimpute_tau_diagonal", False):
        if fr.area_km2 is None or not np.all(np.isfinite(fr.area_km2)):
            raise ValueError("reimpute_tau_diagonal=True requires frame.area_km2 "
                             "(attach geometry first)")
        own = (params.own_time_c * np.sqrt(fr.area_km2 / np.pi)
               / params.own_time_speed_kmh * 60.0)
        own = np.maximum(own, params.own_time_floor_min)
        np.fill_diagonal(tau, own)

    return TTM(tau=tau, meta=meta, source=str(npz_path))
