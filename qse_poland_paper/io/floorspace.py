"""
io/floorspace.py — gmina residential floor-space price Q_n (zloty / m^2).

Loads the hedonic gmina index (``index_zl_m2``) and aligns it to the frame.
Currently only a single 2021 timestamp exists; the same index is reused for all
model years (2011, 2021, 2026). This is a documented limitation — the loader
takes an explicit path so a per-year index can be swapped in later with no code
change.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..frame import Frame, code7


@dataclass
class Floorspace:
    Q_n: np.ndarray           # zloty/m^2, frame-ordered
    source: str
    single_timestamp: bool    # True while a per-year index is unavailable
    extras: dict              # optional auxiliary columns (pop, land_ppm2, ...)


_VALUE_CANDIDATES = ("index_zl_m2", "Q_n", "rentindex", "price_m2")


def load_floorspace(csv_path, fr: Frame, single_timestamp: bool = True) -> Floorspace:
    df = pd.read_csv(csv_path, dtype={"gmina_teryt": str})
    key = "gmina_teryt" if "gmina_teryt" in df.columns else (
        "teryt7" if "teryt7" in df.columns else df.columns[0])
    df[key] = df[key].map(code7)
    val = next((c for c in _VALUE_CANDIDATES if c in df.columns), None)
    if val is None:
        raise ValueError(f"no recognised price column in {csv_path}; "
                         f"looked for {_VALUE_CANDIDATES}")

    Q = fr.reindex_vector(df[key], df[val], fill=np.nan)
    n_bad = int(np.sum(~np.isfinite(Q)) + np.sum(Q <= 0))
    if n_bad:
        raise ValueError(f"floor-space price: {n_bad} non-finite/non-positive values "
                         f"after aligning {csv_path}")

    extras = {}
    for c in ("pop", "land_ppm2", "gus_median", "has_rcn", "n_obs"):
        if c in df.columns:
            try:
                extras[c] = fr.reindex_vector(df[key], df[c], fill=np.nan)
            except Exception:
                pass
    return Floorspace(Q_n=Q, source=str(csv_path),
                      single_timestamp=single_timestamp, extras=extras)
