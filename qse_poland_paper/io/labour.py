"""
io/labour.py — labour-market observables from labor_tidy_<year>.csv.

Returns, aligned to the frame:
    w_n   workplace wage        (median_income_workplace)  — raw zloty, NOT normalised
    Lw_n  workplace employment  (employment_workplace)
    Rr_n  residence employment  (employment_residence)

The MRRH normalisations (mean-1 wage, sum-N employment margins) are applied later
in `solve.build_observables`, from the assembled commuting matrix, so that the
margins are internally consistent with the flows.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..frame import Frame, code7


@dataclass
class Labour:
    w_n: np.ndarray       # workplace wage (raw)
    Lw_n: np.ndarray      # workplace employment (raw counts)
    Rr_n: np.ndarray      # residence employment (raw counts)
    w_source: np.ndarray | None = None
    r_source: np.ndarray | None = None


def load_labour(csv_path, fr: Frame) -> Labour:
    df = pd.read_csv(csv_path, dtype={"region_id": str, "teryt7": str, "powiat": str})
    df["teryt7"] = df["teryt7"].map(code7)

    w = fr.reindex_vector(df["teryt7"], df["median_income_workplace"], fill=np.nan)
    Lw = fr.reindex_vector(df["teryt7"], df["employment_workplace"], fill=np.nan)
    Rr = fr.reindex_vector(df["teryt7"], df["employment_residence"], fill=np.nan)

    for name, arr in (("wage", w), ("workplace emp", Lw), ("residence emp", Rr)):
        n_bad = int(np.sum(~np.isfinite(arr)) + np.sum(arr <= 0))
        if n_bad:
            raise ValueError(f"labour {name}: {n_bad} non-finite/non-positive values "
                             f"after aligning {csv_path} to the frame")

    wsrc = df.set_index("teryt7").reindex(fr.codes).get("wage_workplace_source")
    rsrc = df.set_index("teryt7").reindex(fr.codes).get("res_source")
    return Labour(w_n=w, Lw_n=Lw, Rr_n=Rr,
                  w_source=None if wsrc is None else wsrc.values.astype(object),
                  r_source=None if rsrc is None else rsrc.values.astype(object))
