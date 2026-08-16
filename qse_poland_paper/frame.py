"""
frame.py — the canonical spatial frame.

Everything in the model is aligned to a single ordered list of 7-digit TERYT
gmina codes (the 2021 vintage, N=2477). All inputs — labour, travel time,
floor-space price, geometry — are re-indexed onto this order before any
computation, so positional alignment can never silently drift.

TERYT robustness
----------------
GUS files sometimes store the 7-digit code ``WWPPGGR`` as an integer, which
strips the leading zero for voivodeships 02–09 (e.g. ``0201011`` -> ``201011``).
``code7`` zero-pads to 7 digits *before* anything else, and ``code6`` derives the
whole-gmina key from that. Getting this order wrong silently mis-assigns ~8
voivodeships (verified: the naive ``s[:6].zfill(6)`` leaves 754 of the 2011 flow
codes unmatched; the correct ``s.zfill(7)[:6]`` matches all).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Code helpers
# --------------------------------------------------------------------------- #
def _digits(kod) -> str:
    return "".join(ch for ch in str(kod) if ch.isdigit())


def code7(kod) -> str:
    """Canonical 7-digit TERYT gmina code WWPPGGR (zero-padded)."""
    return _digits(kod).zfill(7)


def code6(kod) -> str:
    """Whole-gmina 6-digit key from a *7-digit TERYT* code WWPPGGR: pad to 7
    (restoring a stripped leading zero) then drop the RODZ digit."""
    return code7(kod)[:6]


def whole6(kod) -> str:
    """Normalise an *already 6-digit* whole-gmina code WWPPGG (e.g. crosswalk
    src/dst codes). These must NOT be padded to 7 first — that would shift the
    voivodeship. Zero-pad to 6."""
    return _digits(kod).zfill(6)


# --------------------------------------------------------------------------- #
# Frame object
# --------------------------------------------------------------------------- #
@dataclass
class Frame:
    codes: np.ndarray          # (N,) 7-digit teryt strings, canonical order
    N: int
    index: dict                # teryt7 -> positional index
    # labels / geography (aligned to `codes`)
    nazwa: np.ndarray | None = None
    powiat: np.ndarray | None = None
    woj: np.ndarray | None = None
    rodz_class: np.ndarray | None = None
    area_km2: np.ndarray | None = None    # optional (from geometry); None if unknown
    pop: np.ndarray | None = None         # optional (from floor-space index)

    # ------------------------------------------------------------------ #
    def reindex_vector(self, codes, values, fill=np.nan, dtype=float) -> np.ndarray:
        """Map a (code -> value) pair of arrays onto the frame order."""
        s = pd.Series(np.asarray(values, dtype=dtype),
                      index=[code7(c) for c in codes])
        s = s[~s.index.duplicated(keep="first")]
        out = s.reindex(self.codes).values
        if fill is not None:
            out = np.where(np.isnan(out), fill, out) if dtype == float else out
        return out.astype(dtype)

    def positions(self, codes) -> np.ndarray:
        """Positional indices of a list of teryt codes within the frame."""
        return np.array([self.index[code7(c)] for c in codes], dtype=int)


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #
def build_frame(labour_csv, floorspace_csv=None) -> Frame:
    """Canonical frame from a labor_tidy csv (teryt7 ascending). Optionally attach
    population from the floor-space index."""
    lab = pd.read_csv(labour_csv, dtype={"region_id": str, "teryt7": str, "powiat": str})
    lab["teryt7"] = lab["teryt7"].map(code7)
    lab = lab.drop_duplicates("teryt7").sort_values("teryt7").reset_index(drop=True)

    codes = lab["teryt7"].values.astype("U7")
    N = len(codes)
    index = {c: i for i, c in enumerate(codes)}

    fr = Frame(
        codes=codes, N=N, index=index,
        nazwa=lab.get("nazwa", pd.Series([""] * N)).values.astype(object),
        powiat=lab.get("powiat", pd.Series([""] * N)).astype(str).values.astype(object),
        woj=np.array([c[:2] for c in codes], dtype=object),
        rodz_class=lab.get("rodz_class", pd.Series([""] * N)).values.astype(object),
    )

    if floorspace_csv is not None:
        try:
            fs = pd.read_csv(floorspace_csv, dtype={"gmina_teryt": str})
            if "pop" in fs.columns:
                fr.pop = fr.reindex_vector(fs["gmina_teryt"], fs["pop"], fill=np.nan)
        except Exception:
            pass
    return fr


def attach_area_from_gpkg(fr: Frame, gpkg_path, teryt_field_candidates=(
        "teryt7", "TERYT", "JPT_KOD_JE", "kod", "gmina_teryt", "code7")) -> Frame:
    """Attach official polygon area (km^2) from the commune GeoPackage, if geopandas
    is available. Non-fatal: leaves area_km2=None on any failure."""
    try:
        import geopandas as gpd
    except Exception:
        return fr
    try:
        g = gpd.read_file(gpkg_path)
        key = next((c for c in teryt_field_candidates if c in g.columns), None)
        if key is None:
            return fr
        g["_k"] = g[key].map(code7)
        # metric CRS assumed (EPSG:2180); area in m^2 -> km^2
        area = g.geometry.area / 1e6
        fr.area_km2 = fr.reindex_vector(g["_k"], area, fill=np.nan)
    except Exception:
        pass
    return fr
