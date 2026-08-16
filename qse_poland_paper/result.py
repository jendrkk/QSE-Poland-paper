"""
result.py — the single self-describing run artefact.

A RunResult bundles EVERYTHING about one model solve: the parameters that were
set or calibrated AND every array the solver produced. It is pickled to a single
``run.pkl`` (the authoritative artefact). A human-readable ``manifest.json`` (no
large arrays) and the resolved config are written alongside for convenience and
provenance.

The bundle is deliberately exhaustive so the visualization layer can draw as many
figures, maps and tables as the data allow from one file, and compare two files
without re-running anything.

Sections
--------
meta         provenance: run id, year, timestamps, sources, versions, flags
params       structural parameters actually used (incl. estimated phi, resolved mu)
frame        spatial frame: codes, labels, geography (dict form)
inputs       observed/assembled observables fed to the model
calibrated   recovered fundamentals and solved equilibrium objects
estimation   in-sample estimates (gravity)
diagnostics  solver iterations, gaps, flow/margin reconciliation
validation   invariant check results
"""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class RunResult:
    schema_version: int
    run_id: str
    year: int
    created_at: str
    package_version: str
    meta: dict[str, Any] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    frame: dict[str, Any] = field(default_factory=dict)
    inputs: dict[str, Any] = field(default_factory=dict)
    calibrated: dict[str, Any] = field(default_factory=dict)
    estimation: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    validation: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    @classmethod
    def new(cls, run_id, year, package_version, **sections) -> "RunResult":
        from . import SCHEMA_VERSION
        return cls(schema_version=SCHEMA_VERSION, run_id=run_id, year=int(year),
                   created_at=datetime.now(timezone.utc).isoformat(),
                   package_version=package_version, **sections)

    # ------------------------------------------------------------------ #
    def _compact_inplace(self):
        """Downcast bulk N×N float64 arrays to float32 to roughly halve the pickle.
        The 1-D recovered fundamentals (A_n, b_n, w_n, ...) stay float64. Model
        re-solves from a loaded run upcast to float64 internally."""
        for sec in (self.inputs, self.calibrated):
            for k, v in list(sec.items()):
                if isinstance(v, np.ndarray) and v.ndim == 2 and v.dtype == np.float64:
                    sec[k] = v.astype(np.float32)

    def save(self, run_dir, compact: bool = True) -> Path:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        if compact:
            self._compact_inplace()
        pkl = run_dir / "run.pkl"
        with open(pkl, "wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)
        (run_dir / "manifest.json").write_text(
            json.dumps(self.manifest(), indent=2, ensure_ascii=False, default=_json_default))
        return pkl

    @staticmethod
    def load(path) -> "RunResult":
        path = Path(path)
        if path.is_dir():
            path = path / "run.pkl"
        with open(path, "rb") as fh:
            return pickle.load(fh)

    # ------------------------------------------------------------------ #
    def manifest(self) -> dict:
        """Scalars, params, provenance and array shapes — never the big arrays."""
        def shapes(d: dict) -> dict:
            out = {}
            for k, v in d.items():
                if isinstance(v, np.ndarray):
                    out[k] = dict(shape=list(v.shape), dtype=str(v.dtype))
                elif isinstance(v, (int, float, str, bool)) or v is None:
                    out[k] = v
                elif isinstance(v, dict):
                    out[k] = shapes(v)
                else:
                    out[k] = str(type(v))
            return out

        return dict(
            schema_version=self.schema_version, run_id=self.run_id, year=self.year,
            created_at=self.created_at, package_version=self.package_version,
            meta=self.meta, params=self.params, estimation=self.estimation,
            diagnostics=_scalars_only(self.diagnostics),
            validation=self.validation,
            frame=dict(N=self.frame.get("N")),
            inputs=shapes(self.inputs), calibrated=shapes(self.calibrated),
        )

    # convenience accessors ------------------------------------------------- #
    def array(self, section: str, name: str) -> np.ndarray:
        return np.asarray(getattr(self, section)[name])

    @property
    def N(self) -> int:
        return int(self.frame["N"])

    @property
    def codes(self) -> np.ndarray:
        return np.asarray(self.frame["codes"])


def _scalars_only(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, np.ndarray):
            out[k] = dict(shape=list(v.shape))
        elif isinstance(v, dict):
            out[k] = _scalars_only(v)
        elif isinstance(v, (int, float, str, bool)) or v is None:
            out[k] = v
        else:
            out[k] = str(v)[:200]
    return out


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
