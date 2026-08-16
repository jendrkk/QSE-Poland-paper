"""
validate.py — model invariants.

Every RunResult stores the outcome of these checks. The orchestrator refuses to
save a run that fails a HARD invariant when solver.strict_invariants is True.
Mirrors the validation gates enforced by the reference toolkit.
"""
from __future__ import annotations

import numpy as np


HARD = {"sum_L_eq_N", "sum_R_eq_N", "uncond_sums_to_one", "A_positive",
        "tau_finite", "b_geomean_one", "income_eq_expenditure"}


def check(*, N, L_n, R_n, uncondCom, A_n, b_n, tau,
          income=None, expend=None, atol=1e-4) -> dict:
    """Return {name: {passed, value, detail}} for each invariant."""
    out = {}

    def rec(name, passed, value, detail=""):
        out[name] = dict(passed=bool(passed), value=float(value), detail=detail,
                         hard=name in HARD)

    rec("sum_L_eq_N", abs(L_n.sum() - N) < 1e-3, L_n.sum(), f"target {N}")
    rec("sum_R_eq_N", abs(R_n.sum() - N) < 1e-3, R_n.sum(), f"target {N}")
    rec("uncond_sums_to_one", abs(uncondCom.sum() - 1.0) < 1e-8, uncondCom.sum())
    rec("A_positive", np.all(A_n > 0), float(np.min(A_n)), "min A_n")
    rec("b_geomean_one", abs(np.exp(np.log(b_n).mean()) - 1.0) < 1e-6,
        np.exp(np.log(b_n).mean()))
    rec("tau_finite", np.all(np.isfinite(tau)), float(np.max(tau)))
    if income is not None and expend is not None:
        gap = float(np.max(np.abs(income - expend)))
        scale = float(np.max(np.abs(income)))
        rec("income_eq_expenditure", gap < 1e-3 * max(scale, 1.0), gap,
            f"max|inc-exp| (scale {scale:.3g})")
    return out


def all_hard_passed(results: dict) -> bool:
    return all(v["passed"] for v in results.values() if v.get("hard"))


def summary(results: dict) -> str:
    lines = []
    for k, v in results.items():
        flag = "OK " if v["passed"] else "FAIL"
        hard = "*" if v.get("hard") else " "
        lines.append(f"  [{flag}]{hard} {k:24s} = {v['value']:.6g}  {v['detail']}")
    return "\n".join(lines)
