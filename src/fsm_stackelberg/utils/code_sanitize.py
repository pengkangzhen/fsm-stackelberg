"""Deterministic post-processing for LLM-generated Gurobi code.

Primary target: model/mode variable shadowing that caused PD regen failures
after ME force-zero strip in Exp-A causal cells (ea_causal_r1/r2):

    m = gp.Model(...)
    for (i, j, m) in arcs:          # shadows model → str
        x[...] = m.addVar(...)      # AttributeError: str has no addVar
    ...
    m.dispose()                     # AttributeError: str has no dispose
"""

from __future__ import annotations

import re
from typing import Tuple

# Gurobi Model API surface commonly emitted by PD.
_MODEL_MEMBERS = (
    "addVar",
    "addVars",
    "addConstr",
    "addConstrs",
    "addLConstr",
    "addQConstr",
    "addGenConstrIndicator",
    "optimize",
    "update",
    "dispose",
    "setObjective",
    "setParam",
    "setParams",
    "getVars",
    "getConstrs",
    "getVarByName",
    "reset",
    "write",
    "printStats",
    "terminate",
    "Status",
    "ObjVal",
    "ObjBound",
    "SolCount",
    "ModelName",
    "NumVars",
    "NumConstrs",
    "IsMIP",
)

_MODEL_MEMBER_RE = re.compile(
    r"\bm\.(" + "|".join(_MODEL_MEMBERS) + r")\b"
)
_MODEL_ASSIGN_RE = re.compile(r"(?m)^(\s*)m(\s*=\s*gp\.Model\b)")
# Tuple unpack that binds `m` (mode index) — the shadow source.
_FOR_MODE_M_RE = re.compile(
    r"for\s*\((?:[^)]*,\s*)*m(?:\s*,[^)]*)?\)\s*in\b"
)


def detects_model_mode_shadowing(code: str) -> bool:
    """True when `m = gp.Model(...)` coexists with `for (..., m) in ...`."""
    if not code:
        return False
    return bool(_MODEL_ASSIGN_RE.search(code) and _FOR_MODE_M_RE.search(code))


def unshadow_gurobi_model_m(code: str) -> Tuple[str, bool]:
    """Rename Gurobi model `m` → `model` when shadowed by mode index `m`.

    Leaves loop/index uses of bare `m` (e.g. f-strings, dict keys) untouched;
    only rewrites `m = gp.Model(...)` and `m.<ModelAPI>`.

    Returns (possibly_rewritten_code, did_rewrite).
    """
    if not detects_model_mode_shadowing(code):
        return code, False
    out = _MODEL_ASSIGN_RE.sub(r"\1model\2", code)
    out = _MODEL_MEMBER_RE.sub(r"model.\1", out)
    return out, out != code
