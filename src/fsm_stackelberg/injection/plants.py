"""Fault-plant catalog for Exp-I attribution pilots.

Plants mutate blackboard agent outputs *after* the guilty layer finishes its
forward work and *before* downstream agents consume them. Labels
(``true_root_cause``) are evaluation-only — never written into prompts.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

from ..schemas import Constraint, DataEngineerOutput, ModelExpertOutput

logger = logging.getLogger(__name__)

_STAGE2_BALANCE_RE = re.compile(
    r"(inventory[_\s-]*balance|inland[_\s-]*balance|"
    r"balance.*(hub|non.?hub|spoke|dry|general|node)|"
    r"stage\s*2.*balance|flow\s*conservation)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PlantSpec:
    """One injectable fault fixture."""

    plant_id: str
    true_root_cause: str  # data_engineer | model_expert | python_developer
    description: str
    apply: Callable[[Dict[str, Any]], Dict[str, Any]]
    target_layer: str = "model_expert"  # injector boundary that applies this plant


def _as_me(state: Dict[str, Any]) -> ModelExpertOutput:
    me = state.get("model_expert_output")
    if me is None:
        raise ValueError("plant requires model_expert_output")
    if isinstance(me, dict):
        return ModelExpertOutput.model_validate(me)
    return me


def _as_de(state: Dict[str, Any]) -> DataEngineerOutput:
    de = state.get("data_engineer_output")
    if de is None:
        raise ValueError("plant requires data_engineer_output")
    if isinstance(de, dict):
        return DataEngineerOutput.model_validate(de)
    return de


_DEMAND_RE = re.compile(r"demand", re.IGNORECASE)
_SUPPLY_RE = re.compile(r"supply", re.IGNORECASE)
_ADD_CONSTR_RE = re.compile(r"addConstr|add_constr", re.IGNORECASE)
_BALANCE_TOKEN_RE = re.compile(r"balance", re.IGNORECASE)
_STR_LIT_RE = re.compile(r'"[^"\n]*"|\'[^\'\n]*\'')
_MAX_CONSTR_SPAN_LINES = 30


def _constr_statement_spans(lines: List[str]) -> List[tuple]:
    """Inclusive (start, end) spans of addConstr(...) statements.

    Real LLM-written code frequently splits the call across lines with the
    ``name=`` argument (where the balance token lives) on a continuation
    line, so matching must be statement-level, not line-level. String
    literals are stripped before counting parentheses; a call whose
    parentheses never balance within _MAX_CONSTR_SPAN_LINES falls back to
    the opener line only.
    """
    spans = []
    i = 0
    while i < len(lines):
        if _ADD_CONSTR_RE.search(lines[i]) and not lines[i].lstrip().startswith("#"):
            depth, opened, end = 0, False, None
            for k in range(i, min(i + _MAX_CONSTR_SPAN_LINES, len(lines))):
                bare = _STR_LIT_RE.sub("", lines[k])
                depth += bare.count("(") - bare.count(")")
                if "(" in bare:
                    opened = True
                if opened and depth <= 0:
                    end = k
                    break
            spans.append((i, end if end is not None else i))
            i = (end if end is not None else i) + 1
        else:
            i += 1
    return spans


def _find_param_index(params, pattern) -> Optional[int]:
    """Index of the first parameter whose symbol/description matches and has a source."""
    for i, p in enumerate(params):
        blob = f"{p.symbol} {p.description}"
        if pattern.search(blob) and (p.source or "").strip():
            return i
    return None


def _is_stage2_balance(constraint: Constraint) -> bool:
    blob = f"{constraint.name} {constraint.description}"
    return bool(_STAGE2_BALANCE_RE.search(blob))


def _strip_recovery_context(state: Dict[str, Any]) -> Dict[str, Any]:
    """Remove full-text knowledge so PD cannot silently re-derive the ME fault."""
    return {
        **state,
        "loaded_knowledge": "",
    }


def _drop_stage2_balance(state: Dict[str, Any]) -> Dict[str, Any]:
    """Remove inland/stage-2 inventory-balance constraints from ME output."""
    me = _as_me(state)

    original = list(me.model_components.constraints)
    kept = [c for c in original if not _is_stage2_balance(c)]
    dropped = [c.name for c in original if _is_stage2_balance(c)]

    if not dropped:
        kept = [c for c in original if "balance" not in c.name.lower()]
        dropped = [c.name for c in original if "balance" in c.name.lower()]

    if not dropped:
        raise ValueError(
            "me_drop_stage2_balance: no balance constraints found to drop — "
            "plant cannot guarantee a labeled ME fault"
        )
    if not kept:
        raise ValueError(
            "me_drop_stage2_balance: refusing to drop all constraints"
        )

    new_components = me.model_components.model_copy(update={"constraints": kept})
    new_me = me.model_copy(
        update={
            "model_components": new_components,
            "knowledge_requests": [],
        }
    )
    logger.warning(
        "Fault plant me_drop_stage2_balance: dropped %d constraint(s): %s",
        len(dropped),
        dropped,
    )
    return {
        **state,
        "model_expert_output": new_me,
        "fault_plant_id": "me_drop_stage2_balance",
        "fault_injected": True,
        "true_root_cause": state.get("true_root_cause") or "model_expert",
    }


def _me_force_zero_sea(state: Dict[str, Any]) -> Dict[str, Any]:
    """Tight ME plant: force sea repositioning to zero; keep min + balances.

    Produces a feasible but wrong DEP (Spurious Optimal / large gap) without
    flipping the objective or stripping balances (those cascaded into PD bugs
    and muddied attribution in v2).
    """
    me = _as_me(state)
    components = me.model_components
    constraints = [
        c
        for c in components.constraints
        if c.name != "Injected_Force_Zero_Sea_Reposition"
    ]
    constraints.append(
        Constraint(
            name="Injected_Force_Zero_Sea_Reposition",
            expression=(
                "(forall h hubs (forall t periods "
                "(and (= y_in[h,t] 0) (= y_out[h,t] 0))))"
            ),
            description=(
                "Force all stage-1 sea repositioning arrivals and departures "
                "to zero for every hub and period."
            ),
        )
    )

    new_components = components.model_copy(update={"constraints": constraints})
    # Keep objective direction as ME wrote it (almost always min).
    new_me = me.model_copy(
        update={
            "model_components": new_components,
            "knowledge_requests": [],
            "assumptions_made": list(me.assumptions_made),
        }
    )
    logger.warning(
        "Fault plant me_force_zero_sea: added Force_Zero_Sea only "
        "(kept objective=%s, kept %d other constraints)",
        components.objective_function.direction,
        len(constraints) - 1,
    )
    updated = {
        **state,
        "model_expert_output": new_me,
        "fault_plant_id": "me_force_zero_sea",
        "fault_injected": True,
        "true_root_cause": state.get("true_root_cause") or "model_expert",
    }
    return _strip_recovery_context(updated)


def _de_swap_demand_supply_source(state: Dict[str, Any]) -> Dict[str, Any]:
    """DE plant: swap the catalog sources of demand-like and supply-like params.

    Under the deterministic data contract DE output is value-free, so the
    data-layer fault class is a wrong semantic mapping: ME/PD code against the
    swapped physical tables, producing a feasible-but-wrong DEP (Spurious
    Optimal) whose surface looks like a modeling/code issue.
    """
    de = _as_de(state)
    params = list(de.model_inputs.parameters)
    di = _find_param_index(params, _DEMAND_RE)
    si = _find_param_index(params, _SUPPLY_RE)
    if di is None or si is None:
        raise ValueError(
            "de_swap_demand_supply_source: no sourced demand/supply parameters "
            "found — plant cannot guarantee a labeled data_engineer fault"
        )
    src_d, src_s = params[di].source, params[si].source
    if src_d == src_s:
        raise ValueError(
            "de_swap_demand_supply_source: demand and supply share source "
            f"{src_d!r} — swap would be a no-op"
        )
    params[di] = params[di].model_copy(update={"source": src_s})
    params[si] = params[si].model_copy(update={"source": src_d})
    new_inputs = de.model_inputs.model_copy(update={"parameters": params})
    new_de = de.model_copy(update={"model_inputs": new_inputs})
    logger.warning(
        "Fault plant de_swap_demand_supply_source: swapped sources of %s <-> %s "
        "(%s <-> %s)",
        params[di].symbol,
        params[si].symbol,
        src_s,
        src_d,
    )
    return {
        **state,
        "data_engineer_output": new_de,
        "fault_plant_id": "de_swap_demand_supply_source",
        "fault_injected": True,
        "true_root_cause": state.get("true_root_cause") or "data_engineer",
    }


def _pd_comment_out_balance(state: Dict[str, Any]) -> Dict[str, Any]:
    """PD plant: comment out addConstr statements registering balance constraints.

    The solver then sees a relaxed DEP (Spurious Optimal / large gap) while the
    ME formulation on the blackboard is correct — a code-layer translation
    fault. Statements are matched paren-balanced (single- or multi-line),
    because the ``balance`` naming usually sits on the ``name=`` continuation
    line of a multi-line call. Guarded: refuses (raises) when no matching
    statement exists, so a run can never be silently mislabeled.

    Caveat (shared with ME plants under regeneration): if diagnosis probes ME
    first and ME regenerates, PD re-codes from the clean ME output and the
    fault vanishes without PD ever being probed. That dynamic is itself grid
    evidence, not a plant bug.
    """
    code = state.get("python_code") or ""
    if not code.strip():
        raise ValueError("pd_comment_out_balance: python_code missing/empty")
    lines = code.splitlines()
    spans = [
        (a, b)
        for a, b in _constr_statement_spans(lines)
        if _BALANCE_TOKEN_RE.search(" ".join(lines[a:b + 1]))
    ]
    if not spans:
        raise ValueError(
            "pd_comment_out_balance: no addConstr balance statements found — "
            "plant cannot guarantee a labeled python_developer fault"
        )
    preview = [lines[a].strip()[:60] for a, _ in spans]
    hit = [i for a, b in spans for i in range(a, b + 1)]
    for i in hit:
        if lines[i].lstrip().startswith("#"):
            continue
        indent = lines[i][: len(lines[i]) - len(lines[i].lstrip())]
        lines[i] = f"{indent}# {lines[i].lstrip()}"
    logger.warning(
        "Fault plant pd_comment_out_balance: commented %d balance addConstr "
        "statement(s) spanning %d line(s): %s",
        len(spans),
        len(hit),
        preview,
    )
    new_code = "\n".join(lines)
    if code.endswith("\n"):
        new_code += "\n"
    return {
        **state,
        "python_code": new_code,
        "fault_plant_id": "pd_comment_out_balance",
        "fault_injected": True,
        "true_root_cause": state.get("true_root_cause") or "python_developer",
    }


PLANTS: Dict[str, PlantSpec] = {
    "me_drop_stage2_balance": PlantSpec(
        plant_id="me_drop_stage2_balance",
        true_root_cause="model_expert",
        description=(
            "Drop stage-2 / inland inventory-balance constraints from "
            "ModelExpertOutput so PD codes an incomplete DEP (upstream ME fault)."
        ),
        apply=_drop_stage2_balance,
    ),
    "me_force_zero_sea": PlantSpec(
        plant_id="me_force_zero_sea",
        true_root_cause="model_expert",
        description=(
            "Tight ME plant: force y_in=y_out=0 only; keep min objective and "
            "balances; clear loaded knowledge bodies so PD cannot silently "
            "undo the forced-zero from docs."
        ),
        apply=_me_force_zero_sea,
    ),
    "de_swap_demand_supply_source": PlantSpec(
        plant_id="de_swap_demand_supply_source",
        true_root_cause="data_engineer",
        description=(
            "DE plant: swap the catalog sources of demand-like and supply-like "
            "parameters so downstream agents code against the wrong physical "
            "tables (upstream data-mapping fault under the value-free contract)."
        ),
        apply=_de_swap_demand_supply_source,
        target_layer="data_engineer",
    ),
    "pd_comment_out_balance": PlantSpec(
        plant_id="pd_comment_out_balance",
        true_root_cause="python_developer",
        description=(
            "PD plant: comment out addConstr lines registering balance "
            "constraints — solver sees a relaxed DEP while the ME formulation "
            "stays correct (code-layer translation fault)."
        ),
        apply=_pd_comment_out_balance,
        target_layer="python_developer",
    ),
}


def list_plants() -> Sequence[PlantSpec]:
    return tuple(PLANTS.values())


def get_plant(plant_id: str) -> PlantSpec:
    if plant_id not in PLANTS:
        known = ", ".join(sorted(PLANTS)) or "(none)"
        raise KeyError(f"Unknown plant id {plant_id!r}. Known: {known}")
    return PLANTS[plant_id]


def apply_plant(state: Dict[str, Any], plant_id: Optional[str]) -> Dict[str, Any]:
    """Apply plant once. No-op if plant_id is empty or already injected."""
    if not plant_id:
        return state
    if state.get("fault_injected"):
        return state
    plant = get_plant(plant_id)
    updated = plant.apply(dict(state))
    if not updated.get("true_root_cause"):
        updated["true_root_cause"] = plant.true_root_cause
    updated["fault_plant_id"] = plant.plant_id
    updated["fault_injected"] = True
    return updated
