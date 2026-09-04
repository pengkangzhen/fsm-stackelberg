"""Fault-plant catalog for Exp-I attribution pilots.

Plants mutate blackboard agent outputs *after* the guilty layer finishes its
forward work and *before* downstream agents consume them. Labels
(``true_root_cause``) are evaluation-only — never written into prompts.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence

from ..schemas import Constraint, ModelExpertOutput

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


def _as_me(state: Dict[str, Any]) -> ModelExpertOutput:
    me = state.get("model_expert_output")
    if me is None:
        raise ValueError("plant requires model_expert_output")
    if isinstance(me, dict):
        return ModelExpertOutput.model_validate(me)
    return me


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
