"""LangGraph nodes that apply one-shot fault plants at layer boundaries.

A single global ``fault_injected`` flag guarantees one-shot semantics across
all boundaries: the plant fires exactly once, at the boundary matching its
``target_layer`` (DE plant on the DE→ME edge, ME plant on the ME→PD edge, PD
plant on the PD→solver edge). Post-repair regenerations are never re-corrupted.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict

from ..injection import apply_plant, get_plant
from ..utils.utils import record_agent_output

logger = logging.getLogger(__name__)


def make_fault_injector_node(layer: str) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """Build a boundary injector that only applies plants targeting ``layer``.

    No-op when inject_id is unset, a plant was already applied, or the plant
    targets a different layer. When it fires, the planted output is recorded as
    a forward history entry so diagnosis / backward steps see the
    post-injection blackboard (not the clean pre-plant dump).
    """

    def _node(state: Dict[str, Any]) -> Dict[str, Any]:
        plant_id = state.get("inject_id")
        if not plant_id:
            return state
        if state.get("fault_injected"):
            logger.info("Fault injector(%s): plant %s already applied; skipping", layer, plant_id)
            return state
        plant = get_plant(plant_id)
        if plant.target_layer != layer:
            return state

        logger.info("Fault injector(%s): applying plant %s", layer, plant_id)
        updated = apply_plant(dict(state), plant_id)
        if layer == "data_engineer":
            output = updated.get("data_engineer_output")
        elif layer == "model_expert":
            output = updated.get("model_expert_output")
        else:
            output = updated.get("python_code")
        if output is not None:
            hist = list(updated.get("output_history") or [])
            updated["output_history"] = record_agent_output(
                output_history=hist,
                agent_name=layer,
                output=output,
                current_round=int(updated.get("current_round") or 1),
                step_type="forward",
            )
        return updated

    return _node


# ME→PD boundary injector (original single injection point; default for ME plants).
fault_injector_node = make_fault_injector_node("model_expert")
# DE→ME boundary injector for data-layer plants.
fault_injector_de_node = make_fault_injector_node("data_engineer")
# PD→solver boundary injector for code-layer plants.
fault_injector_pd_node = make_fault_injector_node("python_developer")
