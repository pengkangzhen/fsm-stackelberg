"""LangGraph node that applies a one-shot fault plant before PythonDeveloper."""

from __future__ import annotations

import logging
from typing import Any, Dict

from ..injection import apply_plant
from ..utils.utils import record_agent_output

logger = logging.getLogger(__name__)


def fault_injector_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Apply ``inject_id`` plant once on the ME→PD boundary.

    No-op when inject_id is unset or ``fault_injected`` is already True so that
    post-repair ME regenerations are not re-corrupted.

    Records the planted ME as a forward history entry so diagnosis / ME
    backward see the post-injection blackboard (not the clean pre-plant dump).
    """
    plant_id = state.get("inject_id")
    if not plant_id:
        return state
    if state.get("fault_injected"):
        logger.info("Fault injector: plant %s already applied; skipping", plant_id)
        return state

    logger.info("Fault injector: applying plant %s", plant_id)
    updated = apply_plant(dict(state), plant_id)
    me = updated.get("model_expert_output")
    if me is not None:
        hist = list(updated.get("output_history") or [])
        updated["output_history"] = record_agent_output(
            output_history=hist,
            agent_name="model_expert",
            output=me,
            current_round=int(updated.get("current_round") or 1),
            step_type="forward",
        )
    return updated
