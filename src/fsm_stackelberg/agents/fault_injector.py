"""LangGraph node that applies a one-shot fault plant before PythonDeveloper."""

from __future__ import annotations

import logging
from typing import Any, Dict

from ..injection import apply_plant

logger = logging.getLogger(__name__)


def fault_injector_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Apply ``inject_id`` plant once on the ME→PD boundary.

    No-op when inject_id is unset or ``fault_injected`` is already True so that
    post-repair ME regenerations are not re-corrupted.
    """
    plant_id = state.get("inject_id")
    if not plant_id:
        return state
    if state.get("fault_injected"):
        logger.info("Fault injector: plant %s already applied; skipping", plant_id)
        return state

    logger.info("Fault injector: applying plant %s", plant_id)
    return apply_plant(state, plant_id)
