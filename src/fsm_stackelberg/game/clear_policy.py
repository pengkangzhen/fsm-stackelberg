"""Principled inspection clear / reopen policy (Stackelberg ν).

Deflection is **symptom-scoped**, not a permanent exoneration:

- On deflect, the layer may be cleared so ω can advance (avoid re-probe loops
  on the same surface failure).
- When the surface symptom fingerprint later **changes**, and the layer still
  carries unresolved structural formulation evidence on the blackboard
  (currently: force-zero / injected empty-sea plant smell on ME or code),
  reopen that layer for further probing under executed refutation.

This is a mechanism rule — not an experiment-chasing prompt patch.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Sequence

from .ranking import has_force_zero_sea_smell

logger = logging.getLogger(__name__)


def symptom_fingerprint(state: Dict[str, Any]) -> str:
    """Compact id of the current surface failure (category × status × type)."""
    category = str(state.get("error_category") or "").strip()
    status = str(state.get("gurobi_status") or "").strip().upper()
    err_type = ""
    raw = state.get("error_info")
    if isinstance(raw, dict):
        err_type = str(raw.get("error_type") or raw.get("error_message") or "")[:120]
    elif isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                err_type = str(
                    parsed.get("error_type") or parsed.get("error_message") or ""
                )[:120]
            else:
                err_type = raw.strip()[:120]
        except json.JSONDecodeError:
            err_type = raw.strip()[:120]
    return f"{category}|{status}|{err_type}"


def layer_has_unresolved_formulation_evidence(
    state: Dict[str, Any],
    layer: str,
) -> bool:
    """True if ``layer`` still harbors structural plant/formulation evidence.

    Today this is the force-zero / injected empty-sea smell already used by
    ranking (traceback ≠ root). Extend here if new structural plants appear —
    do not encode draw-specific KeyError strings.
    """
    if layer != "model_expert":
        return False
    return has_force_zero_sea_smell(
        state.get("model_expert_output")
    ) or has_force_zero_sea_smell(state.get("python_code") or "")


def reopen_symptom_scoped_deflects(
    cleared_layers: Sequence[str],
    refutation_log: Sequence[Dict[str, Any]],
    state: Dict[str, Any],
) -> List[str]:
    """Drop cleared layers whose deflect no longer matches the current symptom.

    Comply/overturn clears stay. Deflect clears reopen only when:
      1. last settlement for the layer is ``deflect``, and
      2. recorded ``symptom_fingerprint`` ≠ current fingerprint, and
      3. unresolved formulation evidence remains on that layer.
    """
    cleared = list(cleared_layers or [])
    if not cleared:
        return cleared

    fp_now = symptom_fingerprint(state)
    last_by_layer: Dict[str, Dict[str, Any]] = {}
    for entry in refutation_log or []:
        layer = entry.get("layer")
        if layer:
            last_by_layer[str(layer)] = entry

    kept: List[str] = []
    for layer in cleared:
        last = last_by_layer.get(layer)
        fp_then = (last or {}).get("symptom_fingerprint")
        if (
            last
            and last.get("action") == "deflect"
            and fp_then
            and fp_then != fp_now
            and layer_has_unresolved_formulation_evidence(state, layer)
        ):
            logger.info(
                "Inspection clear policy: reopen layer=%s "
                "(deflect under symptom %r; now %r; formulation evidence remains)",
                layer,
                fp_then,
                fp_now,
            )
            continue
        kept.append(layer)
    return kept
