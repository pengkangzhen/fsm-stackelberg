"""Verified attribution from executed-refutation logs (SPEC §6)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def make_refutation_entry(
    *,
    layer: str,
    action: str,
    error_resolved: bool,
    re_solve_strict_success: bool,
    retry_index: int,
    symptom_cleared: bool | None = None,
    overturn: bool | None = None,
    symptom_fingerprint: str | None = None,
) -> Dict[str, Any]:
    """One probe-settlement record for ``state['refutation_log']``."""
    if overturn is None:
        # Comply claimed but re-solve failed → overturn; deflect upheld → False.
        overturn = bool(error_resolved) and not re_solve_strict_success
    if symptom_cleared is None:
        symptom_cleared = bool(re_solve_strict_success)
    entry: Dict[str, Any] = {
        "layer": layer,
        "action": action,  # "comply" | "deflect"
        "error_resolved": bool(error_resolved),
        "re_solve_strict_success": bool(re_solve_strict_success),
        "symptom_cleared": bool(symptom_cleared),
        "overturn": bool(overturn),
        "retry_index": int(retry_index),
    }
    if symptom_fingerprint is not None:
        entry["symptom_fingerprint"] = str(symptom_fingerprint)
    return entry


def ensure_success_refutation_entry(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Append a success entry when episode ended on comply + Practical Optimal.

    Comply paths that succeed never re-enter diagnosis, so the live log may
    miss the verifying re-solve. Call from finalize.
    """
    from .payoff import is_strict_success

    log: List[Dict[str, Any]] = list(state.get("refutation_log") or [])
    if not is_strict_success(state):
        return log
    if int(state.get("retry_count") or 0) <= 0:
        return log
    if not state.get("error_resolved"):
        return log
    layer = state.get("error_agent")
    if not layer:
        return log
    # Already recorded a successful re-solve for this layer?
    for entry in log:
        if (
            entry.get("layer") == layer
            and entry.get("action") == "comply"
            and entry.get("re_solve_strict_success")
        ):
            return log
    log.append(
        make_refutation_entry(
            layer=layer,
            action="comply",
            error_resolved=True,
            re_solve_strict_success=True,
            retry_index=int(state.get("retry_count") or 0),
            symptom_cleared=True,
            overturn=False,
        )
    )
    return log


def infer_verified_attribution(state: Dict[str, Any]) -> Dict[str, Any]:
    """Primary Exp-B+ metric: verified attribution from executed refutation.

    Freeze (SPEC §6.1 + unit tests):
      1. comply + subsequent Practical Optimal → â_ver = that layer
      2. only deflects / failed repairs → â_ver = None (never last-comply alone)
      3. never from LLM confidence
    """
    log = ensure_success_refutation_entry(state)
    verified_layer: Optional[str] = None
    for entry in log:
        if entry.get("action") == "comply" and entry.get("re_solve_strict_success"):
            verified_layer = entry.get("layer")

    true_root = state.get("true_root_cause")
    verified_hit: Optional[bool] = None
    if true_root is not None:
        verified_hit = verified_layer == true_root

    successes = [
        e for e in log
        if e.get("action") == "comply" and e.get("re_solve_strict_success")
    ]
    overturns = [e for e in log if e.get("overturn")]
    summary = {
        "n_probes": len(log),
        "n_comply_success": len(successes),
        "n_overturn": len(overturns),
        "layers_probed": [e.get("layer") for e in log],
    }

    return {
        "verified_attributed_layer": verified_layer,
        "verified_attribution_hit": verified_hit,
        "refutation_log": log,
        "refutation_log_summary": summary,
    }
