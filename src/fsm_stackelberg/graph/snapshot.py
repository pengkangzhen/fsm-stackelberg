"""Failure-time snapshot freeze / resume for cost-controlled experiment grids.

Freeze: when ``snapshot_dir`` is set, a gate node between the solver and
diagnosis serializes the full blackboard state at the moment the graph is
about to route into diagnosis. Everything up to that point is
policy-invariant (same forward pipeline, same plant, same failure surface),
so multiple diagnosis arms can branch from one frozen failure realization —
this both cuts API cost and removes forward-pipeline noise as a confound.

Resume: ``build_resume_state`` restores that state (rebuilding non-JSON
pieces such as the knowledge-loader instance) and ``create_mako_graph`` is
compiled with entry ``diagnosis_agent``, so only the treatment path (rank,
probes, backward repairs, regeneration, re-solve) runs live.

Provenance: the snapshot manifest records provider/model/temperature, git
commit, plant, failure surface, and the forward token baseline; resumed runs
record ``resumed_from`` / ``snapshot_dir`` / ``snapshot_forward_tokens`` in
``run_manifest.json`` (see ``utils/run_log.build_run_manifest``).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple

from ..schemas import DataEngineerOutput, ModelExpertOutput
from ..utils.run_log import _jsonable, get_git_commit, utc_now_iso

logger = logging.getLogger(__name__)

SNAPSHOT_STATE_FILE = "snapshot_state.json"
SNAPSHOT_MANIFEST_FILE = "snapshot_manifest.json"

# Runtime instances that cannot round-trip through JSON: dropped at
# serialization and rebuilt in ``_restore_state``.
_DROPPED_KEYS = ("knowledge_loader", "snapshot_written")

# Episode-scoped leftovers that must not leak across the freeze boundary.
_EPISODE_TRANSIENT_KEYS = (
    "error_agent",
    "error_resolved",
    "backward_reason",
    "attributed_layer",
    "episode_payoff",
    "snapshot_dir",
    "snapshot_written",
)


def _serialize_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """JSON-safe copy of the failure blackboard.

    The two pydantic agent outputs serialize through ``model_dump`` (via
    ``_jsonable``); everything else in ``AgentState`` is already JSON-safe.
    """
    return {
        key: _jsonable(value)
        for key, value in state.items()
        if key not in _DROPPED_KEYS
    }


def write_failure_snapshot(
    state: Dict[str, Any],
    snapshot_dir: str,
    *,
    provenance: Optional[Dict[str, Any]] = None,
) -> Path:
    """Freeze the failure blackboard to ``snapshot_dir``.

    Raises ``ValueError`` when the state is not a failed solve — a snapshot
    of a success or a mid-forward state is meaningless for arm branching.
    """
    execution_result = state.get("execution_result") or {}
    if not execution_result.get("diagnosis_required"):
        raise ValueError(
            "snapshot requires a failed solve (execution_result.diagnosis_required=True)"
        )

    out = Path(snapshot_dir)
    out.mkdir(parents=True, exist_ok=True)

    state_path = out / SNAPSHOT_STATE_FILE
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(_serialize_state(state), f, ensure_ascii=False, indent=2, default=str)

    manifest = {
        "schema_version": 1,
        "created_at": utc_now_iso(),
        "git_commit": get_git_commit(),
        "config": {
            "provider": state.get("provider"),
            "model": state.get("model"),
            "temperature": state.get("temperature"),
            "knowledge_injection_mode": state.get("knowledge_injection_mode"),
            "inject_id": state.get("inject_id"),
            "fault_plant_id": state.get("fault_plant_id"),
            "true_root_cause": state.get("true_root_cause"),
        },
        "failure_surface": {
            "gurobi_status": state.get("gurobi_status") or "",
            "error_category": state.get("error_category") or "",
        },
        "forward_totals": {
            "total_tokens": int(state.get("total_tokens") or 0),
            "total_duration_s": round(float(state.get("total_duration_s") or 0.0), 3),
            "step_events": len(state.get("step_metrics") or []),
        },
        "freeze_run": {
            "run_id": state.get("run_id"),
            "result_dir": state.get("result_dir"),
        },
        "provenance": dict(provenance or {}),
    }
    manifest_path = out / SNAPSHOT_MANIFEST_FILE
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, default=str)

    logger.info(
        "Failure snapshot written: %s (state: %s)", manifest_path, state_path
    )
    return manifest_path


def load_snapshot(snapshot_dir: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Read back (snapshot_state, snapshot_manifest) from a freeze dir."""
    base = Path(snapshot_dir)
    with open(base / SNAPSHOT_STATE_FILE, encoding="utf-8") as f:
        state = json.load(f)
    with open(base / SNAPSHOT_MANIFEST_FILE, encoding="utf-8") as f:
        manifest = json.load(f)
    return state, manifest


def read_snapshot_manifest(snapshot_dir: str) -> Dict[str, Any]:
    with open(Path(snapshot_dir) / SNAPSHOT_MANIFEST_FILE, encoding="utf-8") as f:
        return json.load(f)


def _restore_state(snapshot_state: Dict[str, Any]) -> Dict[str, Any]:
    """Rebuild runtime objects the JSON dump cannot carry."""
    state = dict(snapshot_state)

    de = state.get("data_engineer_output")
    if isinstance(de, dict):
        state["data_engineer_output"] = DataEngineerOutput.model_validate(de)
    me = state.get("model_expert_output")
    if isinstance(me, dict):
        state["model_expert_output"] = ModelExpertOutput.model_validate(me)

    # Rebuild the knowledge plug-in instance from its serialized mode fields;
    # loaded modules / catalog text themselves live in plain state keys.
    from ..knowledge.progressive import ProgressiveKnowledgeInjection

    mode = state.get("knowledge_injection_mode") or "progressive"
    excluded = state.get("knowledge_excluded_modules") or []
    plugin = ProgressiveKnowledgeInjection.from_mode(mode, excluded_modules=excluded)
    state["knowledge_loader"] = plugin.loader if plugin.enabled else None

    return state


def build_resume_state(
    snapshot_dir: str,
    *,
    diagnosis_mode: Optional[str] = None,
    probe_order: Optional[str] = None,
    probe_seed: Optional[int] = None,
    omega_source: Optional[str] = None,
    rank_method: Optional[str] = None,
    max_retries: Optional[int] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    result_dir: Optional[str] = None,
    run_id: Optional[str] = None,
    log_prompts: bool = False,
    knowledge_excluded_modules: Optional[list] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Restore a frozen failure blackboard with diagnosis-side overrides.

    Every non-``None`` override replaces the snapshot value outright.
    ``provider`` / ``model`` / ``temperature`` follow the caller's defaults
    (CLI defaults match the usual provider); pass them explicitly to pin a
    snapshot's values. Returns ``(resume_state, snapshot_manifest)``.
    """
    snapshot_state, manifest = load_snapshot(snapshot_dir)
    state = _restore_state(snapshot_state)

    overrides = (
        ("diagnosis_mode", diagnosis_mode),
        ("probe_order", probe_order),
        ("probe_seed", probe_seed),
        ("omega_source", omega_source),
        ("rank_method", rank_method),
        ("max_retries", max_retries),
        ("provider", provider),
        ("model", model),
        ("temperature", temperature),
    )
    for key, value in overrides:
        if value is not None:
            state[key] = value
    if knowledge_excluded_modules is not None:
        state["knowledge_excluded_modules"] = list(knowledge_excluded_modules)

    # Fresh run identity for the resumed (treatment-only) segment.
    if result_dir is not None:
        state["result_dir"] = result_dir
    if run_id is not None:
        state["run_id"] = run_id
    state["log_prompts"] = bool(log_prompts)

    for key in _EPISODE_TRANSIENT_KEYS:
        state.pop(key, None)

    return state, manifest


def snapshot_gate_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Graph node: freeze the failure blackboard on first entry, else pass."""
    snapshot_dir = state.get("snapshot_dir")
    if not snapshot_dir or state.get("snapshot_written"):
        return {}
    write_failure_snapshot(state, snapshot_dir)
    return {"snapshot_written": True}


def route_after_snapshot_gate(state: Dict[str, Any]) -> Literal["diagnosis_agent", "end"]:
    """Frozen runs end here; live runs continue into diagnosis unchanged."""
    if state.get("snapshot_written"):
        return "end"
    return "diagnosis_agent"
