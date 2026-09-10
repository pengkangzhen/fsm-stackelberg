"""Structured run logging: run_manifest.json + events.jsonl + artifacts.

Three layers (see HANDOVER / run-log plan):

1. ``run_manifest.json`` — static config + terminal summary (machine-readable tables)
2. ``events.jsonl`` — chronological per-node / LLM events
3. ``artifacts/`` — bulky agent outputs; manifest/events hold paths only

``workflow.log`` remains a human-readable mirror and is not the authority for tables.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Minimal provider → env map (avoid importing llm_config / LangChain at log time).
_PROVIDER_BASE_ENV = {
    "DeepSeek": "DEEPSEEK_BASE_URL",
    "OpenAI": "OPENAI_BASE_URL",
    "Anthropic": "ANTHROPIC_BASE_URL",
    "Gemini": "GEMINI_BASE_URL",
    "Qwen": "QWEN_BASE_URL",
    "OpenRouter": "OPENROUTER_BASE_URL",
    "MiniMax": "MINIMAX_ANTHROPIC_BASE_URL",
    "ZhipuAI": "ZHIPUAI_BASE_URL",
    "DashScope": "QWEN_BASE_URL",
    "DashScopeGLM": "DASHSCOPE_GLM_BASE_URL",
    "Moonshot": "MOONSHOT_BASE_URL",
    "MiMo": "MIMO_BASE_URL",
    "NVIDIA": "NVIDIA_BASE_URL",
}

# Module-level logger bound for the active run (thread-safe enough for one workflow).
_ACTIVE: Optional["RunLogger"] = None
_LOCK = threading.Lock()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6]


def get_git_commit() -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip() or None
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


def resolve_api_base(provider: Optional[str]) -> Optional[str]:
    """Return provider base URL from env (no API key)."""
    if not provider:
        return None
    base_var = _PROVIDER_BASE_ENV.get(provider)
    if not base_var:
        return None
    return os.environ.get(base_var) or None


def _jsonable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump()
        except Exception:
            return str(obj)
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(x) for x in obj]
    return str(obj)


def infer_commitment_fields(state: Dict[str, Any]) -> Dict[str, Any]:
    """Commitment-order diagnostics for Exp-I (also mirrored into episode_payoff)."""
    policy = state.get("inspection_policy") or {}
    omega: List[str] = list(policy.get("omega") or state.get("probe_queue") or [])
    true_root = state.get("true_root_cause")

    first_probe = omega[0] if omega else None
    history = state.get("backtrack_history") or []
    if history:
        for event in history:
            agent = event.get("error_agent") if isinstance(event, dict) else None
            if agent:
                first_probe = agent
                break

    true_root_rank: Optional[int] = None
    if true_root and omega and true_root in omega:
        true_root_rank = omega.index(true_root)

    first_probe_hit: Optional[bool] = None
    if true_root is not None and first_probe is not None:
        first_probe_hit = first_probe == true_root

    plant_complied = False
    if true_root:
        for event in history:
            if not isinstance(event, dict):
                continue
            if event.get("error_agent") == true_root and event.get("error_resolved"):
                plant_complied = True
                break
        if state.get("error_agent") == true_root and state.get("error_resolved"):
            plant_complied = True

    return {
        "committed_omega": omega,
        "first_probe_layer": first_probe,
        "first_probe_hit": first_probe_hit,
        "true_root_rank_in_omega": true_root_rank,
        "plant_layer_complied": plant_complied if true_root else None,
        "kill_hit": first_probe_hit,
    }


def first_symptom_surface(state: Dict[str, Any]) -> Dict[str, Any]:
    """First failure surface for CRASH vs OPTIMAL-gap stratification."""
    history = state.get("backtrack_history") or []
    if history and isinstance(history[0], dict):
        return {
            "gurobi_status": history[0].get("gurobi_status") or "",
            "error_category": history[0].get("error_category") or "",
        }
    return {
        "gurobi_status": state.get("gurobi_status") or "",
        "error_category": state.get("error_category") or "",
    }


class RunLogger:
    """Writes events.jsonl and artifacts under a result directory."""

    def __init__(
        self,
        result_dir: Union[str, Path],
        *,
        run_id: Optional[str] = None,
        provider: str = "",
        model: str = "",
        temperature: float = 0.0,
        log_prompts: bool = False,
        api_base: Optional[str] = None,
    ):
        self.result_dir = Path(result_dir)
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir = self.result_dir / "artifacts"
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.prompts_dir = self.result_dir / "prompts"
        self.run_id = run_id or new_run_id()
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.log_prompts = bool(log_prompts)
        self.api_base = api_base if api_base is not None else resolve_api_base(provider)
        self.started_at = utc_now_iso()
        self.events_path = self.result_dir / "events.jsonl"
        self._counter = 0
        self._lock = threading.Lock()
        # Truncate / create empty events file for this run.
        self.events_path.write_text("", encoding="utf-8")

    def emit(
        self,
        *,
        node: str,
        step_type: str,
        round_num: Optional[int] = None,
        duration_s: float = 0.0,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
        ok: bool = True,
        error_type: Optional[str] = None,
        error_message: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        artifact_refs: Optional[List[str]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        with self._lock:
            self._counter += 1
            event_id = f"e{self._counter:04d}"
            event: Dict[str, Any] = {
                "t": utc_now_iso(),
                "event_id": event_id,
                "run_id": self.run_id,
                "round": round_num,
                "node": node,
                "step_type": step_type,
                "provider": provider if provider is not None else self.provider,
                "model": model if model is not None else self.model,
                "duration_s": round(float(duration_s), 3) if duration_s is not None else 0.0,
                "prompt_tokens": int(prompt_tokens or 0),
                "completion_tokens": int(completion_tokens or 0),
                "total_tokens": int(total_tokens or 0),
                "ok": bool(ok),
            }
            if error_type:
                event["error_type"] = error_type
            if error_message:
                event["error_message"] = str(error_message)[:500]
            if artifact_refs:
                event["artifact_refs"] = artifact_refs
            if extra:
                event.update(_jsonable(extra))

            with open(self.events_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
            return event_id

    def save_artifact(
        self,
        name: str,
        data: Any,
        *,
        subdir: Optional[str] = None,
    ) -> str:
        """Write JSON (or text) artifact; return path relative to result_dir."""
        target_dir = self.artifacts_dir if not subdir else self.artifacts_dir / subdir
        target_dir.mkdir(parents=True, exist_ok=True)
        safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in name)
        if isinstance(data, str):
            path = target_dir / f"{safe}.txt"
            path.write_text(data, encoding="utf-8")
        else:
            path = target_dir / f"{safe}.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(_jsonable(data), f, indent=2, ensure_ascii=False, default=str)
        return str(path.relative_to(self.result_dir))

    def maybe_save_prompt(self, round_num: int, node: str, prompt_text: Optional[str]) -> Optional[str]:
        if not self.log_prompts or not prompt_text:
            return None
        self.prompts_dir.mkdir(parents=True, exist_ok=True)
        safe_node = "".join(c if c.isalnum() or c in "._-" else "_" for c in node)
        path = self.prompts_dir / f"round{round_num}_{safe_node}.txt"
        path.write_text(prompt_text, encoding="utf-8")
        return str(path.relative_to(self.result_dir))


def init_run_logger(
    result_dir: Union[str, Path],
    **kwargs: Any,
) -> RunLogger:
    global _ACTIVE
    logger = RunLogger(result_dir, **kwargs)
    with _LOCK:
        _ACTIVE = logger
    return logger


def get_run_logger() -> Optional[RunLogger]:
    return _ACTIVE


def clear_run_logger() -> None:
    global _ACTIVE
    with _LOCK:
        _ACTIVE = None


def logger_from_state(state: Optional[Dict[str, Any]]) -> Optional[RunLogger]:
    """Prefer active module logger; else reconstruct from state.result_dir."""
    active = get_run_logger()
    if active is not None:
        return active
    if not state:
        return None
    result_dir = state.get("result_dir")
    if not result_dir:
        return None
    return RunLogger(
        result_dir,
        run_id=state.get("run_id"),
        provider=state.get("provider") or "",
        model=state.get("model") or "",
        temperature=float(state.get("temperature") or 0.0),
        log_prompts=bool(state.get("log_prompts")),
    )


def record_step_event(
    state: Dict[str, Any],
    *,
    node: str,
    step_type: str,
    duration_s: float = 0.0,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    total_tokens: int = 0,
    ok: bool = True,
    error_type: Optional[str] = None,
    error_message: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
    artifact: Any = None,
    artifact_name: Optional[str] = None,
    prompt_text: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Append a step_metrics row and mirror it to events.jsonl when logging is active."""
    step_metrics = list(state.get("step_metrics") or [])
    metric = {
        "node": node,
        "step_type": step_type,
        "duration_s": round(float(duration_s), 3),
        "prompt_tokens": int(prompt_tokens or 0),
        "completion_tokens": int(completion_tokens or 0),
        "total_tokens": int(total_tokens or 0),
    }
    step_metrics.append(metric)

    run_logger = logger_from_state(state)
    artifact_refs: List[str] = []
    if run_logger is not None:
        round_num = int(state.get("current_round") or 1)
        if artifact is not None and artifact_name:
            artifact_refs.append(run_logger.save_artifact(artifact_name, artifact))
        prompt_ref = run_logger.maybe_save_prompt(round_num, node, prompt_text)
        if prompt_ref:
            artifact_refs.append(prompt_ref)
        event_id = run_logger.emit(
            node=node,
            step_type=step_type,
            round_num=round_num,
            duration_s=duration_s,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            ok=ok,
            error_type=error_type,
            error_message=error_message,
            provider=state.get("provider"),
            model=state.get("model"),
            artifact_refs=artifact_refs or None,
            extra=extra,
        )
        metric["event_id"] = event_id
        if artifact_refs:
            metric["artifact_refs"] = artifact_refs
    return step_metrics


def save_diagnosis_artifact(
    state: Dict[str, Any],
    *,
    diagnosis_mode: str,
    error_agent: Optional[str],
    confidence: float,
    reason: str,
    inspection_policy: Optional[Dict[str, Any]],
    probe_queue: List[str],
    cleared_layers: List[str],
    error_category: str,
    gurobi_status: str,
) -> Optional[str]:
    run_logger = logger_from_state(state)
    if run_logger is None:
        return None
    retry = int(state.get("retry_count") or 0) + 1
    payload = {
        "retry_count": retry,
        "diagnosis_mode": diagnosis_mode,
        "probed_agent": error_agent,
        "suspected_agent": error_agent,
        "confidence": confidence,
        "reason": reason,
        "committed_omega": (inspection_policy or {}).get("omega") or probe_queue,
        "inspection_policy": inspection_policy,
        "probe_queue": probe_queue,
        "cleared_layers": cleared_layers,
        "error_category": error_category,
        "gurobi_status": gurobi_status,
    }
    return run_logger.save_artifact(f"diagnosis_r{retry}", payload)


def save_backward_artifact(
    state: Dict[str, Any],
    *,
    agent: str,
    is_caused_by_you: bool,
    error_resolved: bool,
    reason: str,
    refined_present: bool = False,
) -> Optional[str]:
    run_logger = logger_from_state(state)
    if run_logger is None:
        return None
    round_num = int(state.get("current_round") or 1)
    action = "comply" if error_resolved else "deflect"
    payload = {
        "agent": agent,
        "round": round_num,
        "is_caused_by_you": is_caused_by_you,
        "error_resolved": error_resolved,
        "action": action,
        "reason": reason,
        "refined_present": refined_present,
        "gurobi_status": state.get("gurobi_status") or "",
        "error_category": state.get("error_category") or "",
    }
    return run_logger.save_artifact(f"backward_{agent}_r{round_num}", payload)


def save_inject_artifact(
    state: Dict[str, Any],
    *,
    plant_id: str,
    note: str = "",
    before: Any = None,
    after: Any = None,
) -> Optional[str]:
    """Structured inject event artifact (no-op if run logger inactive)."""
    run_logger = logger_from_state(state)
    if run_logger is None:
        return None
    payload = {
        "plant_id": plant_id,
        "note": note,
        "timestamp": utc_now_iso(),
        "before": before,
        "after": after,
    }
    ref = run_logger.save_artifact(f"inject_{plant_id}", payload)
    run_logger.emit(
        node="fault_injector",
        step_type="inject",
        round_num=int(state.get("current_round") or 1),
        ok=True,
        artifact_refs=[ref],
        extra={"plant_id": plant_id, "note": note},
    )
    return ref


def gap_percent(obj_value: Any, expected_value: Any) -> Optional[float]:
    if obj_value is None or expected_value in (None, 0):
        return None
    try:
        return round(abs(float(obj_value) - float(expected_value)) / abs(float(expected_value)) * 100, 4)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def build_run_manifest(
    *,
    exp: Any,
    state: Optional[Dict[str, Any]] = None,
    run_id: str,
    started_at: str,
    ended_at: Optional[str] = None,
    temperature: float = 0.0,
    api_base: Optional[str] = None,
    probe_seed: Optional[int] = None,
    inject_id: Optional[str] = None,
    knowledge_max_rounds: Optional[int] = None,
    git_commit: Optional[str] = None,
    events_path: str = "events.jsonl",
    artifacts_dir: str = "artifacts",
    resumed_from: Optional[str] = None,
    snapshot_dir: Optional[str] = None,
    snapshot_forward_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """Flatten ExperimentResult + payoff + commitment diagnostics into run_manifest."""
    state = state or {}
    episode_payoff = dict(getattr(exp, "episode_payoff", None) or state.get("episode_payoff") or {})
    commitment = infer_commitment_fields(state)
    # Prefer payoff-embedded commitment fields when present.
    for key, value in commitment.items():
        episode_payoff.setdefault(key, value)

    obj_value = getattr(exp, "obj_value", None)
    expected_value = getattr(exp, "expected_value", None)
    gp = gap_percent(obj_value, expected_value)
    gurobi_status = getattr(exp, "gurobi_status", "") or ""
    solver_optimal = (state.get("execution_result") or {}).get("diagnosis_required") is False
    within_gap = bool(getattr(exp, "status", False))
    symptom = first_symptom_surface(state)

    inject = (
        inject_id
        or state.get("fault_plant_id")
        or state.get("inject_id")
        or getattr(exp, "inject_id", None)
        or ""
    )
    seed = probe_seed if probe_seed is not None else state.get("probe_seed")

    return {
        "schema_version": 1,
        "run_id": run_id,
        "timestamp_start": started_at,
        "timestamp_end": ended_at or utc_now_iso(),
        "git_commit": git_commit if git_commit is not None else get_git_commit(),
        "command": getattr(exp, "command", None) or (
            exp._build_command() if hasattr(exp, "_build_command") else ""
        ),
        "config": {
            "algorithm": getattr(exp, "algorithm", ""),
            "provider": getattr(exp, "provider", ""),
            "model": getattr(exp, "model", ""),
            "temperature": temperature,
            "api_base": api_base if api_base is not None else resolve_api_base(getattr(exp, "provider", "")),
            "dataset": getattr(exp, "dataset", ""),
            "prob_name": getattr(exp, "prob_name", ""),
            "expected_value": expected_value,
            "diagnosis_mode": getattr(exp, "orchestrator_mode", ""),
            "probe_order": getattr(exp, "probe_order", "") or state.get("probe_order") or "",
            "probe_seed": seed,
            "max_retries": getattr(exp, "max_collaborate_nums", 0),
            "knowledge_mode": getattr(exp, "knowledge_mode", ""),
            "knowledge_max_rounds": knowledge_max_rounds
            if knowledge_max_rounds is not None
            else state.get("knowledge_max_rounds"),
            "inject_id": inject or None,
            "true_root_cause": getattr(exp, "true_root_cause", None)
            or state.get("true_root_cause")
            or None,
            # Snapshot freeze/resume provenance (None on ordinary runs)
            "resumed_from": resumed_from,
            "snapshot_dir": snapshot_dir,
            "snapshot_forward_tokens": snapshot_forward_tokens,
        },
        "outcome": {
            "gurobi_status": gurobi_status,
            "obj_value": obj_value,
            "gap_percent": gp,
            "solver_optimal": solver_optimal,
            "within_gap": within_gap,
            "practical_optimal": bool(within_gap and solver_optimal),
            "status": bool(getattr(exp, "status", False)),
            "error_category": getattr(exp, "error_category", "none"),
            "error_agent": getattr(exp, "error_agent", "") or None,
            "first_symptom": symptom,
        },
        "cost": {
            "total_duration_s": round(float(getattr(exp, "total_duration_s", 0.0) or 0.0), 3),
            "prompt_tokens": int(getattr(exp, "total_prompt_tokens", 0) or 0),
            "completion_tokens": int(getattr(exp, "total_completion_tokens", 0) or 0),
            "total_tokens": int(getattr(exp, "total_tokens", 0) or 0),
            "num_forward_steps": int(getattr(exp, "num_forward_steps", 0) or 0),
            "num_backward_steps": int(getattr(exp, "num_backward_steps", 0) or 0),
            "probe_rounds": episode_payoff.get("probe_rounds", state.get("retry_count", 0)),
            "retry_count": int(state.get("retry_count") or 0),
        },
        "inspection": {
            "committed_omega": episode_payoff.get("committed_omega", commitment["committed_omega"]),
            "first_probe_layer": episode_payoff.get("first_probe_layer", commitment["first_probe_layer"]),
            "first_probe_hit": episode_payoff.get("first_probe_hit", commitment["first_probe_hit"]),
            "true_root_rank_in_omega": episode_payoff.get(
                "true_root_rank_in_omega", commitment["true_root_rank_in_omega"]
            ),
            "attributed_layer": getattr(exp, "attributed_layer", None)
            or episode_payoff.get("attributed_layer"),
            "attribution_hit": episode_payoff.get("attribution_hit"),
            "plant_layer_complied": episode_payoff.get(
                "plant_layer_complied", commitment["plant_layer_complied"]
            ),
            "kill_hit": episode_payoff.get("kill_hit", commitment["kill_hit"]),
            "episode_payoff": episode_payoff,
        },
        "paths": {
            "result_dir": getattr(exp, "result_path", "") or "",
            "events": events_path,
            "artifacts": artifacts_dir,
            "experiment_result": "experiment_result.json",
            "workflow_log": "workflow.log",
        },
        "agent_metrics": getattr(exp, "agent_metrics", []) or [],
        "node_metrics": getattr(exp, "node_metrics", []) or [],
    }


def save_run_manifest(result_dir: Union[str, Path], manifest: Dict[str, Any]) -> Path:
    path = Path(result_dir) / "run_manifest.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False, default=str)
    return path


def synthesize_events_from_steps(
    result_dir: Union[str, Path],
    steps: List[Dict[str, Any]],
    *,
    run_id: str,
    provider: str = "",
    model: str = "",
) -> Path:
    """Fallback: write events.jsonl from step_metrics when live emission was unavailable."""
    path = Path(result_dir) / "events.jsonl"
    if path.exists() and path.stat().st_size > 0:
        return path
    lines = []
    for i, step in enumerate(steps or [], start=1):
        event = {
            "t": utc_now_iso(),
            "event_id": f"e{i:04d}",
            "run_id": run_id,
            "round": step.get("round"),
            "node": step.get("node"),
            "step_type": step.get("step_type"),
            "provider": provider,
            "model": model,
            "duration_s": step.get("duration_s", 0),
            "prompt_tokens": step.get("prompt_tokens", 0),
            "completion_tokens": step.get("completion_tokens", 0),
            "total_tokens": step.get("total_tokens", 0),
            "ok": True,
            "synthesized": True,
        }
        lines.append(json.dumps(event, ensure_ascii=False))
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return path
