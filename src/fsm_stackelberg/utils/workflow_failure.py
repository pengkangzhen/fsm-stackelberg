"""Helpers for converting uncaught node exceptions into failure-shaped workflow state."""

from __future__ import annotations

import copy
import json
import time
import traceback

from .run_log import record_step_event


class WorkflowNodeError(RuntimeError):
    """Exception carrying a partial workflow state snapshot."""

    def __init__(self, message: str, state_snapshot: dict):
        super().__init__(message)
        self.state_snapshot = state_snapshot


def build_failure_state(
    state: dict,
    agent_name: str,
    exc: Exception,
    started_at: float,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    total_tokens: int = 0,
) -> dict:
    """Merge an agent exception into the accumulated workflow state."""
    duration = time.time() - started_at

    step_metrics = record_step_event(
        state,
        node=agent_name,
        step_type="forward",
        duration_s=duration,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        ok=False,
        error_type=type(exc).__name__,
        error_message=str(exc),
    )

    node_metrics = copy.deepcopy(state.get("node_metrics", {}))
    node_metric = node_metrics.setdefault(
        agent_name,
        {"total_duration_s": 0.0, "total_tokens": 0, "num_calls": 0},
    )
    node_metric["total_duration_s"] += duration
    node_metric["total_tokens"] += total_tokens
    node_metric["num_calls"] += 1

    agent_metrics = copy.deepcopy(state.get("agent_metrics", {}))
    if agent_name in {"data_engineer", "model_expert", "python_developer", "solver_executor"}:
        agent_metric = agent_metrics.setdefault(
            agent_name,
            {"total_duration_s": 0.0, "total_tokens": 0, "num_calls": 0},
        )
        agent_metric["total_duration_s"] += duration
        agent_metric["total_tokens"] += total_tokens
        agent_metric["num_calls"] += 1

    return {
        **state,
        "execution_result": {
            "diagnosis_required": True,
            "gurobi_status": "",
            "result": None,
        },
        "error_info": json.dumps(
            {
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "stack_trace": traceback.format_exc(),
            },
            ensure_ascii=False,
        ),
        "error_category": "execution",
        "error_agent": agent_name,
        "step_metrics": step_metrics,
        "node_metrics": node_metrics,
        "agent_metrics": agent_metrics,
        "total_tokens": state.get("total_tokens", 0) + total_tokens,
        "total_duration_s": state.get("total_duration_s", 0.0) + duration,
        "backtrack_history": state.get("backtrack_history", []),
        "output_history": state.get("output_history", []),
    }
