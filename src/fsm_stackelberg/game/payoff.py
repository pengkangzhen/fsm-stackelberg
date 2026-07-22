"""Numeric episode payoffs for the Stackelberg inspection game.

Analysis-mode payoffs (manuscript Phase 2): LLMs remain black-box best-
responders; the orchestrator records u_L / u_F on each finished episode so
trajectories are comparable across diagnosis modes and probe-order ablations.

    u_F = 1{â = a*} − λ_K · K − λ_C · (C / C_0)
    u_L = S − μ_K · K − μ_C · (C / C_0)

where S is strict success, â is the attributed layer, a* is ground-truth
root cause (optional until Phase 4 injection), K is probe rounds, and C is
token spend. Coefficients are design parameters, not claimed LLM utilities.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class PayoffConfig:
    """Design parameters for episode utilities."""

    lambda_k: float = 0.05  # follower: cost per probe round
    lambda_c: float = 0.10  # follower: token-cost weight
    mu_k: float = 0.05  # leader: cost per probe round
    mu_c: float = 0.10  # leader: token-cost weight
    c0: float = 10_000.0  # token normalization scale
    gap_tol: float = 0.01  # relative gap for strict success (matches main.py)


DEFAULT_PAYOFF_CONFIG = PayoffConfig()


def is_strict_success(
    state: Dict[str, Any],
    *,
    expected_value: Optional[float] = None,
    gap_tol: float = DEFAULT_PAYOFF_CONFIG.gap_tol,
) -> bool:
    """OPTIMAL under diagnosis_required=False and optional expected-value gap."""
    execution_result = state.get("execution_result") or {}
    if execution_result.get("diagnosis_required") is not False:
        return False

    obj_value = None
    result = execution_result.get("result") or {}
    if isinstance(result, dict):
        obj_value = result.get("objective_value")

    expected = expected_value if expected_value is not None else state.get("expected_value")
    if obj_value is None or expected in (None, 0):
        # Solver path reported success but no GT to check — treat as success.
        return True
    return abs(float(obj_value) - float(expected)) / abs(float(expected)) <= gap_tol


def infer_attributed_layer(state: Dict[str, Any]) -> Optional[str]:
    """Infer â from the inspection trajectory.

    Confirmed attribution: a layer complied (error_resolved) and the subsequent
    re-solve passed strict success — that layer is the mechanism's root-cause
    claim. First-pass success with no diagnosis yields None (no inspection game).
    Failed / exhausted inspection yields None.
    """
    if not is_strict_success(state):
        return None

    if int(state.get("retry_count") or 0) <= 0:
        return None

    if state.get("error_resolved") and state.get("error_agent"):
        return state["error_agent"]

    # Success after diagnosis but missing flags: fall back to last probe event.
    history = state.get("backtrack_history") or []
    for event in reversed(history):
        agent = event.get("error_agent")
        if agent:
            return agent
    return None


def compute_episode_payoffs(
    *,
    success: bool,
    attributed_layer: Optional[str],
    true_root_cause: Optional[str],
    probe_rounds: int,
    tokens: int,
    config: PayoffConfig = DEFAULT_PAYOFF_CONFIG,
) -> Dict[str, Any]:
    """Compute numeric u_L / u_F for one diagnosis-repair episode."""
    K = max(0, int(probe_rounds))
    C = max(0, int(tokens))
    S = 1.0 if success else 0.0
    token_norm = C / config.c0 if config.c0 else 0.0

    cost_k_l = config.mu_k * K
    cost_c_l = config.mu_c * token_norm
    cost_k_f = config.lambda_k * K
    cost_c_f = config.lambda_c * token_norm

    u_L = S - cost_k_l - cost_c_l

    attribution_hit: Optional[bool] = None
    u_F: Optional[float] = None
    if true_root_cause is not None:
        attribution_hit = attributed_layer == true_root_cause
        A = 1.0 if attribution_hit else 0.0
        u_F = A - cost_k_f - cost_c_f

    return {
        "S": S,
        "attributed_layer": attributed_layer,
        "true_root_cause": true_root_cause,
        "attribution_hit": attribution_hit,
        "probe_rounds": K,
        "tokens": C,
        "token_norm": round(token_norm, 6),
        "u_L": round(u_L, 6),
        "u_F": None if u_F is None else round(u_F, 6),
        "costs": {
            "leader_round": round(cost_k_l, 6),
            "leader_token": round(cost_c_l, 6),
            "follower_round": round(cost_k_f, 6),
            "follower_token": round(cost_c_f, 6),
        },
        "config": asdict(config),
        "mode": "analysis",  # black-box best-responders; payoff is evaluative
    }


def finalize_episode_payoffs(
    state: Dict[str, Any],
    *,
    config: PayoffConfig = DEFAULT_PAYOFF_CONFIG,
) -> Dict[str, Any]:
    """Derive payoffs from a finished AgentState and return the payoff dict."""
    success = is_strict_success(state, gap_tol=config.gap_tol)
    attributed = infer_attributed_layer(state)
    return compute_episode_payoffs(
        success=success,
        attributed_layer=attributed,
        true_root_cause=state.get("true_root_cause"),
        probe_rounds=int(state.get("retry_count") or 0),
        tokens=int(state.get("total_tokens") or 0),
        config=config,
    )
