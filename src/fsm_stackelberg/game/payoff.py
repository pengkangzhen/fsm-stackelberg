"""Numeric episode payoffs for the Stackelberg inspection game.

Analysis-mode payoffs (manuscript Phase 2): LLMs remain black-box best-
responders; the orchestrator records u_L / u_F on each finished episode so
trajectories are comparable across diagnosis modes and probe-order ablations.

    u_F = 1{â = a*} − λ_K · K − λ_C · (C / C_0)
    u_L = S − μ_K · K − μ_C · (C / C_0)

where S is strict success, â is the attributed layer, a* is ground-truth
root cause (optional until Phase 4 injection), K is probe rounds, and C is
token spend. Coefficients are design parameters, not claimed LLM utilities.

Exp-I kill criteria also records commitment-order diagnostics:
``first_probe_hit`` = 1{first probed layer = a*} (order ablation signal),
distinct from last-comply ``attribution_hit`` (which confounds cascade repairs).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


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
    """Infer â from the inspection trajectory (last successful comply).

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


def infer_commitment_diagnostics(state: Dict[str, Any]) -> Dict[str, Any]:
    """Commitment-order diagnostics for Exp-I kill criteria.

    ``first_probe_hit`` asks whether the committed ω probed a* first — the
    ablation signal — independent of whether a downstream cascade later
    "stole" last-comply credit.
    """
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


def compute_episode_payoffs(
    *,
    success: bool,
    attributed_layer: Optional[str],
    true_root_cause: Optional[str],
    probe_rounds: int,
    tokens: int,
    config: PayoffConfig = DEFAULT_PAYOFF_CONFIG,
    commitment: Optional[Dict[str, Any]] = None,
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

    out: Dict[str, Any] = {
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
    if commitment:
        out.update(commitment)
    return out


def finalize_episode_payoffs(
    state: Dict[str, Any],
    *,
    config: PayoffConfig = DEFAULT_PAYOFF_CONFIG,
) -> Dict[str, Any]:
    """Derive payoffs from a finished AgentState and return the payoff dict."""
    success = is_strict_success(state, gap_tol=config.gap_tol)
    attributed = infer_attributed_layer(state)
    commitment = infer_commitment_diagnostics(state)
    return compute_episode_payoffs(
        success=success,
        attributed_layer=attributed,
        true_root_cause=state.get("true_root_cause"),
        probe_rounds=int(state.get("retry_count") or 0),
        tokens=int(state.get("total_tokens") or 0),
        config=config,
        commitment=commitment,
    )
