"""Stackelberg inspection-game utilities (payoffs, episode accounting, probe policy)."""

from .payoff import (
    DEFAULT_PAYOFF_CONFIG,
    PayoffConfig,
    compute_episode_payoffs,
    finalize_episode_payoffs,
    infer_attributed_layer,
    is_strict_success,
)
from .inspection import (
    CAUSAL_LAYERS,
    VERIFICATION_RULE,
    build_probe_order,
    get_candidate_agents,
    next_unclear_layer,
)

__all__ = [
    "DEFAULT_PAYOFF_CONFIG",
    "PayoffConfig",
    "compute_episode_payoffs",
    "finalize_episode_payoffs",
    "infer_attributed_layer",
    "is_strict_success",
    "CAUSAL_LAYERS",
    "VERIFICATION_RULE",
    "build_probe_order",
    "get_candidate_agents",
    "next_unclear_layer",
]
