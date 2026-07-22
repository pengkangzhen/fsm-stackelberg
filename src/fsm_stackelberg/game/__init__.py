"""Stackelberg inspection-game utilities (payoffs, episode accounting)."""

from .payoff import (
    DEFAULT_PAYOFF_CONFIG,
    PayoffConfig,
    compute_episode_payoffs,
    finalize_episode_payoffs,
    infer_attributed_layer,
    is_strict_success,
)

__all__ = [
    "DEFAULT_PAYOFF_CONFIG",
    "PayoffConfig",
    "compute_episode_payoffs",
    "finalize_episode_payoffs",
    "infer_attributed_layer",
    "is_strict_success",
]
