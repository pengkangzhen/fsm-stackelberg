"""Stackelberg inspection-game utilities (payoffs, episode accounting, probe policy)."""

from .payoff import (
    DEFAULT_PAYOFF_CONFIG,
    PayoffConfig,
    compute_episode_payoffs,
    finalize_episode_payoffs,
    infer_attributed_layer,
    infer_commitment_diagnostics,
    is_strict_success,
)

__all__ = [
    "DEFAULT_PAYOFF_CONFIG",
    "PayoffConfig",
    "compute_episode_payoffs",
    "finalize_episode_payoffs",
    "infer_attributed_layer",
    "infer_commitment_diagnostics",
    "is_strict_success",
    "CAUSAL_LAYERS",
    "VERIFICATION_RULE",
    "build_probe_order",
    "get_candidate_agents",
    "next_unclear_layer",
]


def __getattr__(name: str):
    """Lazy-load inspection helpers (pulls agents/LangChain only when needed)."""
    if name in {
        "CAUSAL_LAYERS",
        "VERIFICATION_RULE",
        "build_probe_order",
        "get_candidate_agents",
        "next_unclear_layer",
    }:
        from . import inspection

        return getattr(inspection, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
