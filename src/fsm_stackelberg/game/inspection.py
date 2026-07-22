"""Stackelberg inspection-game policy helpers (pluggable diagnosis feature).

Probe-order construction and verification-rule constants live with the
inspector implementation in ``agents.diagnosis_agent``; this module re-exports
the public surface so callers can depend on ``fsm_stackelberg.game`` without
importing agent nodes.
"""

from ..agents.diagnosis_agent import (
    CAUSAL_LAYERS,
    VERIFICATION_RULE,
    build_probe_order,
    get_candidate_agents,
    next_unclear_layer,
)

__all__ = [
    "CAUSAL_LAYERS",
    "VERIFICATION_RULE",
    "build_probe_order",
    "get_candidate_agents",
    "next_unclear_layer",
]
