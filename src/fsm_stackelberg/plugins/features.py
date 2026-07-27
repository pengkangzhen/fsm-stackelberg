"""Compose optional pipeline features without coupling their internals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

from ..knowledge.progressive import ProgressiveKnowledgeInjection


@dataclass(frozen=True)
class FeatureBundle:
    """Enabled optional features for one ``run_mako`` invocation.

    Attributes:
        knowledge: Progressive knowledge plug-in (catalog → request → load).
        diagnosis_mode: ``stackelberg`` | ``adversarial`` | ``sequential``.
        probe_order: Stackelberg commitment-order ablation knob.
        omega_source: ``evidence_rank`` (default) | ``status_prior``.
        rank_method: ``llm_rank`` | ``heuristic`` | ``hybrid``.
    """

    knowledge: ProgressiveKnowledgeInjection
    diagnosis_mode: str = "stackelberg"
    probe_order: str = "causal"
    omega_source: str = "evidence_rank"
    rank_method: str = "llm_rank"

    def bootstrap_state(self) -> Dict[str, Any]:
        """State fields owned by feature plug-ins (merged into AgentState)."""
        state = self.knowledge.bootstrap_state()
        state["diagnosis_mode"] = self.diagnosis_mode
        state["probe_order"] = self.probe_order
        state["omega_source"] = self.omega_source
        state["rank_method"] = self.rank_method
        # Stackelberg episode fields — empty until first failure.
        state.setdefault("inspection_policy", None)
        state.setdefault("probe_queue", [])
        state.setdefault("cleared_layers", [])
        state.setdefault("refutation_log", [])
        return state


def build_feature_bundle(
    *,
    knowledge_mode: str = "progressive",
    diagnosis_mode: str = "stackelberg",
    probe_order: str = "causal",
    omega_source: str = "evidence_rank",
    rank_method: str = "llm_rank",
    knowledge_excluded_modules: Optional[Iterable[str]] = None,
) -> FeatureBundle:
    """Factory used by ``run_mako`` / CLI."""
    knowledge = ProgressiveKnowledgeInjection.from_mode(
        knowledge_mode,
        excluded_modules=knowledge_excluded_modules,
    )
    return FeatureBundle(
        knowledge=knowledge,
        diagnosis_mode=diagnosis_mode,
        probe_order=probe_order,
        omega_source=omega_source,
        rank_method=rank_method,
    )
