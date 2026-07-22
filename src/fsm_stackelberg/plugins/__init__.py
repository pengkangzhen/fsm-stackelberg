"""Pluggable feature bundles for the FSM pipeline.

Keep mako-inherited scaffolding (agents, graph) separate from optional
research features:

- progressive knowledge injection (``knowledge``)
- Stackelberg / baseline diagnosis modes (``game`` / diagnosis CLI)

Workflow code should depend on this facade rather than hard-wiring preload
heuristics or game internals into routing.
"""

from .features import FeatureBundle, build_feature_bundle

__all__ = ["FeatureBundle", "build_feature_bundle"]
