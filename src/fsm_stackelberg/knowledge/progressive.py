"""Progressive (on-demand) knowledge injection.

Design (not RAG, not dump-all):
1. ModelExpert sees a *catalog* only (module name + short description).
2. It requests modules by name via ``knowledge_requests``.
3. KnowledgeLoader injects full markdown bodies for those names.
4. Control returns to ModelExpert for refinement (bounded rounds).

This module is intentionally independent of diagnosis / Stackelberg code so
the two features stay pluggable on the shared FSM.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from .knowledge_loader import KnowledgeLoader

logger = logging.getLogger(__name__)

# Successful load rounds (catalog → request → inject → refine).
DEFAULT_MAX_KNOWLEDGE_ROUNDS = 3


@dataclass(frozen=True)
class ProgressiveKnowledgeConfig:
    """Config for the progressive-knowledge feature plug-in."""

    enabled: bool = True
    max_rounds: int = DEFAULT_MAX_KNOWLEDGE_ROUNDS
    excluded_modules: tuple[str, ...] = ()


class ProgressiveKnowledgeInjection:
    """High-cohesion helper for catalog-first knowledge injection."""

    def __init__(
        self,
        config: ProgressiveKnowledgeConfig,
        loader: Optional[KnowledgeLoader] = None,
    ) -> None:
        self.config = config
        if not config.enabled:
            self.loader = None
        else:
            self.loader = loader if loader is not None else KnowledgeLoader()

    @classmethod
    def from_mode(
        cls,
        knowledge_mode: str,
        *,
        excluded_modules: Optional[Iterable[str]] = None,
        max_rounds: int = DEFAULT_MAX_KNOWLEDGE_ROUNDS,
    ) -> "ProgressiveKnowledgeInjection":
        """Build from CLI/workflow mode string.

        Accepted enabled aliases: ``progressive``, ``enable`` (legacy).
        Disabled: ``disable``.
        """
        mode = (knowledge_mode or "progressive").strip().lower()
        enabled = mode not in {"disable", "off", "none", "0", "false"}
        excluded = tuple(m for m in (excluded_modules or []) if m)
        return cls(
            ProgressiveKnowledgeConfig(
                enabled=enabled,
                max_rounds=max_rounds,
                excluded_modules=excluded,
            )
        )

    @property
    def enabled(self) -> bool:
        return self.config.enabled and self.loader is not None

    def bootstrap_state(self) -> Dict[str, Any]:
        """Initial AgentState fields: catalog only, no full-text preload."""
        if not self.enabled or self.loader is None:
            logger.info("Progressive knowledge injection disabled")
            return {
                "knowledge_loader": None,
                "knowledge_catalog": "",
                "knowledge_round": 0,
                "loaded_knowledge": None,
                "loaded_knowledge_modules": [],
                "knowledge_excluded_modules": list(self.config.excluded_modules),
                "knowledge_loader_loaded": False,
                "knowledge_max_rounds": self.config.max_rounds,
                "knowledge_injection_mode": "disable",
            }

        catalog = self.loader.get_knowledge_catalog_only(
            excluded_modules=self.config.excluded_modules,
        )
        logger.info(
            "Progressive knowledge: catalog-only bootstrap (%d modules visible)",
            len(self.loader.list_module_names(excluded_modules=self.config.excluded_modules)),
        )
        if self.config.excluded_modules:
            logger.info(
                "Knowledge ablation excluded modules: %s",
                list(self.config.excluded_modules),
            )

        return {
            "knowledge_loader": self.loader,
            "knowledge_catalog": catalog,
            "knowledge_round": 0,
            "loaded_knowledge": None,
            "loaded_knowledge_modules": [],
            "knowledge_excluded_modules": list(self.config.excluded_modules),
            "knowledge_loader_loaded": False,
            "knowledge_max_rounds": self.config.max_rounds,
            "knowledge_injection_mode": "progressive",
        }

    @staticmethod
    def format_loaded_section(loaded_knowledge: Optional[str]) -> str:
        """Prompt block for modules loaded after ModelExpert requests."""
        if not loaded_knowledge:
            return ""
        return f"""---

## Loaded Domain Knowledge (on demand)

The following modules were loaded because you (or a prior round) requested them
by name from the catalog. Apply these rules when defining variables and
constraints. You may request additional catalog modules via
``knowledge_requests`` if still needed.

{loaded_knowledge}

---"""

    @staticmethod
    def pending_requests(
        requested: Optional[List[str]],
        loaded_modules: Iterable[str],
        excluded_modules: Iterable[str],
    ) -> List[str]:
        loaded = set(loaded_modules or [])
        excluded = set(excluded_modules or [])
        if not requested:
            return []
        return [name for name in requested if name not in loaded and name not in excluded]
