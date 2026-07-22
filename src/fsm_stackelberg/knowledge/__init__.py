"""知识库模块：文件扫描 + 渐进式注入策略。"""

from .knowledge_loader import KnowledgeLoader, AssembledKnowledge, KnowledgeModule
from .progressive import (
    DEFAULT_MAX_KNOWLEDGE_ROUNDS,
    ProgressiveKnowledgeConfig,
    ProgressiveKnowledgeInjection,
)

__all__ = [
    "KnowledgeLoader",
    "AssembledKnowledge",
    "KnowledgeModule",
    "DEFAULT_MAX_KNOWLEDGE_ROUNDS",
    "ProgressiveKnowledgeConfig",
    "ProgressiveKnowledgeInjection",
]
