"""Baseline algorithms for optimization problem solving.

This module contains baseline implementations for comparison with MAKO:
- SPM: Standard Prompting Method - single-shot code generation
- CoT: Chain-of-Thought - multi-stage reasoning
"""

from .spm import run_spm
from .cot import run_cot

__all__ = ["run_spm", "run_cot"]
