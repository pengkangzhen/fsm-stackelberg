"""Chain-of-Experts pipeline — core orchestration.

Migrated from CoE's original main.py. Runs the multi-round, multi-expert
collaboration loop: Conductor selects experts → experts comment → Reducer
generates final code.

Key adaptations for MAKO:
- Input format: {description, code_example} (code_example is a Gurobi template)
- No backward/reflection by default (enable_reflection=False)
- Uses MAKO's get_llm() for all LLM calls
"""

import logging

import numpy as np

from .comment import Comment
from .comment_pool import CommentPool
from .conductor import Conductor
from .reducer import Reducer
from .experts import (
    ModelingExpert,
    ProgrammingExpert,
    ParameterExtractor,
    ModelingKnowledgeSupplementExpert,
    TerminologyInterpreter,
    ProgrammingExampleProvider,
    CodeReviewer,
)
from .utils import extract_code_from_string

logger = logging.getLogger(__name__)


def chain_of_experts(
    problem: dict,
    max_collaborate_nums: int = 3,
    model: str | None = None,
    provider: str | None = None,
    enable_reflection: bool = False,
    max_trials: int = 1,
) -> str:
    """Run Chain-of-Experts pipeline.

    Args:
        problem: dict with keys 'description' and 'code_example'.
        max_collaborate_nums: Number of expert collaboration rounds.
        model: LLM model name.
        provider: LLM provider name.
        enable_reflection: Whether to enable backward reflection.
        max_trials: Max retry trials with reflection.

    Returns:
        Extracted Python code string.
    """
    all_experts = [
        TerminologyInterpreter(model=model, provider=provider),
        ParameterExtractor(model=model, provider=provider),
        ModelingExpert(model=model, provider=provider),
        ProgrammingExampleProvider(model=model, provider=provider),
        ProgrammingExpert(model=model, provider=provider),
        ModelingKnowledgeSupplementExpert(model=model, provider=provider),
        CodeReviewer(model=model, provider=provider),
    ]
    num_experts = len(all_experts)

    reducer = Reducer(model=model, provider=provider)
    comment_pool = CommentPool(
        all_experts,
        visible_matrix=np.ones((num_experts, num_experts)),
    )
    conductor = Conductor(model=model, provider=provider)
    expert_stack = []

    for _ in range(max_trials):
        for _ in range(max_collaborate_nums):
            next_expert = conductor.forward(problem, comment_pool, max_collaborate_nums)
            logger.info(f"Conductor chose expert: {next_expert.name}")

            comment_text = next_expert.forward(problem, comment_pool)
            logger.info(f"Expert {next_expert.name} gave comment ({len(comment_text)} chars)")

            comment_pool.add_comment(Comment(next_expert, comment_text))
            expert_stack.append(next_expert)

        # Reducer generates final code
        answer = reducer.forward(problem, comment_pool)
        code = extract_code_from_string(answer)

        if not enable_reflection:
            return code

        # Reflection phase (simplified — no evaluator backward for baseline)
        logger.info("Reflection enabled but evaluator backward is disabled for baseline.")
        return code

    return code
