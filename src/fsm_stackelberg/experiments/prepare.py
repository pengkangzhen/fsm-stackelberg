"""Shared problem preparation — unified interface for all experiment methods.

All methods (SPM, CoT, CoE, OptiMUS, MAKO) receive the same inputs:
- ``problem_description`` — natural language description
- ``data_access_guide`` — compact text from auto_preprocess (keys, types, sample values)
- ``preprocessed_data`` — tuple-keyed sparse dicts for sandbox execution

Usage::

    from fsm_stackelberg.experiments.prepare import prepare_problem

    description, processed_data, data_access_guide = prepare_problem(problem)
    # Pass data_access_guide to LLM prompts
    # Pass processed_data to sandbox_exec_code()
"""

from typing import Dict, Any, Tuple

from fsm_stackelberg.data.auto_preprocessor import auto_preprocess


def prepare_problem(problem: Dict[str, Any]) -> Tuple[str, Dict, str]:
    """Prepare problem data for any experiment method.

    Uses MAKO's ``auto_preprocess`` to convert raw sample data into
    tuple-keyed sparse dicts and generate a ``data_access_guide`` —
    the same treatment MAKO's own agents receive.

    Args:
        problem: Problem dict with keys:
            - description: str — natural language problem description
            - sample: dict — raw problem data (records format)
            - schema: dict — optional metadata

    Returns:
        (description, processed_data, data_access_guide)
        - description: the problem description
        - processed_data: preprocessed dict with tuple-keyed sparse dicts
        - data_access_guide: compact text summary for LLM prompts
    """
    description = problem.get("description", "")
    sample = problem.get("sample", {})
    processed_data, data_access_guide = auto_preprocess(sample)
    return description, processed_data, data_access_guide
