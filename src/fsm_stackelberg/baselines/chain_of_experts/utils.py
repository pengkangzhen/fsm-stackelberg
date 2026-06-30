"""Utility functions for Chain-of-Experts baseline."""

import re


def extract_code_from_string(input_string: str) -> str:
    """Extract Python code from markdown code blocks.

    Tries ```python ... ``` first, then falls back to returning the raw string.
    """
    pattern = r"```(?:python)?\s*(.*?)\s*```"
    code_blocks = re.findall(pattern, input_string, re.DOTALL)

    if len(code_blocks) == 0:
        return input_string
    if len(code_blocks) == 1:
        return code_blocks[0]

    # Filter out pip-install blocks
    code_blocks = [code for code in code_blocks if "pip" not in code]
    return "\n".join(code_blocks)
