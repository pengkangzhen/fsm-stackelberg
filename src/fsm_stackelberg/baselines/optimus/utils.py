"""Utility functions for OptiMUS baseline."""

import ast
import json
import re


def parse_json_from_string(text: str):
    """Parse JSON from a string that may contain code blocks or extra text."""
    if text is None:
        raise ValueError("Input text is None")

    raw = str(text).strip()
    candidates = [raw]

    # Extract from code blocks
    code = _extract_code_block(raw)
    if code and code not in candidates:
        candidates.append(code)

    # Extract first JSON blob
    blob = _extract_first_json_blob(raw)
    if blob and blob not in candidates:
        candidates.append(blob)

    blob = _extract_first_json_blob(code)
    if blob and blob not in candidates:
        candidates.append(blob)

    for candidate in candidates:
        for parser in (json.loads, ast.literal_eval):
            try:
                return parser(candidate)
            except (ValueError, SyntaxError, json.JSONDecodeError):
                continue

    raise ValueError(f"Cannot parse JSON from text: {raw[:200]}")


def _extract_code_block(text: str) -> str:
    """Extract code from markdown code blocks."""
    pattern = r"```(?:python|json)?\s*(.*?)\s*```"
    blocks = re.findall(pattern, text, re.DOTALL)
    if not blocks:
        return text
    if len(blocks) == 1:
        return blocks[0]
    return "\n".join(blocks)


def _extract_first_json_blob(text: str) -> str | None:
    """Extract the first balanced JSON object/array from text."""
    starts = [i for i, c in enumerate(text) if c in ["{", "["]]
    for start in starts:
        stack = []
        quote_char = None
        escaped = False
        for idx in range(start, len(text)):
            ch = text[idx]
            if quote_char is not None:
                if escaped:
                    escaped = False
                    continue
                if ch == "\\":
                    escaped = True
                    continue
                if ch == quote_char:
                    quote_char = None
                    continue
            if ch in ['"', "'"]:
                quote_char = ch
            elif ch in ["{", "["]:
                stack.append(ch)
            elif ch in ["}", "]"]:
                if not stack:
                    break
                left = stack.pop()
                if (left == "{" and ch != "}") or (left == "[" and ch != "]"):
                    break
            if not stack:
                return text[start : idx + 1]
    return None
