"""Bounded retry for structured-output chains that return ``None``.

Function-calling structured output returns ``None`` when the model answers
in plain text instead of calling the provided tool (observed with
glm-5.3-flash's always-on thinking on 2026-09-11: the ModelExpert forward
call skipped the tool and the node crashed on ``result.knowledge_requests``).
All agent call sites route through :func:`invoke_structured` so a ``None``
triggers ONE retry with an explicit call-the-tool reminder appended to the
task, and a clear error if the model still refuses.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

_TOOL_CALL_REMINDER = (
    "\n\nOUTPUT CONTRACT (mandatory): respond ONLY by calling the provided "
    "function/tool with the complete structured payload. Do not answer in "
    "plain text and do not leave the tool call empty."
)


def invoke_structured(chain, payload: Dict[str, Any], *, node: str = "agent"):
    """Invoke a structured-output chain, retrying once when it returns None.

    Args:
        chain: ``prompt | llm.with_structured_output(...)`` runnable.
        payload: The chain input (expects a ``{"role", "task"}`` dict; the
            reminder is appended to the first string field found).
        node: Caller name for log/error messages.

    Returns:
        The parsed structured output (not ``None``).

    Raises:
        ValueError: If the model returns no tool call twice in a row.
    """
    result = chain.invoke(payload)
    if result is not None:
        return result

    logger.warning(
        "%s: structured output returned None (no tool call); retrying once "
        "with an explicit tool-call reminder",
        node,
    )
    retry_payload = payload
    if isinstance(payload, dict):
        retry_payload = dict(payload)
        for key in ("task", "query", "input"):
            if isinstance(retry_payload.get(key), str):
                retry_payload[key] = retry_payload[key] + _TOOL_CALL_REMINDER
                break

    result = chain.invoke(retry_payload)
    if result is not None:
        return result

    raise ValueError(
        f"{node}: model returned no tool call after 2 attempts; "
        "structured output unavailable for this prompt"
    )
