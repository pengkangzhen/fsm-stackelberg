"""Agent base class for OptiMUS baseline — adapted from agents/agent.py.

Original used raw OpenAI client; migrated to use MAKO's get_llm().
"""

import logging
from typing import Optional

from fsm_stackelberg.utils.llm_config import get_llm

logger = logging.getLogger(__name__)


class Agent:
    """Base agent class for OptiMUS.

    Each agent has a name, description, system prompt, and LLM.
    Subclasses override generate_reply() to implement agent-specific logic.
    """

    def __init__(self, name: str, description: str, model: str | None = None,
                 provider: str | None = None, max_tokens: int | None = None):
        self.name = name
        self.description = description
        self.model = model
        self.provider = provider
        self.system_prompt = "You are a helpful assistant."
        self.llm = get_llm(provider=provider, model=model, temperature=0,
                           max_tokens=max_tokens)

    def llm_call(self, prompt: Optional[str] = None,
                 messages: Optional[list] = None) -> str:
        """Call the LLM with a prompt or message list."""
        assert (prompt is None) != (messages is None), \
            "Provide exactly one of prompt or messages"

        if prompt is not None:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ]

        from langchain_core.messages import HumanMessage, SystemMessage
        lc_messages = []
        for msg in messages:
            if msg["role"] == "system":
                lc_messages.append(SystemMessage(content=msg["content"]))
            elif msg["role"] == "user":
                lc_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                from langchain_core.messages import AIMessage
                lc_messages.append(AIMessage(content=msg["content"]))

        response = self.llm.invoke(lc_messages)
        return response.content

    def generate_reply(self, task: str, state: dict, sender: "Agent") -> tuple[str, dict]:
        """Generate a reply given a task, current state, and the sender agent.

        Returns:
            (reply_text, updated_state)
        """
        return ("Reply not implemented. Terminate.", state)

    def __str__(self):
        return f"{self.name}: {self.description}"
