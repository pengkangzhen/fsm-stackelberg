"""BaseExpert — migrated from CoE's original base_expert.py.

Key migration changes:
- `from langchain import PromptTemplate, LLMChain` → `langchain_core.prompts.PromptTemplate`
- `from langchain.chat_models import ChatOpenAI` → uses MAKO's `get_llm()`
- `LLMChain.predict(**kwargs)` → `llm.invoke(prompt.format(**kwargs))`
"""

import logging

from langchain_core.prompts import PromptTemplate

from fsm_stackelberg.utils.llm_config import get_llm

logger = logging.getLogger(__name__)


class BaseExpert:
    """Base class for all CoE experts.

    Subclasses must define:
        ROLE_DESCRIPTION: str — expert role description
        FORWARD_TASK: str — forward task prompt template with {placeholders}
    Optionally:
        BACKWARD_TASK: str — backward (reflection) task prompt template
    """

    ROLE_DESCRIPTION: str = ""
    FORWARD_TASK: str = ""

    def __init__(self, name: str, description: str, model: str | None = None,
                 provider: str | None = None):
        self.name = name
        self.description = description
        self.model = model
        self.provider = provider
        self.llm = get_llm(provider=provider, model=model, temperature=0)

        # Build forward chain: prompt | llm
        forward_template = self.ROLE_DESCRIPTION + "\n" + self.FORWARD_TASK
        self.forward_prompt = PromptTemplate.from_template(forward_template)

        # Build backward chain if defined
        if hasattr(self, "BACKWARD_TASK") and self.BACKWARD_TASK:
            backward_template = self.ROLE_DESCRIPTION + "\n" + self.BACKWARD_TASK
            self.backward_prompt = PromptTemplate.from_template(backward_template)

    def _predict(self, prompt: PromptTemplate, **kwargs) -> str:
        """Format prompt template with kwargs and invoke LLM."""
        formatted = prompt.format(**kwargs)
        response = self.llm.invoke(formatted)
        return response.content

    def forward(self, problem, comment_pool):
        """Override in subclass."""
        raise NotImplementedError

    def backward(self, feedback_pool):
        """Override in subclass if reflection is enabled."""
        raise NotImplementedError

    def __str__(self):
        return f"{self.name}: {self.description}"
