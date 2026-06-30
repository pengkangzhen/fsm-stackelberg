"""Conductor — selects the next expert in the CoE collaboration loop.

Migrated from CoE's original conductor.py. Uses LLM to select which expert
to consult next, with a max_tokens constraint to keep responses short.
"""

import random
import logging

from langchain_core.prompts import PromptTemplate

from .experts.base_expert import BaseExpert
from fsm_stackelberg.utils.llm_config import get_llm

logger = logging.getLogger(__name__)


class Conductor(BaseExpert):
    ROLE_DESCRIPTION = "you will take on the role of the conductor for a multi-expert system."

    FORWARD_TASK = """Now, you are presented with an operational optimization-related problem:
{problem_description}

In this multi-expert system, there are many experts, each of whom is responsible for solving part of the problem.
Your task is to CHOOSE THE NEXT EXPERT TO CONSULT.
The names of the experts and their capabilities are listed below:
{experts_info}

Experts that have already been commented include:
{commented_experts}

Please select an expert to consult from the remaining expert names {remaining_experts}.
Please note that the maximum number of asked experts is {max_collaborate_nums}, and you can ask {remaining_collaborate_nums} more times.
You should output the name of expert directly. The next expert is:"""

    def __init__(self, model=None, provider=None):
        super().__init__(
            name="Conductor",
            description="A special expert that collaborates all other experts.",
            model=model,
            provider=provider,
        )
        # Conductor needs short responses — bind max_tokens
        self.llm = get_llm(provider=provider, model=model, temperature=0).bind(max_tokens=50)

    def forward(self, problem, comment_pool, max_collaborate_nums):
        all_experts = comment_pool.all_experts
        all_experts_name = [e.name for e in all_experts]
        commented_experts_name = [c.expert.name for c in comment_pool.comments]

        experts_info = "\n".join(str(e) for e in all_experts)
        commented_experts = str(commented_experts_name)
        remaining_experts = str(list(set(all_experts_name) - set(commented_experts_name)))

        answer = self._predict(
            self.forward_prompt,
            problem_description=problem["description"],
            experts_info=experts_info,
            commented_experts=commented_experts,
            remaining_experts=remaining_experts,
            max_collaborate_nums=str(max_collaborate_nums),
            remaining_collaborate_nums=str(max_collaborate_nums - len(commented_experts_name)),
        )

        expert_name_to_obj = {e.name: e for e in all_experts}
        for name, expert in expert_name_to_obj.items():
            if name in answer:
                return expert

        logger.warning("Conductor could not find expert in response, choosing randomly!")
        return random.choice(list(expert_name_to_obj.values()))
