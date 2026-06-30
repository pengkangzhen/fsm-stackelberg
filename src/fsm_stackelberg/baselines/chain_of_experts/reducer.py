"""Reducer — summarizes all expert comments into final code.

Migrated from CoE's original reducer.py. The Reducer is responsible for
taking all expert comments and producing the final Python code.
"""

from .experts.base_expert import BaseExpert


class Reducer(BaseExpert):
    ROLE_DESCRIPTION = (
        "You are an expert that responsible for summarize the comment of "
        "all other experts then conclude the final answer"
    )

    FORWARD_TASK = """Now, you are an expert of Operations Research.
You are supposed to give the final code of an problem.
Text description of the problem: {problem_description}

Your colleagues are all experts in various related fields. They have given their own insights. I hope you will carefully refer to these comments when giving the final code:
{comment_text}

You MUST write your solution as a complete Python function following this template:
{code_example}

Requirements:
1. The function MUST be named `optimize` and accept a single argument `data` (a dict containing all problem data).
2. Use `gurobipy` to formulate and solve the optimization model.
3. The function MUST return a dict with keys: "status" (str), "objective_value" (float), "variables" (dict).
4. No code is required outside the function except for the import package (No test code).

Your final code is as following:
"""

    def __init__(self, model=None, provider=None):
        super().__init__(
            name="Solver",
            description="Reduce all comments given by other experts",
            model=model,
            provider=provider,
        )

    def forward(self, problem, comment_pool):
        comment_text = comment_pool.get_current_comment_text()
        answer = self._predict(
            self.forward_prompt,
            problem_description=problem["description"],
            code_example=problem.get("code_example", ""),
            comment_text=comment_text,
        )
        return answer
