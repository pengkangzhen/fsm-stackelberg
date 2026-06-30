"""Evaluator agent — evaluates generated code by executing it.

Migrated from OptiMUS v0.2 agents/evaluator.py.
Uses MAKO's sandbox_exec_code() for safe code execution.
"""

import json
import logging

from .agents import Agent

logger = logging.getLogger(__name__)

EVALUATOR_SYSTEM = """You are an optimization solution evaluator. \
You check whether a generated Gurobi solution is correct and feasible.

When given a solution result, you analyze:
1. Whether the model status is OPTIMAL
2. Whether the objective value is reasonable
3. Whether all constraints are satisfied
4. Whether the solution makes sense given the problem description
"""

EVALUATE_TASK = """Evaluate the following optimization solution.

Problem description:
{background}

Execution result:
{execution_result}

Objective value: {obj_val}
Solver status: {solver_status}

Is this solution correct? If there are issues, describe them.
If the solution is correct, respond with: SOLUTION_CORRECT
If there are issues, describe them and suggest what needs to be fixed.
"""


class Evaluator(Agent):
    def __init__(self, model=None, provider=None):
        super().__init__(
            name="Evaluator",
            description="Evaluates optimization solutions for correctness and feasibility.",
            model=model,
            provider=provider,
        )
        self.system_prompt = EVALUATOR_SYSTEM

    def generate_reply(self, task: str, state: dict, sender: Agent) -> tuple[str, dict]:
        background = state.get("background", "")
        execution_result = json.dumps(
            state.get("execution_result", {}), indent=2, ensure_ascii=False, default=str
        )
        obj_val = state.get("obj_val", "N/A")
        solver_status = state.get("solver_output_status", "N/A")

        prompt = EVALUATE_TASK.format(
            background=background,
            execution_result=execution_result,
            obj_val=obj_val,
            solver_status=solver_status,
        )

        reply = self.llm_call(prompt=prompt)

        # Check if solution is marked correct
        if "SOLUTION_CORRECT" in reply:
            state["solution_status"] = "correct"
        else:
            state["solution_status"] = "needs_fix"
            state["error_message"] = reply

        return reply, state
