"""Programmer agent — writes and debugs optimization code.

Migrated from OptiMUS v0.2 agents/programmer.py (22KB, largest file).
The Programmer takes formulations and generates complete Gurobi solver code.
"""

import json
import logging
import re

from .agents import Agent

logger = logging.getLogger(__name__)

PROGRAMMER_SYSTEM = """You are an expert Python programmer specializing in optimization. \
You write clean, efficient Gurobi code to solve optimization problems.

IMPORTANT RULES:
1. Always use `import gurobipy as gp` and `from gurobipy import GRB`
2. The code must define an `optimize(data)` function
3. The optimize function takes a `data` dict with problem parameters
4. It must return: {"status": str, "objective_value": float, "variables": dict}
5. Do NOT wrap the entire function body in try/except — let errors propagate naturally
6. Use math, numpy, itertools, collections as needed (they are pre-imported)
"""

WRITE_CODE_TASK = """Write complete Gurobi solver code for this optimization problem.

Problem description:
{background}

The `data` dict passed to optimize():
{data_description}

Write a complete Python solution. The code must:
1. Define `def optimize(data):` as the entry point
2. Extract data using the keys described above
3. Create a Gurobi model
4. Define all decision variables with proper bounds and types
5. Add all constraints
6. Set the objective function
7. Optimize and return: {{"status": "OPTIMAL", "objective_value": m.ObjVal, "variables": {{}}}}
8. Do NOT use try/except — let errors propagate naturally

Variables: {variables}
Constraints: {constraints}
Objective: {objective}

Return the code in a single ```python``` block only, no explanation.
"""

DEBUG_CODE_TASK = """The following Gurobi code has an error.

Problem description:
{background}

The `data` dict passed to optimize():
{data_description}

Current code:
{code}

Error message:
{error_message}

Task: Fix the code. You MUST use the data keys described above. Do NOT invent keys.
Return the corrected complete code in a single ```python``` block only.
"""


class Programmer(Agent):
    def __init__(self, model=None, provider=None):
        super().__init__(
            name="Programmer",
            description="Writes and debugs Gurobi solver code for optimization problems.",
            model=model,
            provider=provider,
            max_tokens=8192,
        )
        self.system_prompt = PROGRAMMER_SYSTEM

    def generate_reply(self, task: str, state: dict, sender: Agent) -> tuple[str, dict]:
        background = state.get("background", "")
        data_description = state.get("data_description", "Not provided")
        parameters = json.dumps(state.get("parameters", {}), indent=2, ensure_ascii=False)
        variables = json.dumps(state.get("variables", []), indent=2, ensure_ascii=False)
        constraints = json.dumps(state.get("constraint", []), indent=2, ensure_ascii=False)
        objective = json.dumps(state.get("objective", []), indent=2, ensure_ascii=False)

        if "fix" in task.lower() or "debug" in task.lower() or "error" in task.lower():
            # Debug mode
            code = state.get("code", "")
            error_message = state.get("error_message", "")
            prompt = DEBUG_CODE_TASK.format(
                background=background,
                data_description=data_description,
                code=code,
                error_message=error_message,
            )
        else:
            # Write mode
            prompt = WRITE_CODE_TASK.format(
                background=background,
                data_description=data_description,
                parameters=parameters,
                variables=variables,
                constraints=constraints,
                objective=objective,
            )

        reply = self.llm_call(prompt=prompt)

        # Extract code from reply
        code = self._extract_code(reply)
        if code:
            state["code"] = code
            logger.info(f"Programmer: Generated/updated code ({len(code)} chars)")

        return reply, state

    @staticmethod
    def _extract_code(text: str) -> str:
        """Extract Python code from markdown or raw text."""
        pattern = r"```(?:python)?\s*(.*?)\s*```"
        code_blocks = re.findall(pattern, text, re.DOTALL)

        if code_blocks:
            # Find the block with def optimize
            for block in code_blocks:
                if "def optimize" in block:
                    return block.strip()
            # Return the longest block
            return max(code_blocks, key=len).strip()

        # No code blocks — check if the text itself is code
        if "def optimize" in text:
            return text.strip()

        return ""
