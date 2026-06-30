"""Formulator agent — formulates mathematical constraints and objectives.

Migrated from OptiMUS v0.2 agents/formulator.py.
Handles: formulating constraints, fixing formulations, formulating objectives.
"""

import json
import logging

from .agents import Agent

logger = logging.getLogger(__name__)

FORMULATOR_SYSTEM = """You are an expert in mathematical optimization formulation. \
Your job is to translate natural language constraints and objectives into precise \
mathematical formulations using standard notation (summation, indices, etc.).

You work with state that tracks which constraints and objectives have been formulated.
Each constraint/objective has a "status" field: "not_formulated", "formulated", or "error".

When formulating:
1. Identify all decision variables and their domains
2. Express constraints as mathematical inequalities/equalities
3. Express the objective function clearly
4. Ensure all indices, sets, and parameters are properly defined
"""

FORMULATE_TASK = """You are working on an optimization problem.

Problem description:
{background}

Parameters:
{parameters}

Current constraints status:
{constraints}

Current objective status:
{objective}

Your task: {task}

Respond with updated formulations. For each constraint/objective, provide:
- description: the natural language description
- formulation: the mathematical formulation
- code: Python code implementing this constraint/objective using Gurobi (gurobipy as gp)
- status: "formulated" if successful, "error" if there's an issue

Format your response as JSON:
{{
    "variables": [...list of variable definitions...],
    "constraints": [...updated constraint objects...],
    "objective": [...updated objective objects...]
}}
"""

FIX_FORMULATION_TASK = """You are working on an optimization problem.

Problem description:
{background}

Parameters:
{parameters}

The following formulation has an error:
{error_context}

Error message: {error_message}

Your task: Fix the formulation.

Provide the corrected formulation as JSON:
{{
    "variables": [...updated variable definitions...],
    "constraints": [...corrected constraint objects...],
    "objective": [...corrected objective objects...]
}}
"""


class Formulator(Agent):
    def __init__(self, model=None, provider=None):
        super().__init__(
            name="Formulator",
            description="Formulates mathematical constraints and objectives from natural language descriptions.",
            model=model,
            provider=provider,
        )
        self.system_prompt = FORMULATOR_SYSTEM

    def generate_reply(self, task: str, state: dict, sender: Agent) -> tuple[str, dict]:
        background = state.get("background", "")
        parameters = json.dumps(state.get("parameters", {}), indent=2, ensure_ascii=False)
        constraints = json.dumps(state.get("constraint", []), indent=2, ensure_ascii=False)
        objective = json.dumps(state.get("objective", []), indent=2, ensure_ascii=False)

        prompt = FORMULATE_TASK.format(
            background=background,
            parameters=parameters,
            constraints=constraints,
            objective=objective,
            task=task,
        )

        reply = self.llm_call(prompt=prompt)

        # Try to parse JSON from reply to update state
        try:
            from .utils import parse_json_from_string
            parsed = parse_json_from_string(reply)
            if isinstance(parsed, dict):
                if "variables" in parsed:
                    state["variables"] = parsed["variables"]
                if "constraints" in parsed:
                    state["constraint"] = parsed["constraints"]
                if "objective" in parsed:
                    state["objective"] = parsed["objective"]
        except (ValueError, json.JSONDecodeError):
            logger.warning("Formulator: Could not parse JSON from reply, returning raw text")

        return reply, state
