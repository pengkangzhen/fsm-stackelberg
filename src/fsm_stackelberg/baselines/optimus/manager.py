"""GroupChatManager — orchestrates OptiMUS agents.

Migrated from OptiMUS v0.2 agents/manager.py.
The Manager is an LLM-based orchestrator that selects which agent to call next,
generates a task for that agent, and manages the iterative problem-solving loop.

Architecture (from the paper):
1. Manager reviews the current state (conversation history + problem status)
2. Manager selects the next agent and generates a task description
3. The selected agent processes the task and updates the state
4. Repeat until solution is found or max iterations reached
"""

import json
import logging

from .agents import Agent

logger = logging.getLogger(__name__)

MANAGER_SYSTEM = """You are the manager of a team of optimization experts. \
Your team consists of:
- Formulator: Translates natural language into mathematical formulations
- Programmer: Writes and debugs Gurobi solver code
- Evaluator: Evaluates solutions for correctness

STRATEGY:
1. Call Formulator ONCE to define variables, constraints, and objective.
2. Then call Programmer to write code using the formulation.
3. If code fails, call Programmer again to debug (use the word "fix" or "error" in task).
4. If code succeeds, call Evaluator to verify.
5. Do NOT call Formulator more than twice — move to Programmer quickly.

Respond in JSON format:
{
    "next_agent": "Formulator|Programmer|Evaluator|DONE",
    "task": "Clear description of what the agent should do",
    "reasoning": "Why you chose this agent and task"
}

If the problem is fully solved (correct solution found), respond with "DONE".
"""

MANAGER_PROMPT = """Current problem state:

Problem: {background}

Variables defined: {variables_count}
Constraints status: {constraints_status}
Objective status: {objective_status}
Code generated: {has_code}
Execution status: {execution_status}
Solution status: {solution_status}
Error: {error_message}

Conversation history (last {history_len} messages):
{history}

Which agent should act next and what should they do?
"""


class GroupChatManager(Agent):
    """Manager agent that coordinates Formulator, Programmer, and Evaluator."""

    def __init__(self, agents: list[Agent], model=None, provider=None,
                 max_selections: int = 8):
        super().__init__(
            name="Manager",
            description="Orchestrates the optimization team.",
            model=model,
            provider=provider,
        )
        self.system_prompt = MANAGER_SYSTEM
        self.agents = {agent.name: agent for agent in agents}
        self.max_selections = max_selections

    def solve(self, state: dict) -> dict:
        """Run the iterative agent loop until solution is found or max iterations.

        Args:
            state: Initial problem state dict.

        Returns:
            Final state after the loop completes.
        """
        history = []

        for selection_idx in range(self.max_selections):
            logger.info(f"Manager: Selection {selection_idx + 1}/{self.max_selections}")

            # Build manager prompt from state
            prompt = self._build_prompt(state, history)

            # Get manager's decision
            reply = self.llm_call(prompt=prompt)

            # Parse manager's decision
            decision = self._parse_decision(reply)
            next_agent_name = decision.get("next_agent", "DONE")
            task = decision.get("task", "")
            reasoning = decision.get("reasoning", "")

            logger.info(f"Manager chose: {next_agent_name} — Task: {task}")
            logger.info(f"Reasoning: {reasoning}")

            # Check if done
            if next_agent_name == "DONE":
                logger.info("Manager declared DONE")
                break

            # Get the selected agent
            agent = self.agents.get(next_agent_name)
            if agent is None:
                logger.warning(f"Manager chose unknown agent: {next_agent_name}")
                break

            # Agent generates reply and updates state
            agent_reply, state = agent.generate_reply(task=task, state=state, sender=self)

            # Record in history
            history.append({
                "agent": next_agent_name,
                "task": task,
                "reply_summary": agent_reply[:500] if agent_reply else "",
            })

            # If programmer generated code, execute it
            if next_agent_name == "Programmer" and state.get("code"):
                state = self._execute_code(state)

            logger.info(f"State after {next_agent_name}: "
                       f"solution_status={state.get('solution_status')}, "
                       f"has_code={bool(state.get('code'))}")

        # Return final state
        state["history"] = history
        return state

    def _build_prompt(self, state: dict, history: list) -> str:
        """Build the manager prompt from current state."""
        constraints = state.get("constraint", [])
        objective = state.get("objective", [])

        constraints_status = []
        for c in constraints:
            desc = c.get("description", "")[:60] if isinstance(c, dict) else str(c)[:60]
            status = c.get("status", "unknown") if isinstance(c, dict) else "unknown"
            constraints_status.append(f"{desc}... [{status}]")

        objective_status = []
        for o in objective:
            desc = o.get("description", "")[:60] if isinstance(o, dict) else str(o)[:60]
            status = o.get("status", "unknown") if isinstance(o, dict) else "unknown"
            objective_status.append(f"{desc}... [{status}]")

        history_text = ""
        for h in history[-5:]:  # Last 5 messages
            history_text += f"\n[{h['agent']}]: Task: {h['task']}\nReply: {h['reply_summary'][:200]}\n"

        return MANAGER_PROMPT.format(
            background=state.get("background", "")[:500],
            variables_count=len(state.get("variables", [])),
            constraints_status="\n".join(constraints_status) if constraints_status else "none",
            objective_status="\n".join(objective_status) if objective_status else "none",
            has_code="yes" if state.get("code") else "no",
            execution_status=state.get("solver_output_status", "not executed"),
            solution_status=state.get("solution_status", "not started"),
            error_message=state.get("error_message", "none"),
            history_len=len(history),
            history=history_text,
        )

    def _parse_decision(self, reply: str) -> dict:
        """Parse manager's JSON decision from reply."""
        try:
            from .utils import parse_json_from_string
            return parse_json_from_string(reply)
        except (ValueError, json.JSONDecodeError):
            logger.warning("Manager: Could not parse decision, defaulting to DONE")
            return {"next_agent": "DONE", "task": "", "reasoning": "Parse error"}

    def _execute_code(self, state: dict) -> dict:
        """Execute generated code using MAKO's sandbox."""
        from fsm_stackelberg.agents.solver_executor import sandbox_exec_code

        code = state.get("code", "")
        data = state.get("data", {})

        if not code:
            state["error_message"] = "No code to execute"
            return state

        logger.info("Manager: Executing generated code in sandbox...")
        try:
            exec_report = sandbox_exec_code(data, code)

            state["execution_result"] = exec_report

            if exec_report.get("diagnosis_required") is False:
                result = exec_report.get("result", {})
                state["solver_output_status"] = exec_report.get("gurobi_status", "")
                state["obj_val"] = result.get("objective_value")
                state["solution_status"] = "optimal"
                state["error_message"] = None
                logger.info(f"Code execution successful! obj_val={state['obj_val']}")
            else:
                error_details = exec_report.get("error_details", {})
                state["solver_output_status"] = exec_report.get("gurobi_status", "")
                state["solution_status"] = "error"
                if isinstance(error_details, dict):
                    state["error_message"] = error_details.get("error_message", "Unknown error")
                else:
                    state["error_message"] = str(error_details)
                logger.warning(f"Code execution failed: {state['error_message']}")

        except Exception as e:
            state["solution_status"] = "error"
            state["error_message"] = str(e)
            logger.error(f"Code execution exception: {e}")

        return state
