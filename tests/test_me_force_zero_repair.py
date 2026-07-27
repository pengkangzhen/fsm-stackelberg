"""Unit tests for ME force-zero repair hygiene (no API)."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from fsm_stackelberg.agents.model_expert import (
    model_expert_backward_step,
    strip_force_zero_sea_constraints,
)
from fsm_stackelberg.schemas import (
    BackwardStepOutput,
    Constraint,
    DecisionVariable,
    ModelComponents,
    ModelExpertOutput,
    ObjectiveFunction,
)


def _me_with_force_zero() -> ModelExpertOutput:
    return ModelExpertOutput(
        knowledge_requests=[],
        model_components=ModelComponents(
            decision_variables=[
                DecisionVariable(
                    symbol="y_in",
                    indices=["h", "t"],
                    shape=["hubs", "periods"],
                    type="Continuous",
                    description="in",
                ),
            ],
            objective_function=ObjectiveFunction(
                direction="min",
                expression="(+ cost)",
                description="min cost",
            ),
            constraints=[
                Constraint(
                    name="Injected_Force_Zero_Sea_Reposition",
                    expression=(
                        "(forall h hubs (forall t periods "
                        "(and (= y_in[h,t] 0) (= y_out[h,t] 0))))"
                    ),
                    description=(
                        "Force all stage-1 sea repositioning arrivals and "
                        "departures to zero for every hub and period."
                    ),
                ),
                Constraint(
                    name="hub_inventory_balance",
                    expression="(= I[h,t,k] I0)",
                    description="balance",
                ),
            ],
        ),
    )


class TestStripForceZero(unittest.TestCase):
    def test_strips_injected_constraint_keeps_others(self):
        me = _me_with_force_zero()
        cleaned, dropped = strip_force_zero_sea_constraints(me)
        self.assertEqual(dropped, ["Injected_Force_Zero_Sea_Reposition"])
        names = [c.name for c in cleaned.model_components.constraints]
        self.assertEqual(names, ["hub_inventory_balance"])

    def test_noop_when_absent(self):
        me = _me_with_force_zero()
        me = me.model_copy(
            update={
                "model_components": me.model_components.model_copy(
                    update={
                        "constraints": [
                            c
                            for c in me.model_components.constraints
                            if c.name != "Injected_Force_Zero_Sea_Reposition"
                        ]
                    }
                )
            }
        )
        cleaned, dropped = strip_force_zero_sea_constraints(me)
        self.assertEqual(dropped, [])
        self.assertEqual(len(cleaned.model_components.constraints), 1)

    def test_keeps_nonnegativity_gte_zero(self):
        """Regression: (>= y_in 0) must NOT be treated as force-zero."""
        me = _me_with_force_zero()
        cons = list(me.model_components.constraints)
        cons.append(
            Constraint(
                name="nonnegativity",
                expression=(
                    "(forall h hubs (forall t periods (>= y_in[h,t] 0))) "
                    "(forall h hubs (forall t periods (>= y_out[h,t] 0))) "
                    "(forall (i j m) arcs (forall t periods (forall k scenarios "
                    "(>= x[i,j,m,t,k] 0))))"
                ),
                description="All decision variables are nonnegative continuous",
            )
        )
        me = me.model_copy(
            update={
                "model_components": me.model_components.model_copy(
                    update={"constraints": cons}
                )
            }
        )
        cleaned, dropped = strip_force_zero_sea_constraints(me)
        self.assertEqual(dropped, ["Injected_Force_Zero_Sea_Reposition"])
        names = [c.name for c in cleaned.model_components.constraints]
        self.assertIn("nonnegativity", names)
        self.assertIn("hub_inventory_balance", names)


class TestMEBackwardMinimalStrip(unittest.TestCase):
    @patch("fsm_stackelberg.agents.model_expert.get_openai_callback")
    @patch("fsm_stackelberg.agents.model_expert.get_llm")
    def test_comply_with_force_zero_uses_minimal_strip(self, mock_get_llm, mock_cb):
        # LLM admits but "fixes" the wrong thing — we must prefer strip.
        wrong_refined = _me_with_force_zero().model_dump()
        # Drop plant in refined but also rewrite balance (simulating bad rewrite);
        # current blackboard still has plant → minimal strip of *current* wins.
        wrong_refined["model_components"]["constraints"] = [
            {
                "name": "hub_inventory_balance",
                "expression": "(= I broken)",
                "description": "bad rewrite",
            }
        ]
        llm_out = BackwardStepOutput(
            is_caused_by_you=True,
            reason="I rewrote sea_outbound_floor",
            refined_result=wrong_refined,
        )

        chain = MagicMock()
        chain.invoke.return_value = llm_out
        llm = MagicMock()
        llm.with_structured_output.return_value = MagicMock()
        # prompt | chain pattern: get_llm returns llm; we patch at format level
        mock_get_llm.return_value = llm
        # Make `prompt | llm.with_structured_output(...)` return our chain
        with patch(
            "fsm_stackelberg.agents.model_expert.ChatPromptTemplate"
        ) as mock_pt:
            mock_pt.from_messages.return_value = MagicMock(
                __or__=MagicMock(return_value=chain)
            )
            cb = MagicMock()
            cb.total_tokens = 10
            cb.prompt_tokens = 5
            cb.completion_tokens = 5
            mock_cb.return_value.__enter__.return_value = cb
            mock_cb.return_value.__exit__.return_value = False

            state = {
                "error_info": "OPTIMAL gap=70%",
                "gurobi_status": "OPTIMAL",
                "problem_description": "tslp",
                "data_engineer_output": None,
                "model_expert_output": _me_with_force_zero(),
                "output_history": [],
                "step_metrics": [],
                "node_metrics": {},
                "agent_metrics": {},
                "total_tokens": 0,
                "total_duration_s": 0.0,
                "current_round": 1,
                "provider": "DashScope",
                "model": "deepseek-v4-flash",
            }
            out = model_expert_backward_step(state)

        self.assertTrue(out["error_resolved"])
        me = out["model_expert_output"]
        names = [c.name for c in me.model_components.constraints]
        self.assertNotIn("Injected_Force_Zero_Sea_Reposition", names)
        # Minimal strip keeps original balance, not LLM's broken rewrite.
        self.assertEqual(names, ["hub_inventory_balance"])
        bal = me.model_components.constraints[0]
        self.assertEqual(bal.expression, "(= I[h,t,k] I0)")
        self.assertIn("minimal_strip", out["backward_reason"])


if __name__ == "__main__":
    unittest.main()
