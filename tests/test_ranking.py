"""Unit tests for evidence-informed omega alignment (no API)."""

from __future__ import annotations

import unittest

from fsm_stackelberg.agents.diagnosis_agent import CAUSAL_LAYERS
from fsm_stackelberg.game.ranking import align_omega, rank_layers, repair_rank
from fsm_stackelberg.schemas import (
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
                    expression="(= I 0)",
                    description="balance",
                ),
            ],
        ),
    )


class TestRepairRank(unittest.TestCase):
    def test_permutation_passthrough(self):
        r = ["model_expert", "python_developer", "data_engineer"]
        self.assertEqual(repair_rank(r), r)

    def test_appends_missing_in_causal_order(self):
        self.assertEqual(
            repair_rank(["python_developer"]),
            ["python_developer", "data_engineer", "model_expert"],
        )

    def test_dedup_and_invalid(self):
        self.assertEqual(
            repair_rank(["model_expert", "model_expert", "nope"]),
            ["model_expert", "data_engineer", "python_developer"],
        )


class TestAlignOmegaEvidenceRank(unittest.TestCase):
    def test_causal_rotates_to_rank_tip(self):
        omega = align_omega(
            probe_order="causal",
            rank=["python_developer", "model_expert", "data_engineer"],
            omega_source="evidence_rank",
        )
        self.assertEqual(omega[0], "python_developer")
        self.assertEqual(omega, ["python_developer", "data_engineer", "model_expert"])
        self.assertEqual(set(omega), set(CAUSAL_LAYERS))

    def test_causal_me_tip(self):
        omega = align_omega(
            probe_order="causal",
            rank=["model_expert", "data_engineer", "python_developer"],
            omega_source="evidence_rank",
        )
        self.assertEqual(
            omega,
            ["model_expert", "python_developer", "data_engineer"],
        )

    def test_reverse_ignores_rank(self):
        omega = align_omega(
            probe_order="reverse",
            rank=["data_engineer", "model_expert", "python_developer"],
            omega_source="evidence_rank",
        )
        self.assertEqual(omega, list(reversed(CAUSAL_LAYERS)))
        self.assertEqual(omega[0], "python_developer")

    def test_random_ignores_rank_and_is_seeded(self):
        a = align_omega(
            probe_order="random",
            rank=["data_engineer", "model_expert", "python_developer"],
            omega_source="evidence_rank",
            probe_seed=42,
        )
        b = align_omega(
            probe_order="random",
            rank=["python_developer", "model_expert", "data_engineer"],
            omega_source="evidence_rank",
            probe_seed=42,
        )
        self.assertEqual(a, b)
        self.assertEqual(set(a), set(CAUSAL_LAYERS))


class TestAlignOmegaStatusPrior(unittest.TestCase):
    def test_causal_uses_status_prior(self):
        omega = align_omega(
            probe_order="causal",
            rank=["python_developer", "model_expert", "data_engineer"],
            gurobi_status="INFEASIBLE",
            omega_source="status_prior",
        )
        # Prior(INFEASIBLE) tip = model_expert — rank tip ignored.
        self.assertEqual(omega[0], "model_expert")


class TestHeuristicForceZero(unittest.TestCase):
    def test_force_zero_in_me_tips_model_expert_despite_crash_traceback(self):
        state = {
            "gurobi_status": "",  # CRASH prior often tips PD
            "stack_trace": (
                "Traceback (most recent call last):\n"
                "  File \"code.py\", line 10\nNameError: x"
            ),
            "error_info": "NameError in python_developer",
            "python_code": "raise NameError('x')",
            "model_expert_output": _me_with_force_zero(),
        }
        out = rank_layers(state, method="heuristic")
        self.assertEqual(out.rank[0], "model_expert")
        self.assertTrue(out.raw.get("force_zero_sea_smell"))

    def test_force_zero_in_code_tips_model_expert(self):
        state = {
            "gurobi_status": "OPTIMAL",
            "stack_trace": "",
            "python_code": (
                "m.addConstr(y_in[(h, t)] == 0, name=f'force_zero_yin_{h}_{t}')\n"
                "m.addConstr(y_out[(h, t)] == 0, name=f'force_zero_yout_{h}_{t}')\n"
            ),
            "model_expert_output": None,
        }
        out = rank_layers(state, method="heuristic")
        self.assertEqual(out.rank[0], "model_expert")


if __name__ == "__main__":
    unittest.main()
