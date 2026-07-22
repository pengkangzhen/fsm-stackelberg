"""Unit tests for Stackelberg episode payoffs (stdlib unittest; no pytest)."""

from __future__ import annotations

import unittest

from fsm_stackelberg.game.payoff import (
    PayoffConfig,
    compute_episode_payoffs,
    finalize_episode_payoffs,
    infer_attributed_layer,
    is_strict_success,
)


class TestComputeEpisodePayoffs(unittest.TestCase):
    def test_leader_success_no_cost(self):
        out = compute_episode_payoffs(
            success=True,
            attributed_layer=None,
            true_root_cause=None,
            probe_rounds=0,
            tokens=0,
        )
        self.assertEqual(out["S"], 1.0)
        self.assertEqual(out["u_L"], 1.0)
        self.assertIsNone(out["u_F"])
        self.assertIsNone(out["attribution_hit"])

    def test_follower_hit_with_costs(self):
        cfg = PayoffConfig(lambda_k=0.05, lambda_c=0.1, mu_k=0.05, mu_c=0.1, c0=10_000.0)
        out = compute_episode_payoffs(
            success=True,
            attributed_layer="model_expert",
            true_root_cause="model_expert",
            probe_rounds=2,
            tokens=5_000,
            config=cfg,
        )
        # u_F = 1 - 0.05*2 - 0.1*(5000/10000) = 1 - 0.1 - 0.05 = 0.85
        self.assertTrue(out["attribution_hit"])
        self.assertAlmostEqual(out["u_F"], 0.85)
        # u_L = 1 - 0.05*2 - 0.1*0.5 = 0.85
        self.assertAlmostEqual(out["u_L"], 0.85)

    def test_follower_miss(self):
        out = compute_episode_payoffs(
            success=False,
            attributed_layer="python_developer",
            true_root_cause="data_engineer",
            probe_rounds=3,
            tokens=0,
        )
        self.assertFalse(out["attribution_hit"])
        self.assertAlmostEqual(out["u_F"], -0.15)  # 0 - 0.05*3
        self.assertAlmostEqual(out["u_L"], -0.15)  # 0 - 0.05*3


class TestStateHelpers(unittest.TestCase):
    def test_strict_success_with_gap(self):
        state = {
            "execution_result": {
                "diagnosis_required": False,
                "result": {"objective_value": 100.0},
            },
            "expected_value": 100.0,
        }
        self.assertTrue(is_strict_success(state))

        state["execution_result"]["result"]["objective_value"] = 110.0
        self.assertFalse(is_strict_success(state))

    def test_infer_attributed_on_confirmed_repair(self):
        state = {
            "execution_result": {"diagnosis_required": False, "result": {"objective_value": 1.0}},
            "expected_value": 1.0,
            "retry_count": 2,
            "error_resolved": True,
            "error_agent": "model_expert",
        }
        self.assertEqual(infer_attributed_layer(state), "model_expert")

    def test_infer_none_on_first_pass_success(self):
        state = {
            "execution_result": {"diagnosis_required": False, "result": {}},
            "retry_count": 0,
        }
        self.assertIsNone(infer_attributed_layer(state))

    def test_finalize_from_state(self):
        state = {
            "execution_result": {
                "diagnosis_required": False,
                "result": {"objective_value": 50.0},
            },
            "expected_value": 50.0,
            "retry_count": 1,
            "error_resolved": True,
            "error_agent": "python_developer",
            "true_root_cause": "python_developer",
            "total_tokens": 0,
        }
        payoff = finalize_episode_payoffs(state)
        self.assertEqual(payoff["attributed_layer"], "python_developer")
        self.assertTrue(payoff["attribution_hit"])
        self.assertAlmostEqual(payoff["u_L"], 0.95)
        self.assertAlmostEqual(payoff["u_F"], 0.95)


if __name__ == "__main__":
    unittest.main()
