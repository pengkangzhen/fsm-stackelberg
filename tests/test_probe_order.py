"""Unit tests for probe-order commitment (stdlib unittest)."""

from __future__ import annotations

import unittest

from fsm_stackelberg.agents.diagnosis_agent import build_probe_order, CAUSAL_LAYERS
from fsm_stackelberg.game.payoff import finalize_episode_payoffs, infer_commitment_diagnostics


class TestBuildProbeOrder(unittest.TestCase):
    def test_causal_rotates_to_prior_seed_for_infeasible(self):
        omega = build_probe_order("INFEASIBLE", "causal")
        self.assertEqual(omega[0], "model_expert")
        self.assertEqual(set(omega), set(CAUSAL_LAYERS))

    def test_reverse_is_pure_reverse_no_prior_rotation(self):
        omega = build_probe_order("INFEASIBLE", "reverse")
        self.assertEqual(omega, list(reversed(CAUSAL_LAYERS)))
        # Must start at PD, not rotated to ME.
        self.assertEqual(omega[0], "python_developer")

    def test_random_reproducible_and_not_prior_forced(self):
        a = build_probe_order("INFEASIBLE", "random", probe_seed=42)
        b = build_probe_order("INFEASIBLE", "random", probe_seed=42)
        c = build_probe_order("INFEASIBLE", "random", probe_seed=99)
        self.assertEqual(a, b)
        self.assertEqual(set(a), set(CAUSAL_LAYERS))
        # With two seeds, at least one shuffle should differ from Prior-rotated causal.
        causal = build_probe_order("INFEASIBLE", "causal")
        self.assertTrue(a != causal or c != causal)


class TestCommitmentDiagnostics(unittest.TestCase):
    def test_first_probe_hit(self):
        state = {
            "execution_result": {"diagnosis_required": True},
            "true_root_cause": "model_expert",
            "inspection_policy": {
                "omega": ["model_expert", "python_developer", "data_engineer"],
            },
            "backtrack_history": [
                {"error_agent": "model_expert", "error_resolved": True},
            ],
            "retry_count": 1,
            "total_tokens": 0,
        }
        d = infer_commitment_diagnostics(state)
        self.assertTrue(d["first_probe_hit"])
        self.assertTrue(d["kill_hit"])
        self.assertEqual(d["first_probe_layer"], "model_expert")
        self.assertEqual(d["true_root_rank_in_omega"], 0)

    def test_finalize_includes_kill_fields(self):
        state = {
            "execution_result": {
                "diagnosis_required": False,
                "result": {"objective_value": 1.0},
            },
            "expected_value": 1.0,
            "retry_count": 1,
            "error_resolved": True,
            "error_agent": "python_developer",
            "true_root_cause": "model_expert",
            "inspection_policy": {
                "omega": ["model_expert", "python_developer", "data_engineer"],
            },
            "backtrack_history": [
                {"error_agent": "model_expert", "error_resolved": True},
                {"error_agent": "python_developer", "error_resolved": True},
            ],
            "total_tokens": 0,
        }
        payoff = finalize_episode_payoffs(state)
        self.assertEqual(payoff["attributed_layer"], "python_developer")
        self.assertFalse(payoff["attribution_hit"])
        self.assertTrue(payoff["first_probe_hit"])
        self.assertTrue(payoff["kill_hit"])


if __name__ == "__main__":
    unittest.main()
