"""Unit tests for verified attribution from refutation logs (no API)."""

from __future__ import annotations

import unittest

from fsm_stackelberg.game.payoff import finalize_episode_payoffs
from fsm_stackelberg.game.verified import (
    infer_verified_attribution,
    make_refutation_entry,
)


def _success_state(**extra):
    base = {
        "execution_result": {
            "diagnosis_required": False,
            "result": {"objective_value": 1.0},
        },
        "expected_value": 1.0,
        "retry_count": 1,
        "error_resolved": True,
        "error_agent": "model_expert",
        "true_root_cause": "model_expert",
        "total_tokens": 0,
        "refutation_log": [],
        "inspection_policy": {
            "omega": ["model_expert", "python_developer", "data_engineer"],
            "omega_source": "evidence_rank",
            "rank": ["model_expert", "python_developer", "data_engineer"],
            "rank_method": "heuristic",
        },
        "backtrack_history": [
            {"error_agent": "model_expert", "error_resolved": True},
        ],
    }
    base.update(extra)
    return base


class TestVerifiedAttribution(unittest.TestCase):
    def test_comply_plus_practical_optimal_is_verified(self):
        state = _success_state()
        out = infer_verified_attribution(state)
        self.assertEqual(out["verified_attributed_layer"], "model_expert")
        self.assertTrue(out["verified_attribution_hit"])
        self.assertEqual(len(out["refutation_log"]), 1)
        self.assertTrue(out["refutation_log"][0]["re_solve_strict_success"])

    def test_only_deflects_yields_no_verified(self):
        state = {
            "execution_result": {"diagnosis_required": True},
            "retry_count": 2,
            "error_resolved": False,
            "error_agent": "python_developer",
            "true_root_cause": "model_expert",
            "refutation_log": [
                make_refutation_entry(
                    layer="model_expert",
                    action="deflect",
                    error_resolved=False,
                    re_solve_strict_success=False,
                    retry_index=1,
                ),
                make_refutation_entry(
                    layer="python_developer",
                    action="deflect",
                    error_resolved=False,
                    re_solve_strict_success=False,
                    retry_index=2,
                ),
            ],
        }
        out = infer_verified_attribution(state)
        self.assertIsNone(out["verified_attributed_layer"])
        self.assertFalse(out["verified_attribution_hit"])

    def test_failed_comply_overturn_not_verified(self):
        state = {
            "execution_result": {"diagnosis_required": True},
            "retry_count": 2,
            "error_resolved": False,
            "error_agent": "python_developer",
            "true_root_cause": "model_expert",
            "refutation_log": [
                make_refutation_entry(
                    layer="model_expert",
                    action="comply",
                    error_resolved=True,
                    re_solve_strict_success=False,
                    retry_index=1,
                    overturn=True,
                ),
            ],
        }
        out = infer_verified_attribution(state)
        self.assertIsNone(out["verified_attributed_layer"])
        self.assertFalse(out["verified_attribution_hit"])

    def test_last_comply_alone_is_not_verified_without_success(self):
        # Episode still failing: last-comply flags set but no Practical Optimal.
        state = {
            "execution_result": {"diagnosis_required": True},
            "retry_count": 1,
            "error_resolved": True,
            "error_agent": "model_expert",
            "true_root_cause": "model_expert",
            "refutation_log": [],
        }
        out = infer_verified_attribution(state)
        self.assertIsNone(out["verified_attributed_layer"])
        self.assertFalse(out["verified_attribution_hit"])

    def test_finalize_includes_verified_fields(self):
        payoff = finalize_episode_payoffs(_success_state())
        self.assertEqual(payoff["verified_attributed_layer"], "model_expert")
        self.assertTrue(payoff["verified_attribution_hit"])
        self.assertTrue(payoff["attribution_hit"])  # legacy last-comply also true
        self.assertTrue(payoff["first_probe_hit"])
        self.assertIsNotNone(payoff.get("refutation_log_summary"))
        self.assertEqual(payoff.get("omega_source"), "evidence_rank")

    def test_cascade_last_comply_differs_from_verified(self):
        # ME tip first; PD last-comply after cascade success — if only PD's
        # comply reached Practical Optimal, verified=PD even if a*=ME.
        state = _success_state(
            error_agent="python_developer",
            error_resolved=True,
            true_root_cause="model_expert",
            retry_count=2,
            refutation_log=[
                make_refutation_entry(
                    layer="model_expert",
                    action="deflect",
                    error_resolved=False,
                    re_solve_strict_success=False,
                    retry_index=1,
                ),
            ],
            backtrack_history=[
                {"error_agent": "model_expert", "error_resolved": False},
                {"error_agent": "python_developer", "error_resolved": True},
            ],
        )
        payoff = finalize_episode_payoffs(state)
        self.assertEqual(payoff["verified_attributed_layer"], "python_developer")
        self.assertFalse(payoff["verified_attribution_hit"])
        self.assertEqual(payoff["attributed_layer"], "python_developer")
        self.assertFalse(payoff["attribution_hit"])
        self.assertTrue(payoff["first_probe_hit"])  # ω tip was ME


if __name__ == "__main__":
    unittest.main()
