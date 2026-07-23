"""Unit tests for structured run logging (stdlib unittest)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from fsm_stackelberg.game.payoff import finalize_episode_payoffs, infer_commitment_diagnostics
from fsm_stackelberg.utils.experiment_result import ExperimentResult
from fsm_stackelberg.utils.run_log import (
    build_run_manifest,
    clear_run_logger,
    init_run_logger,
    record_step_event,
    save_backward_artifact,
    save_inject_artifact,
    save_run_manifest,
)


class TestRunLogger(unittest.TestCase):
    def tearDown(self):
        clear_run_logger()

    def test_events_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            logger = init_run_logger(
                run_dir,
                run_id="test_run_1",
                provider="Qwen",
                model="qwen3.7-plus",
                temperature=0.0,
                log_prompts=True,
            )
            state = {
                "step_metrics": [],
                "current_round": 1,
                "provider": "Qwen",
                "model": "qwen3.7-plus",
                "result_dir": str(run_dir),
                "run_id": "test_run_1",
                "log_prompts": True,
                "inspection_policy": {
                    "omega": ["model_expert", "python_developer", "data_engineer"],
                    "nu": "executed_refutation",
                },
                "probe_queue": ["model_expert", "python_developer", "data_engineer"],
                "true_root_cause": "model_expert",
                "backtrack_history": [
                    {
                        "error_agent": "model_expert",
                        "error_resolved": True,
                        "gurobi_status": "OPTIMAL",
                        "error_category": "optimization",
                    }
                ],
                "execution_result": {
                    "diagnosis_required": False,
                    "gurobi_status": "OPTIMAL",
                    "result": {"objective_value": 100.0},
                },
                "expected_value": 100.0,
                "retry_count": 1,
                "error_resolved": True,
                "error_agent": "model_expert",
                "total_tokens": 1200,
            }

            steps = record_step_event(
                state,
                node="data_engineer",
                step_type="forward",
                duration_s=1.25,
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=150,
                prompt_text="ROLE\nTASK for DE",
                artifact={"sets": []},
                artifact_name="de_forward_r1",
            )
            state["step_metrics"] = steps

            steps = record_step_event(
                state,
                node="diagnosis_agent",
                step_type="diagnosis",
                duration_s=0.01,
                total_tokens=0,
                artifact={
                    "probed_agent": "model_expert",
                    "committed_omega": ["model_expert", "python_developer", "data_engineer"],
                },
                artifact_name="diagnosis_r1",
                extra={"probed_agent": "model_expert"},
            )
            state["step_metrics"] = steps

            save_backward_artifact(
                state,
                agent="model_expert",
                is_caused_by_you=True,
                error_resolved=True,
                reason="fixed stage-2 balance",
                refined_present=True,
            )
            save_inject_artifact(state, plant_id="me_force_zero_sea", note="unit-test")

            events = [
                json.loads(line)
                for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertGreaterEqual(len(events), 3)
            self.assertEqual(events[0]["node"], "data_engineer")
            self.assertEqual(events[0]["prompt_tokens"], 100)
            self.assertTrue((run_dir / "prompts" / "round1_data_engineer.txt").exists())
            self.assertTrue((run_dir / "artifacts" / "de_forward_r1.json").exists())
            self.assertTrue((run_dir / "artifacts" / "diagnosis_r1.json").exists())
            self.assertTrue((run_dir / "artifacts" / "backward_model_expert_r1.json").exists())
            self.assertTrue((run_dir / "artifacts" / "inject_me_force_zero_sea.json").exists())

            payoff = finalize_episode_payoffs(state)
            state["episode_payoff"] = payoff
            exp = ExperimentResult(
                algorithm="mako",
                provider="Qwen",
                model="qwen3.7-plus",
                dataset="prob_tslp_ecr_demand",
                prob_name="smoke_H4_Omega5",
                orchestrator_mode="stackelberg",
                probe_order="causal",
                true_root_cause="model_expert",
                attributed_layer="model_expert",
                episode_payoff=payoff,
                status=True,
                gurobi_status="OPTIMAL",
                obj_value=100.0,
                expected_value=100.0,
                total_tokens=1200,
                total_prompt_tokens=100,
                total_completion_tokens=50,
                total_duration_s=1.25,
                result_path=str(run_dir),
                temperature=0.0,
            )
            manifest = build_run_manifest(
                exp=exp,
                state=state,
                run_id=logger.run_id,
                started_at=logger.started_at,
                temperature=0.0,
            )
            path = save_run_manifest(run_dir, manifest)
            self.assertTrue(path.exists())
            loaded = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(loaded["config"]["model"], "qwen3.7-plus")
            self.assertEqual(loaded["config"]["temperature"], 0.0)
            self.assertTrue(loaded["outcome"]["practical_optimal"])
            self.assertTrue(loaded["inspection"]["first_probe_hit"])
            self.assertEqual(
                loaded["inspection"]["committed_omega"][0],
                "model_expert",
            )


class TestCommitmentDiagnostics(unittest.TestCase):
    def test_first_probe_hit(self):
        state = {
            "inspection_policy": {"omega": ["model_expert", "python_developer"]},
            "true_root_cause": "model_expert",
            "backtrack_history": [{"error_agent": "model_expert"}],
        }
        diag = infer_commitment_diagnostics(state)
        self.assertTrue(diag["first_probe_hit"])
        self.assertEqual(diag["true_root_rank_in_omega"], 0)


if __name__ == "__main__":
    unittest.main()
