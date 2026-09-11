"""Offline tests for failure-snapshot freeze/resume (no LLM, no solver).

Covers: snapshot round-trip (pydantic restore, knowledge-loader rebuild,
override application, cumulative-cost retention), the freeze gate, and the
resume graph entering diagnosis without any forward node running.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fsm_stackelberg.graph.snapshot import (
    SNAPSHOT_MANIFEST_FILE,
    SNAPSHOT_STATE_FILE,
    build_resume_state,
    load_snapshot,
    read_snapshot_manifest,
    route_after_snapshot_gate,
    snapshot_gate_node,
    write_failure_snapshot,
)
from fsm_stackelberg.graph.workflow import create_mako_graph
from fsm_stackelberg.schemas import (
    DataEngineerOutput,
    ModelComponents,
    ModelExpertOutput,
    ModelInputs,
    ObjectiveFunction,
)


def _failure_state(snapshot_dir=None):
    return {
        "problem_description": "demo ECR problem",
        "sample": {"nodes": [1, 2]},
        "schema": {"nodes": [1, 2]},
        "provider": "DashScope",
        "model": "deepseek-v4-flash",
        "temperature": 0.0,
        "result_dir": "/tmp/freeze_run",
        "run_id": "freeze_run_1",
        "diagnosis_mode": "stackelberg",
        "probe_order": "causal",
        "omega_source": "evidence_rank",
        "rank_method": "hybrid",
        "retry_count": 0,
        "max_retries": 3,
        "gurobi_status": "OPTIMAL",
        "error_category": "optimization",
        "error_info": json.dumps({"error_type": "GapError"}),
        "execution_result": {"diagnosis_required": True, "gurobi_status": "OPTIMAL"},
        "data_engineer_output": DataEngineerOutput(
            model_inputs=ModelInputs(sets=[], parameters=[])
        ),
        "model_expert_output": ModelExpertOutput(
            model_components=ModelComponents(
                decision_variables=[],
                objective_function=ObjectiveFunction(
                    direction="min", expression="x", description="demo"
                ),
                constraints=[],
            )
        ),
        "python_code": "print('demo')",
        "total_tokens": 12345,
        "total_duration_s": 42.0,
        "step_metrics": [{"node": "data_engineer", "step_type": "forward"}],
        "node_metrics": {"data_engineer": {"total_tokens": 100}},
        "current_round": 1,
        "output_history": [{"round": 1, "agent": "data_engineer"}],
        "true_root_cause": "model_expert",
        "expected_value": 1389581.0,
        "inject_id": "me_force_zero_sea",
        "fault_plant_id": "me_force_zero_sea",
        "fault_injected": True,
        "probe_seed": 7,
        "knowledge_injection_mode": "progressive",
        "knowledge_excluded_modules": [],
        "loaded_knowledge_modules": ["tslp_sets"],
        "snapshot_dir": snapshot_dir,
    }


class TestSnapshotRoundTrip(unittest.TestCase):
    def test_round_trip_restores_types_and_applies_overrides(self):
        with tempfile.TemporaryDirectory() as tmp:
            snap_dir = str(Path(tmp) / "snap")
            write_failure_snapshot(_failure_state(), snap_dir)

            resume_state, manifest = build_resume_state(
                snap_dir,
                diagnosis_mode="adversarial",
                probe_order="reverse",
                rank_method="heuristic",
                max_retries=2,
                result_dir="/tmp/arm_r1",
                run_id="arm_r1",
            )

            # Pydantic outputs restored as model instances.
            self.assertIsInstance(resume_state["data_engineer_output"], DataEngineerOutput)
            self.assertIsInstance(resume_state["model_expert_output"], ModelExpertOutput)
            # Knowledge loader rebuilt from its mode fields.
            self.assertIsNotNone(resume_state["knowledge_loader"])

            # Diagnosis-side overrides applied.
            self.assertEqual(resume_state["diagnosis_mode"], "adversarial")
            self.assertEqual(resume_state["probe_order"], "reverse")
            self.assertEqual(resume_state["rank_method"], "heuristic")
            self.assertEqual(resume_state["max_retries"], 2)
            self.assertEqual(resume_state["result_dir"], "/tmp/arm_r1")
            self.assertEqual(resume_state["run_id"], "arm_r1")

            # Policy-invariant prefix retained verbatim.
            self.assertEqual(resume_state["total_tokens"], 12345)
            self.assertEqual(resume_state["true_root_cause"], "model_expert")
            self.assertEqual(resume_state["provider"], "DashScope")
            self.assertEqual(resume_state["loaded_knowledge_modules"], ["tslp_sets"])

            # Episode-transient keys must not leak across the boundary.
            for key in ("error_agent", "error_resolved", "episode_payoff",
                        "snapshot_dir", "snapshot_written"):
                self.assertNotIn(key, resume_state)

            # Manifest carries provenance.
            self.assertEqual(manifest["config"]["provider"], "DashScope")
            self.assertEqual(manifest["config"]["inject_id"], "me_force_zero_sea")
            self.assertEqual(manifest["failure_surface"]["gurobi_status"], "OPTIMAL")
            self.assertEqual(manifest["forward_totals"]["total_tokens"], 12345)

    def test_dropped_keys_not_in_snapshot_state_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            snap_dir = str(Path(tmp) / "snap")
            state = _failure_state(snapshot_dir=snap_dir)
            state["snapshot_written"] = True
            state["knowledge_loader"] = object()  # runtime instance
            write_failure_snapshot(state, snap_dir)

            raw = json.loads(
                (Path(snap_dir) / SNAPSHOT_STATE_FILE).read_text(encoding="utf-8")
            )
            self.assertNotIn("knowledge_loader", raw)
            self.assertNotIn("snapshot_written", raw)
            # json round-trip of the dump must succeed
            self.assertEqual(raw["provider"], "DashScope")

    def test_snapshot_requires_failed_solve(self):
        state = _failure_state()
        state["execution_result"] = {"diagnosis_required": False}
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                write_failure_snapshot(state, str(Path(tmp) / "snap"))

    def test_knowledge_disabled_rebuilds_none_loader(self):
        with tempfile.TemporaryDirectory() as tmp:
            snap_dir = str(Path(tmp) / "snap")
            state = _failure_state()
            state["knowledge_injection_mode"] = "disable"
            write_failure_snapshot(state, snap_dir)
            resume_state, _ = build_resume_state(snap_dir)
            self.assertIsNone(resume_state["knowledge_loader"])

    def test_read_snapshot_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            snap_dir = str(Path(tmp) / "snap")
            write_failure_snapshot(
                _failure_state(), snap_dir, provenance={"dataset": "prob_tslp_ecr_demand"}
            )
            manifest = read_snapshot_manifest(snap_dir)
            self.assertEqual(manifest["provenance"]["dataset"], "prob_tslp_ecr_demand")
            self.assertTrue((Path(snap_dir) / SNAPSHOT_MANIFEST_FILE).exists())
            # load_snapshot returns (state, manifest)
            state, man = load_snapshot(snap_dir)
            self.assertEqual(state["provider"], "DashScope")
            self.assertEqual(man["schema_version"], 1)


class TestSnapshotGate(unittest.TestCase):
    def test_gate_freezes_and_ends(self):
        with tempfile.TemporaryDirectory() as tmp:
            snap_dir = str(Path(tmp) / "snap")
            state = _failure_state(snapshot_dir=snap_dir)

            update = snapshot_gate_node(state)
            self.assertEqual(update, {"snapshot_written": True})
            self.assertTrue((Path(snap_dir) / SNAPSHOT_STATE_FILE).exists())
            merged = {**state, **update}
            self.assertEqual(route_after_snapshot_gate(merged), "end")

    def test_gate_passes_through_without_snapshot_dir(self):
        state = _failure_state()
        self.assertEqual(snapshot_gate_node(state), {})
        self.assertEqual(route_after_snapshot_gate(state), "diagnosis_agent")

    def test_gate_is_idempotent_once_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = _failure_state(snapshot_dir=str(Path(tmp) / "snap"))
            merged = {**state, "snapshot_written": True}
            self.assertEqual(snapshot_gate_node(merged), {})
            self.assertEqual(route_after_snapshot_gate(merged), "end")


class TestResumeGraph(unittest.TestCase):
    def test_resume_enters_diagnosis_and_skips_forward(self):
        forward_calls = []

        def _record_forward(state):
            forward_calls.append(state.get("provider"))
            return {}

        with tempfile.TemporaryDirectory() as tmp:
            snap_dir = str(Path(tmp) / "snap")
            write_failure_snapshot(_failure_state(), snap_dir)
            resume_state, _ = build_resume_state(
                snap_dir,
                result_dir=str(Path(tmp) / "arm"),
                run_id="arm_r1",
            )

        def _fake_diagnosis(state):
            # Assert the restored blackboard is visible to the treatment path.
            assert state.get("true_root_cause") == "model_expert"
            assert state.get("knowledge_loader") is not None
            return {"error_agent": None}  # routes to "end"

        with mock.patch(
            "fsm_stackelberg.graph.workflow.data_engineer_node",
            side_effect=_record_forward,
        ), mock.patch(
            "fsm_stackelberg.graph.workflow.model_expert_node",
            side_effect=_record_forward,
        ), mock.patch(
            "fsm_stackelberg.graph.workflow.python_developer_node",
            side_effect=_record_forward,
        ), mock.patch(
            "fsm_stackelberg.graph.workflow.diagnosis_agent_node",
            side_effect=_fake_diagnosis,
        ) as diag:
            graph = create_mako_graph(start_node="diagnosis_agent")
            final_state = graph.invoke(resume_state)

        diag.assert_called_once()
        self.assertEqual(forward_calls, [])  # no forward node ran
        self.assertIsNone(final_state.get("error_agent"))
        self.assertEqual(final_state.get("total_tokens"), 12345)

    def test_freeze_graph_routes_through_gate(self):
        """solver→gate→END when snapshot_dir set; gate node dumps state."""
        gate_calls = []

        with tempfile.TemporaryDirectory() as tmp:
            snap_dir = str(Path(tmp) / "snap")

            def _fake_solver(state):
                return {
                    "execution_result": {
                        "diagnosis_required": True,
                        "gurobi_status": "OPTIMAL",
                    },
                    "gurobi_status": "OPTIMAL",
                    "error_category": "optimization",
                    "total_tokens": 12345,
                    "total_duration_s": 42.0,
                }

            def _fake_gate(state):
                gate_calls.append(state.get("snapshot_dir"))
                return snapshot_gate_node(state)

            base = _failure_state(snapshot_dir=snap_dir)
            base.pop("snapshot_dir")  # provided via invoke input below

            with mock.patch(
                "fsm_stackelberg.graph.workflow.solver_executor_node",
                side_effect=_fake_solver,
            ), mock.patch(
                "fsm_stackelberg.graph.workflow.data_engineer_node",
                side_effect=lambda s: {},
            ), mock.patch(
                "fsm_stackelberg.graph.workflow.fault_injector_node",
                side_effect=lambda s: {},
            ), mock.patch(
                "fsm_stackelberg.graph.workflow.fault_injector_de_node",
                side_effect=lambda s: {},
            ), mock.patch(
                "fsm_stackelberg.graph.workflow.fault_injector_pd_node",
                side_effect=lambda s: {},
            ), mock.patch(
                "fsm_stackelberg.graph.workflow.model_expert_node",
                side_effect=lambda s: {},
            ), mock.patch(
                "fsm_stackelberg.graph.workflow.python_developer_node",
                side_effect=lambda s: {},
            ), mock.patch(
                "fsm_stackelberg.graph.workflow.snapshot_gate_node",
                side_effect=_fake_gate,
            ):
                graph = create_mako_graph(snapshot_gate=True)
                final_state = graph.invoke(base | {"snapshot_dir": snap_dir})

            self.assertEqual(gate_calls, [snap_dir])
            self.assertTrue(final_state.get("snapshot_written"))
            self.assertTrue((Path(snap_dir) / SNAPSHOT_STATE_FILE).exists())


if __name__ == "__main__":
    unittest.main()
