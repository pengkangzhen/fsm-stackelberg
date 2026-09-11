"""Unit tests for debate/reflexion baselines (stdlib unittest, no LLM)."""

from __future__ import annotations

import unittest
from unittest import mock

from fsm_stackelberg.agents import diagnosis_agent as da
from fsm_stackelberg.graph.workflow import route_after_diagnosis, route_after_backward


class TestVoteAggregation(unittest.TestCase):
    def test_majority_wins(self):
        pick, note = da._aggregate_debate_votes(
            ["model_expert", "model_expert", "python_developer"],
            ["python_developer", "model_expert", "data_engineer"])
        self.assertEqual(pick, "model_expert")
        self.assertIn("majority 2/3", note)

    def test_unanimous(self):
        pick, note = da._aggregate_debate_votes(
            ["data_engineer"] * 3,
            ["python_developer", "model_expert", "data_engineer"])
        self.assertEqual(pick, "data_engineer")
        self.assertIn("majority 3/3", note)

    def test_three_way_tie_breaks_to_status_prior(self):
        prior = ["python_developer", "model_expert", "data_engineer"]
        pick, note = da._aggregate_debate_votes(
            ["model_expert", "python_developer", "data_engineer"], prior)
        self.assertEqual(pick, prior[0])
        self.assertIn("tie", note)

    def test_no_valid_votes_falls_back_to_prior(self):
        pick, note = da._aggregate_debate_votes(
            [], ["python_developer", "model_expert", "data_engineer"])
        self.assertEqual(pick, "python_developer")
        self.assertIn("status prior", note)


class TestReflexionMemory(unittest.TestCase):
    def test_empty_history(self):
        mem = da._build_reflexion_memory({})
        self.assertIn("first diagnosis round", mem)

    def test_history_lists_accusations_as_unresolved(self):
        state = {"backtrack_history": [
            {"retry_count": 1, "error_agent": "python_developer",
             "reason": "traceback points at code"},
            {"retry_count": 2, "error_agent": "model_expert",
             "reason": "balance mismatch"},
        ]}
        mem = da._build_reflexion_memory(state)
        self.assertIn("persisted", mem)
        self.assertIn("accused python_developer", mem)
        self.assertIn("accused model_expert", mem)


class TestDebateDiagnosisFlow(unittest.TestCase):
    """_debate_diagnosis with a mocked LLM chain (no network)."""

    def _run(self, votes):
        state = {
            "provider": "DeepSeek", "model": "deepseek-flash",
            "gurobi_status": "INFEASIBLE",
            "stack_trace": "", "backtrack_history": [],
        }

        class FakeOut:
            def __init__(self, suspect):
                self.suspected_agent = suspect
                self.argument = f"case for {suspect}"

        outputs = [FakeOut(v) for v in votes]
        calls = {"n": 0}

        class FakeChain:
            def invoke(self, payload):
                out = outputs[calls["n"]]
                calls["n"] += 1
                return out

        class FakeCB:
            def __enter__(self):
                class CB:
                    total_tokens = 10
                return CB()

            def __exit__(self, *a):
                return False

        fake_prompt = mock.MagicMock()
        fake_prompt.__or__ = mock.MagicMock(return_value=FakeChain())
        fake_llm = mock.MagicMock()
        with mock.patch.object(da.ChatPromptTemplate, "from_messages",
                               return_value=fake_prompt), \
             mock.patch.object(da, "get_llm", return_value=fake_llm), \
             mock.patch.object(da, "get_openai_callback", return_value=FakeCB()), \
             mock.patch.object(da, "invoke_structured",
                               side_effect=lambda chain, payload, node=None:
                               chain.invoke(payload)):
            agent, conf, reason, metrics = da._debate_diagnosis(
                state, "optimization", "INFEASIBLE", "err")
        return agent, conf, reason, metrics

    def test_two_of_three_majority(self):
        agent, _conf, reason, _metrics = self._run(
            ["model_expert", "python_developer", "model_expert"])
        self.assertEqual(agent, "model_expert")
        self.assertIn("majority 2/3", reason)

    def test_invalid_vote_dropped_tie_breaks_to_prior(self):
        # Second debater names a non-candidate layer -> dropped; 1-1 among
        # valid votes -> tie -> status prior (model_expert first for
        # INFEASIBLE).
        agent, _conf, reason, _metrics = self._run(
            ["model_expert", "weather_agent", "data_engineer"])
        self.assertIn("tie", reason)
        self.assertEqual(agent, "model_expert")


class TestRoutingAcceptsNewModes(unittest.TestCase):
    def _state(self, mode):
        return {"diagnosis_mode": mode, "error_agent": "model_expert",
                "retry_count": 1, "max_retries": 3}

    def test_debate_routes_to_backward(self):
        self.assertEqual(route_after_diagnosis(self._state("debate")),
                         "model_expert_backward")

    def test_reflexion_routes_to_backward(self):
        self.assertEqual(route_after_diagnosis(self._state("reflexion")),
                         "model_expert_backward")

    def test_debate_unresolved_reenters_diagnosis(self):
        s = self._state("debate")
        s["error_resolved"] = False
        self.assertEqual(route_after_backward(s), "diagnosis_agent")

    def test_reflexion_resolved_reruns_downstream(self):
        s = self._state("reflexion")
        s["error_resolved"] = True
        self.assertEqual(route_after_backward(s), "fault_injector")


if __name__ == "__main__":
    unittest.main()
