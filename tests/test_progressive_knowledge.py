"""Unit tests for progressive (catalog-first) knowledge injection."""

from fsm_stackelberg.knowledge.progressive import ProgressiveKnowledgeInjection
from fsm_stackelberg.plugins import build_feature_bundle


def test_progressive_bootstrap_is_catalog_only():
    inj = ProgressiveKnowledgeInjection.from_mode("progressive")
    state = inj.bootstrap_state()

    assert state["knowledge_injection_mode"] == "progressive"
    assert state["knowledge_loader"] is not None
    assert state["knowledge_catalog"]
    assert "tslp-" in state["knowledge_catalog"] or "|" in state["knowledge_catalog"]
    assert state["loaded_knowledge"] is None
    assert state["loaded_knowledge_modules"] == []
    assert state["knowledge_round"] == 0
    assert state["knowledge_max_rounds"] >= 1


def test_enable_alias_matches_progressive():
    a = ProgressiveKnowledgeInjection.from_mode("enable").bootstrap_state()
    b = ProgressiveKnowledgeInjection.from_mode("progressive").bootstrap_state()
    assert a["knowledge_injection_mode"] == b["knowledge_injection_mode"] == "progressive"
    assert a["loaded_knowledge"] is None
    assert b["loaded_knowledge"] is None


def test_disable_clears_loader():
    state = ProgressiveKnowledgeInjection.from_mode("disable").bootstrap_state()
    assert state["knowledge_injection_mode"] == "disable"
    assert state["knowledge_loader"] is None
    assert state["knowledge_catalog"] == ""
    assert state["loaded_knowledge"] is None


def test_on_demand_load_by_name():
    inj = ProgressiveKnowledgeInjection.from_mode("progressive")
    state = inj.bootstrap_state()
    loader = state["knowledge_loader"]
    names = loader.list_module_names()
    assert names, "expected TSLP knowledge modules on disk"

    target = names[0]
    body = loader.get_knowledge_by_names([target])
    assert target.replace("tslp-", "") in body or target in body or "###" in body
    assert ProgressiveKnowledgeInjection.pending_requests(
        [target, "missing_mod"], [target], []
    ) == ["missing_mod"]


def test_feature_bundle_wires_diagnosis_without_preload():
    bundle = build_feature_bundle(
        knowledge_mode="progressive",
        diagnosis_mode="stackelberg",
        probe_order="causal",
    )
    state = bundle.bootstrap_state()
    assert state["diagnosis_mode"] == "stackelberg"
    assert state["probe_order"] == "causal"
    assert state["loaded_knowledge"] is None
    assert state["loaded_knowledge_modules"] == []
    assert state["inspection_policy"] is None
