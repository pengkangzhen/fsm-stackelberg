"""Tests for symptom-scoped ME-clear / reopen policy (principled D)."""

from fsm_stackelberg.game.clear_policy import (
    layer_has_unresolved_formulation_evidence,
    reopen_symptom_scoped_deflects,
    symptom_fingerprint,
)
from fsm_stackelberg.game.verified import make_refutation_entry


def _state(**kwargs):
    base = {
        "error_category": "execution",
        "gurobi_status": "",
        "error_info": '{"error_type": "KeyError", "error_message": "vessel_calls"}',
        "model_expert_output": {
            "constraints": [
                {
                    "name": "Injected_Force_Zero_Sea_Reposition",
                    "expression": "(= y_in 0)",
                }
            ]
        },
        "python_code": "model = gp.Model('x')\n",
    }
    base.update(kwargs)
    return base


def test_symptom_fingerprint_changes_with_status():
    s1 = _state(
        error_category="execution",
        gurobi_status="",
        error_info='{"error_type": "KeyError"}',
    )
    s2 = _state(
        error_category="optimization",
        gurobi_status="INFEASIBLE",
        error_info='{"error_type": "OptimizationError"}',
    )
    assert symptom_fingerprint(s1) != symptom_fingerprint(s2)


def test_formulation_evidence_from_force_zero_name():
    assert layer_has_unresolved_formulation_evidence(_state(), "model_expert")
    assert not layer_has_unresolved_formulation_evidence(_state(), "python_developer")
    clean = _state(
        model_expert_output={"constraints": [{"name": "balance", "expression": "x"}]},
        python_code="model.addVar()",
    )
    assert not layer_has_unresolved_formulation_evidence(clean, "model_expert")


def test_same_symptom_does_not_reopen_me():
    """v3 step mid-flight: ME deflect under KeyError; immediate re-entry same fp."""
    state = _state()
    fp = symptom_fingerprint(state)
    log = [
        make_refutation_entry(
            layer="model_expert",
            action="deflect",
            error_resolved=False,
            re_solve_strict_success=False,
            retry_index=1,
            symptom_fingerprint=fp,
        )
    ]
    cleared = reopen_symptom_scoped_deflects(["model_expert"], log, state)
    assert cleared == ["model_expert"]


def test_v3_pattern_reopens_me_when_symptom_changes():
    """KeyError deflect → PD fix → INFEASIBLE: reopen ME while plant smell remains."""
    keyerror_state = _state()
    fp_key = symptom_fingerprint(keyerror_state)
    log = [
        make_refutation_entry(
            layer="model_expert",
            action="deflect",
            error_resolved=False,
            re_solve_strict_success=False,
            retry_index=1,
            symptom_fingerprint=fp_key,
        ),
        make_refutation_entry(
            layer="python_developer",
            action="comply",
            error_resolved=True,
            re_solve_strict_success=False,
            retry_index=2,
            overturn=True,
            symptom_fingerprint=symptom_fingerprint(
                _state(
                    error_category="optimization",
                    gurobi_status="INFEASIBLE",
                    error_info='{"error_type": "OptimizationError"}',
                )
            ),
        ),
    ]
    infeas_state = _state(
        error_category="optimization",
        gurobi_status="INFEASIBLE",
        error_info='{"error_type": "OptimizationError", "error_message": "INFEASIBLE"}',
    )
    cleared = reopen_symptom_scoped_deflects(
        ["model_expert", "python_developer"], log, infeas_state
    )
    assert "model_expert" not in cleared
    assert "python_developer" in cleared  # comply overturn stays cleared


def test_no_reopen_without_formulation_evidence():
    """If ME already stripped, symptom change alone does not reopen."""
    key_fp = symptom_fingerprint(_state())
    log = [
        make_refutation_entry(
            layer="model_expert",
            action="deflect",
            error_resolved=False,
            re_solve_strict_success=False,
            retry_index=1,
            symptom_fingerprint=key_fp,
        )
    ]
    stripped = _state(
        error_category="optimization",
        gurobi_status="INFEASIBLE",
        error_info='{"error_type": "OptimizationError"}',
        model_expert_output={"constraints": [{"name": "balance"}]},
        python_code="model.optimize()",
    )
    cleared = reopen_symptom_scoped_deflects(["model_expert"], log, stripped)
    assert cleared == ["model_expert"]
