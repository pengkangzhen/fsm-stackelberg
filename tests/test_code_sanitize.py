"""Tests for Gurobi model/mode shadowing sanitizer (PD regen fragility)."""

from fsm_stackelberg.utils.code_sanitize import (
    detects_model_mode_shadowing,
    unshadow_gurobi_model_m,
)

# Minimal reproduction of ea_causal_r1/r2 PD failure pattern.
_SHADOWED = '''
import gurobipy as gp

def optimize(data):
    m = gp.Model("empty_container_repositioning")
    try:
        x = {}
        for (i, j, m) in data["transport_cost"]:
            x[(i, j, m)] = m.addVar(lb=0, name=f"x_{i}_{j}_{m}")
        m.update()
        m.setObjective(0, gp.GRB.MINIMIZE)
        m.optimize()
        status = m.Status
        obj = m.ObjVal if status == gp.GRB.OPTIMAL else None
        return {"status": "OPTIMAL" if status == gp.GRB.OPTIMAL else "INFEASIBLE",
                "objective_value": obj, "variables": None}
    finally:
        m.dispose()
'''

_SAFE_MODEL = '''
import gurobipy as gp

def optimize(data):
    model = gp.Model("ecr")
    try:
        for (i, j, m) in data["transport_cost"]:
            model.addVar(lb=0, name=f"x_{i}_{j}_{m}")
        model.optimize()
        return {"status": "OPTIMAL", "objective_value": model.ObjVal, "variables": None}
    finally:
        model.dispose()
'''

_SAFE_MODE = '''
import gurobipy as gp

def optimize(data):
    m = gp.Model("ecr")
    try:
        for (i, j, mode) in data["transport_cost"]:
            m.addVar(lb=0, name=f"x_{i}_{j}_{mode}")
        m.optimize()
        return {"status": "OPTIMAL", "objective_value": m.ObjVal, "variables": None}
    finally:
        m.dispose()
'''


def test_detects_shadowing():
    assert detects_model_mode_shadowing(_SHADOWED) is True
    assert detects_model_mode_shadowing(_SAFE_MODEL) is False
    assert detects_model_mode_shadowing(_SAFE_MODE) is False


def test_unshadow_rewrites_model_api_only():
    fixed, rewritten = unshadow_gurobi_model_m(_SHADOWED)
    assert rewritten is True
    assert "model = gp.Model(" in fixed
    assert "m = gp.Model(" not in fixed
    assert "model.addVar(" in fixed
    assert "model.update()" in fixed
    assert "model.setObjective(" in fixed
    assert "model.optimize()" in fixed
    assert "model.Status" in fixed
    assert "model.ObjVal" in fixed
    assert "model.dispose()" in fixed
    # Mode index `m` in loops / f-strings must remain.
    assert "for (i, j, m) in" in fixed
    assert 'name=f"x_{i}_{j}_{m}"' in fixed
    assert detects_model_mode_shadowing(fixed) is False


def test_unshadow_noop_when_safe():
    for src in (_SAFE_MODEL, _SAFE_MODE):
        fixed, rewritten = unshadow_gurobi_model_m(src)
        assert rewritten is False
        assert fixed == src


def test_unshadow_real_ea_causal_r1_pd2_snippet():
    """Snippet shape from exp_i_pilot_ea_causal_r1 PyDeveloper_round2."""
    code = '''
    m = gp.Model("empty_container_repositioning")

    try:
        y_in = {}
        for h in hubs:
            for t in periods:
                y_in[(h, t)] = m.addVar(lb=0, name=f"y_in_{h}_{t}")

        x = {}
        for (i, j, m) in allowed_arcs:
            for t in periods:
                for k in scenarios:
                    x[(i, j, m, t, k)] = m.addVar(lb=0, name=f"x_{i}_{j}_{m}_{t}_{k}")
        m.update()
        m.optimize()
    finally:
        m.dispose()
'''
    assert detects_model_mode_shadowing(code)
    fixed, rewritten = unshadow_gurobi_model_m(code)
    assert rewritten
    assert "model = gp.Model(" in fixed
    assert "y_in[(h, t)] = model.addVar(" in fixed
    # After rewrite, the loop still binds `m`, but addVar uses model:
    assert "x[(i, j, m, t, k)] = model.addVar(" in fixed
    assert "model.dispose()" in fixed
    assert "m.dispose()" not in fixed
