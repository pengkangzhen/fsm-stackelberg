"""Unit tests for Exp-I fault plants."""

import pytest

from fsm_stackelberg.agents.fault_injector import (
    fault_injector_de_node,
    fault_injector_node,
    fault_injector_pd_node,
)
from fsm_stackelberg.injection import apply_plant, get_plant
from fsm_stackelberg.schemas import (
    Constraint,
    DataEngineerOutput,
    DecisionVariable,
    IndexInfo,
    ModelComponents,
    ModelExpertOutput,
    ModelInputs,
    ObjectiveFunction,
    ParameterDefinition,
)


def _me_with_balances() -> ModelExpertOutput:
    return ModelExpertOutput(
        knowledge_requests=[],
        model_components=ModelComponents(
            decision_variables=[
                DecisionVariable(
                    symbol="x",
                    indices=["i"],
                    shape=["N"],
                    type="Continuous",
                    description="flow",
                )
            ],
            objective_function=ObjectiveFunction(
                direction="min",
                expression="sum x",
                description="cost",
            ),
            constraints=[
                Constraint(
                    name="Sea_Arrival_Capacity",
                    expression="y <= B",
                    description="Sea capacity.",
                ),
                Constraint(
                    name="Inventory_Balance_Hub",
                    expression="I = I_prev + ...",
                    description="Inventory balance for Hubs including sea terms.",
                ),
                Constraint(
                    name="Inventory_Balance_NonHub",
                    expression="I = I_prev + ...",
                    description="Inventory balance for Spokes and Dry Ports.",
                ),
                Constraint(
                    name="Nonnegativity",
                    expression="x >= 0",
                    description="All variables non-negative.",
                ),
            ],
        ),
    )


def test_me_drop_stage2_balance_drops_inventory_balances():
    state = {"model_expert_output": _me_with_balances()}
    out = apply_plant(state, "me_drop_stage2_balance")
    me = out["model_expert_output"]
    names = [c.name for c in me.model_components.constraints]
    assert "Inventory_Balance_Hub" not in names
    assert "Inventory_Balance_NonHub" not in names
    assert "Sea_Arrival_Capacity" in names
    assert "Nonnegativity" in names
    assert out["fault_injected"] is True
    assert out["true_root_cause"] == "model_expert"
    assert out["fault_plant_id"] == "me_drop_stage2_balance"


def test_apply_plant_is_idempotent():
    state = apply_plant({"model_expert_output": _me_with_balances()}, "me_drop_stage2_balance")
    n1 = len(state["model_expert_output"].model_components.constraints)
    state2 = apply_plant(state, "me_drop_stage2_balance")
    n2 = len(state2["model_expert_output"].model_components.constraints)
    assert n1 == n2


def test_get_plant_metadata():
    plant = get_plant("me_drop_stage2_balance")
    assert plant.true_root_cause == "model_expert"


def test_me_force_zero_sea_flips_obj_and_adds_constraint():
    state = {
        "model_expert_output": _me_with_balances(),
        "loaded_knowledge": "FULL TEXT WITH BALANCE FORMULAS",
    }
    out = apply_plant(state, "me_force_zero_sea")
    me = out["model_expert_output"]
    names = [c.name for c in me.model_components.constraints]
    assert "Injected_Force_Zero_Sea_Reposition" in names
    # Tight plant keeps balances and min objective.
    assert "Inventory_Balance_Hub" in names
    assert me.model_components.objective_function.direction == "min"
    assert out["loaded_knowledge"] == ""
    assert out["true_root_cause"] == "model_expert"
    assert out["fault_plant_id"] == "me_force_zero_sea"


# ---------------------------------------------------------------------------
# DE / PD layer plants (Phase 4 grid)
# ---------------------------------------------------------------------------


def _de_with_demand_supply() -> DataEngineerOutput:
    demand = ParameterDefinition(
        symbol="d[p,t,k]",
        indices=[IndexInfo(symbol="p", set="P")],
        description="Spoke demand at node p in period t under scenario k",
        source="supply_demand.demand.value",
    )
    supply = ParameterDefinition(
        symbol="sup[p,t]",
        indices=[IndexInfo(symbol="p", set="P")],
        description="Exogenous supply at node p in period t",
        source="supply_demand.supply.value",
    )
    return DataEngineerOutput(
        model_inputs=ModelInputs(sets=[], parameters=[demand, supply])
    )


def test_de_swap_demand_supply_source_swaps_sources():
    out = apply_plant({"data_engineer_output": _de_with_demand_supply()}, "de_swap_demand_supply_source")
    params = out["data_engineer_output"].model_inputs.parameters
    by_symbol = {p.symbol: p.source for p in params}
    assert by_symbol["d[p,t,k]"] == "supply_demand.supply.value"
    assert by_symbol["sup[p,t]"] == "supply_demand.demand.value"
    assert out["true_root_cause"] == "data_engineer"
    assert out["fault_plant_id"] == "de_swap_demand_supply_source"
    assert out["fault_injected"] is True


def test_de_swap_demand_supply_source_requires_both_params():
    de = _de_with_demand_supply()
    de = de.model_copy(
        update={
            "model_inputs": de.model_inputs.model_copy(
                update={"parameters": de.model_inputs.parameters[:1]}
            )
        }
    )
    with pytest.raises(ValueError):
        apply_plant({"data_engineer_output": de}, "de_swap_demand_supply_source")


_PD_CODE = """\
import gurobipy as gp

model = gp.Model("dep")
model.addConstr(x >= 0, name="Nonnegativity")
model.addConstr(I[h, t] == I_prev + inflow - outflow, name="Inventory_Balance_Hub")
model.addConstr(y <= B, name="Sea_Capacity")
model.addConstr(inv[d, t] == inv_prev + ship - fulfill, name="inland_balance_dry")
model.optimize()
"""


def test_pd_comment_out_balance_comments_balance_lines_only():
    out = apply_plant({"python_code": _PD_CODE}, "pd_comment_out_balance")
    code = out["python_code"]
    lines = code.splitlines()
    assert lines[4].lstrip().startswith("# ")
    assert "Inventory_Balance_Hub" in lines[4]
    assert lines[6].lstrip().startswith("# ")
    assert lines[3].lstrip().startswith("model.addConstr")  # Nonnegativity untouched
    assert lines[5].lstrip().startswith("model.addConstr")  # Sea_Capacity untouched
    assert out["true_root_cause"] == "python_developer"
    assert out["fault_plant_id"] == "pd_comment_out_balance"
    # Guarded idempotency via the fault_injected flag.
    out2 = apply_plant(out, "pd_comment_out_balance")
    assert out2["python_code"] == code


def test_pd_comment_out_balance_guard_when_no_match():
    clean_code = "model.addConstr(x >= 0, name='Nonnegativity')\n"
    with pytest.raises(ValueError):
        apply_plant({"python_code": clean_code}, "pd_comment_out_balance")


def test_plant_target_layers():
    assert get_plant("me_force_zero_sea").target_layer == "model_expert"
    assert get_plant("de_swap_demand_supply_source").target_layer == "data_engineer"
    assert get_plant("pd_comment_out_balance").target_layer == "python_developer"


def test_layer_injectors_route_by_target_layer():
    # DE-boundary injector no-ops for an ME plant (mismatched layer).
    state = {"inject_id": "me_force_zero_sea", "model_expert_output": _me_with_balances()}
    assert fault_injector_de_node(dict(state)) == state

    # DE-boundary injector applies a DE plant and records planted history.
    de_state = {
        "inject_id": "de_swap_demand_supply_source",
        "data_engineer_output": _de_with_demand_supply(),
        "current_round": 1,
    }
    updated = fault_injector_de_node(dict(de_state))
    assert updated["fault_plant_id"] == "de_swap_demand_supply_source"
    hist = updated["output_history"]
    assert hist and hist[-1]["agent"] == "data_engineer"

    # ME-boundary injector no-ops once any plant has fired (global one-shot flag).
    again = fault_injector_node(dict(updated))
    assert again == updated

    # PD-boundary injector no-ops when inject_id is unset.
    assert fault_injector_pd_node({"python_code": _PD_CODE}) == {"python_code": _PD_CODE}


def test_workflow_graph_compiles_with_layer_injectors():
    from fsm_stackelberg.graph.workflow import create_mako_graph

    graph = create_mako_graph()
    assert graph is not None
