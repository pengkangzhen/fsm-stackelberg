"""Unit tests for Exp-I fault plants."""

from fsm_stackelberg.injection import apply_plant, get_plant
from fsm_stackelberg.schemas import (
    Constraint,
    DecisionVariable,
    ModelComponents,
    ModelExpertOutput,
    ObjectiveFunction,
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
