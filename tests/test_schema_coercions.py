"""Regression tests for LLM-output coercions discovered in the subagent
rehearsal (results/subagent_rehearsal/REPORT.md; harness-only, not paper data).

Drift classes observed from real-style LLM outputs:
1. ParameterDefinition.indices emitted as a mapping {"i": "N"} (not a list).
2. DecisionVariable without indices/shape (indexing implicit in the symbol),
   with ``name`` instead of ``symbol``, and lowercase ``type``.
3. ObjectiveFunction.direction in natural language ("minimize").
"""

from __future__ import annotations

import unittest

from fsm_stackelberg.schemas import (
    DataEngineerOutput,
    DecisionVariable,
    ModelComponents,
    ModelExpertOutput,
    ObjectiveFunction,
    ParameterDefinition,
)


class TestParameterIndicesCoercion(unittest.TestCase):
    def test_indices_dict_becomes_ordered_list(self):
        p = ParameterDefinition.model_validate(
            {
                "symbol": "eta[w,i,t]",
                "source": "supply_demand.eta.value",
                "indices": {"w": "Omega", "i": "N", "t": "T"},
                "description": "scenario demand",
            }
        )
        self.assertEqual(
            [(x.symbol, x.set) for x in p.indices],
            [("w", "Omega"), ("i", "N"), ("t", "T")],
        )

    def test_indices_dict_inside_full_de_output(self):
        raw = {
            "model_inputs": {
                "sets": [],
                "parameters": [
                    {
                        "symbol": "xi[i,t]",
                        "source": "supply_demand.xi.value",
                        "indices": {"i": "N", "t": "T"},
                        "description": "supply",
                    }
                ],
            }
        }
        obj = DataEngineerOutput.model_validate(raw)
        self.assertEqual(
            [(x.symbol, x.set) for x in obj.model_inputs.parameters[0].indices],
            [("i", "N"), ("t", "T")],
        )

    def test_canonical_list_form_still_works(self):
        p = ParameterDefinition.model_validate(
            {
                "symbol": "c_spill",
                "source": "inventory.c_spill",
                "indices": [],
                "description": "spill penalty",
            }
        )
        self.assertEqual(p.indices, [])


class TestDecisionVariableCoercion(unittest.TestCase):
    def test_indices_derived_from_symbol_brackets(self):
        v = DecisionVariable.model_validate(
            {
                "name": "sea_repositioning_arrivals_stage1",
                "symbol": "y_in[h,t]",
                "type": "continuous",
                "bounds": "0 <= y_in[h,t] <= B_in[h] * v_calls[h,t]",
                "description": "stage-1 arrivals",
            }
        )
        self.assertEqual(v.indices, ["h", "t"])
        self.assertEqual(v.shape, ["h", "t"])  # descriptive default
        self.assertEqual(v.type, "Continuous")

    def test_scalar_symbol_without_brackets_uses_defaults(self):
        v = DecisionVariable.model_validate(
            {"symbol": "z_total", "indices": [], "shape": [], "type": "Binary", "description": "d"}
        )
        self.assertEqual(v.indices, [])
        self.assertEqual(v.type, "Binary")

    def test_explicit_indices_not_overridden(self):
        v = DecisionVariable.model_validate(
            {"symbol": "x[w,i,j,m,t]", "indices": ["w", "i", "j", "m", "t"], "shape": ["Omega", "N", "N", "M", "T"], "type": "Continuous", "description": "d"}
        )
        self.assertEqual(v.shape, ["Omega", "N", "N", "M", "T"])


class TestObjectiveDirectionCoercion(unittest.TestCase):
    def test_minimize_variants(self):
        for word in ("minimize", "MINIMISE", "minimum", "min", " Min "):
            obj = ObjectiveFunction.model_validate(
                {"direction": word, "expression": "x", "description": "d"}
            )
            self.assertEqual(obj.direction, "min")

    def test_maximize_variants(self):
        for word in ("maximize", "Maximise", "maximum", "max"):
            obj = ObjectiveFunction.model_validate(
                {"direction": word, "expression": "x", "description": "d"}
            )
            self.assertEqual(obj.direction, "max")

    def test_direction_flows_through_model_components_check(self):
        me = ModelExpertOutput.model_validate(
            {
                "knowledge_requests": [],
                "model_components": {
                    "decision_variables": [
                        {
                            "name": "v",
                            "symbol": "y_in[h,t]",
                            "type": "continuous",
                            "bounds": "0+",
                            "description": "d",
                        }
                    ],
                    "objective_function": {
                        "direction": "minimize",
                        "expression": "c * y_in[h,t]",
                        "description": "cost",
                    },
                    "constraints": [
                        {"name": "cap", "expression": "y_in[h,t] <= 1", "description": "d"}
                    ],
                },
            }
        )
        self.assertEqual(me.model_components.objective_function.direction, "min")
        self.assertEqual(me.model_components.decision_variables[0].indices, ["h", "t"])


if __name__ == "__main__":
    unittest.main()


class TestNestingDriftLift(unittest.TestCase):
    def test_lifts_top_level_objective_and_constraints_into_model_components(self):
        # Regression (2026-09-12, live on fam_H4_Omega10 / deepseek-flash): the
        # model hoisted objective_function and constraints to siblings of
        # model_components; the before-validator lifts them back in.
        me = ModelExpertOutput.model_validate(
            {
                "knowledge_requests": [],
                "model_components": {
                    "decision_variables": [
                        {"symbol": "y_in[h,t]", "type": "Continuous",
                         "description": "sea arrival"}
                    ]
                },
                "objective_function": {
                    "direction": "min",
                    "expression": "c * y_in[h,t]",
                    "description": "cost",
                },
                "constraints": [
                    {"name": "cap", "expression": "y_in[h,t] <= 1",
                     "description": "d"}
                ],
            }
        )
        self.assertEqual(
            me.model_components.objective_function.expression, "c * y_in[h,t]")
        self.assertEqual(len(me.model_components.constraints), 1)


class TestDuplicateConstraintNameDedup(unittest.TestCase):
    def test_duplicate_nonnegativity_names_are_renumbered(self):
        # Regression (2026-09-12, live on fam_H4_Omega10 freezes): the model
        # split nonnegativity into six same-named constraints and the
        # consistency validator rejected the formulation mid-campaign.
        me = ModelExpertOutput.model_validate(
            {
                "knowledge_requests": [],
                "model_components": {
                    "decision_variables": [
                        {"symbol": "y[h,t]", "type": "Continuous",
                         "description": "d"}
                    ],
                    "objective_function": {
                        "direction": "min",
                        "expression": "c * y[h,t]",
                        "description": "cost",
                    },
                    "constraints": [
                        {"name": "Nonnegativity",
                         "expression": "y[h,t] >= 0", "description": "d"},
                        {"name": "Nonnegativity",
                         "expression": "x[i,j,t] >= 0", "description": "d"},
                        {"name": "Capacity",
                         "expression": "y[h,t] <= 1", "description": "d"},
                        {"name": "Nonnegativity",
                         "expression": "I[i,t] >= 0", "description": "d"},
                    ],
                },
            }
        )
        names = [c.name for c in me.model_components.constraints]
        self.assertEqual(len(set(names)), len(names))
        self.assertEqual(names[0], "Nonnegativity")
        self.assertEqual(names[1], "Nonnegativity_2")
        self.assertEqual(names[2], "Capacity")
        self.assertEqual(names[3], "Nonnegativity_3")
