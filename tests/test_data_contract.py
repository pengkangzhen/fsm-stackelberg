"""Tests for the deterministic DE semantic-to-physical data contract."""

from fsm_stackelberg.data.data_contract import (
    build_data_catalog,
    build_pd_data_access_context,
    build_symbol_access_guide,
    format_data_catalog,
    validate_data_engineer_output,
)
from fsm_stackelberg.schemas import DataEngineerOutput


_SAMPLE = {
    "sets": {
        "hubs": ["h1"],
        "nodes": ["h1", "n1"],
        "periods": [1, 2],
        "modes": ["road"],
        "scenarios": [0, 1],
    },
    "topology": {
        "arcs": [
            {
                "from": "h1",
                "to": "n1",
                "mode": "road",
                "transit_time": 1,
                "capacity": 100.0,
                "unit_cost": 4.0,
            }
        ]
    },
    "first_stage": {
        "vessel_calls": [
            {"hub": "h1", "period": 1, "calls": 2},
            {"hub": "h1", "period": 2, "calls": 2},
        ]
    },
    "supply_demand": {
        "eta": [
            {
                "scenario": 0,
                "node": "n1",
                "period": 1,
                "value": 10.0,
            }
        ]
    },
}


def _valid_output() -> DataEngineerOutput:
    return DataEngineerOutput.model_validate(
        {
            "model_inputs": {
                "sets": [
                    {
                        "symbol": "H",
                        "index": "h",
                        "description": "hubs",
                        "source": "hubs",
                    },
                    {
                        "symbol": "N",
                        "index": "n",
                        "description": "nodes",
                        "source": "nodes",
                    },
                    {
                        "symbol": "T",
                        "index": "t",
                        "description": "periods",
                        "source": "periods",
                    },
                    {
                        "symbol": "M",
                        "index": "m",
                        "description": "modes",
                        "source": "modes",
                    },
                    {
                        "symbol": "K",
                        "index": "k",
                        "description": "scenarios",
                        "source": "scenarios",
                    },
                ],
                "parameters": [
                    {
                        "symbol": "V[h,t]",
                        "indices": [
                            {"symbol": "h", "set": "H"},
                            {"symbol": "t", "set": "T"},
                        ],
                        "description": "vessel calls",
                        "source": "first_stage.vessel_calls.calls",
                    },
                    {
                        "symbol": "A[i,j,m]",
                        "indices": [
                            {"symbol": "i", "set": "N"},
                            {"symbol": "j", "set": "N"},
                            {"symbol": "m", "set": "M"},
                        ],
                        "description": "allowed arcs",
                        "source": "topology.arcs",
                    },
                    {
                        "symbol": "U_arc[i,j,m]",
                        "indices": [
                            {"symbol": "i", "set": "N"},
                            {"symbol": "j", "set": "N"},
                            {"symbol": "m", "set": "M"},
                        ],
                        "description": "arc capacity",
                        "source": "topology.arcs.capacity",
                    },
                    {
                        "symbol": "eta[k,n,t]",
                        "indices": [
                            {"symbol": "k", "set": "K"},
                            {"symbol": "n", "set": "N"},
                            {"symbol": "t", "set": "T"},
                        ],
                        "description": "scenario demand",
                        "source": "supply_demand.eta.value",
                    },
                ],
                "derived_parameters": [],
            }
        }
    )


def test_catalog_exposes_exact_runtime_locations():
    catalog = build_data_catalog(_SAMPLE)

    assert catalog["hubs"]["access_pattern"] == "data['hubs']"
    assert catalog["topology.arcs.capacity"]["access_pattern"] == (
        "data['topology']['arcs']"
    )
    assert catalog["topology.arcs.capacity"]["index_fields"] == [
        "from",
        "to",
        "mode",
    ]
    assert catalog["topology.arcs.capacity"]["index_sources"] == [
        "nodes",
        "nodes",
        "modes",
    ]
    assert catalog["first_stage.vessel_calls.calls"]["value_field"] == "calls"

    rendered = format_data_catalog(catalog)
    assert "`topology.arcs.capacity`" in rendered
    assert "from→nodes" in rendered


def test_valid_aliases_pass_when_source_and_indices_are_explicit():
    violations = validate_data_engineer_output(
        _valid_output(),
        build_data_catalog(_SAMPLE),
    )
    assert violations == []


def test_symbol_guide_resolves_record_value_alias_from_catalog():
    guide = build_symbol_access_guide(
        _valid_output(),
        build_data_catalog(_SAMPLE),
    )

    vessel_calls = guide["parameters"]["V[h,t]"]
    assert vessel_calls["source"] == "first_stage.vessel_calls.calls"
    assert vessel_calls["access_pattern"] == (
        "data['first_stage']['vessel_calls']"
    )
    assert vessel_calls["kind"] == "record_value"
    assert vessel_calls["index_fields"] == ["hub", "period"]
    assert vessel_calls["index_sources"] == ["hubs", "periods"]
    assert vessel_calls["value_field"] == "calls"


def test_symbol_guide_resolves_records_alias_from_catalog():
    guide = build_symbol_access_guide(
        _valid_output(),
        build_data_catalog(_SAMPLE),
    )

    arcs = guide["parameters"]["A[i,j,m]"]
    assert arcs["source"] == "topology.arcs"
    assert arcs["access_pattern"] == "data['topology']['arcs']"
    assert arcs["kind"] == "records"
    assert arcs["index_fields"] == ["from", "to", "mode"]
    assert arcs["index_sources"] == ["nodes", "nodes", "modes"]
    assert arcs["value_field"] is None


def test_pd_forward_and_backward_share_deterministic_context_builder():
    output = _valid_output()
    catalog = build_data_catalog(_SAMPLE)
    physical_context = "## Schema\nsupplementary"

    forward_guide = build_pd_data_access_context(
        output,
        catalog,
        physical_context,
    )
    backward_guide = build_pd_data_access_context(
        output,
        catalog,
        physical_context,
    )

    assert forward_guide == backward_guide
    assert '"V[h,t]"' in forward_guide
    assert "## Physical Schema Context (supplementary)" in forward_guide


def test_symbol_guide_keeps_derived_parameter_metadata_without_physical_key():
    raw_output = _valid_output().model_dump()
    raw_output["model_inputs"]["derived_parameters"].append(
        {
            "symbol": "D[h,t]",
            "indices": [
                {"symbol": "h", "set": "H"},
                {"symbol": "t", "set": "T"},
            ],
            "description": "computed quantity",
            "derivation_logic": "D[h,t] = 2 * V[h,t]",
            "source_parameters": ["V[h,t]"],
        }
    )
    output = DataEngineerOutput.model_validate(raw_output)

    derived = build_symbol_access_guide(
        output,
        build_data_catalog(_SAMPLE),
    )["derived_parameters"]["D[h,t]"]

    assert derived["kind"] == "derived"
    assert derived["source"] is None
    assert derived["access_pattern"] is None
    assert derived["derivation_logic"] == "D[h,t] = 2 * V[h,t]"
    assert derived["source_parameters"] == ["V[h,t]"]


def test_section_source_and_reordered_indices_are_rejected():
    output = _valid_output().model_copy(deep=True)
    output.model_inputs.parameters[0].source = "first_stage"
    output.model_inputs.parameters[3].indices = [
        output.model_inputs.parameters[3].indices[1],  # N
        output.model_inputs.parameters[3].indices[2],  # T
        output.model_inputs.parameters[3].indices[0],  # K
    ]

    violations = validate_data_engineer_output(
        output,
        build_data_catalog(_SAMPLE),
    )
    codes = [violation.code for violation in violations]

    assert "unknown_parameter_source" in codes
    assert codes.count("index_domain_mismatch") == 3


def test_source_null_must_use_derived_parameters():
    output = _valid_output().model_copy(deep=True)
    output.model_inputs.parameters[0].source = None

    violations = validate_data_engineer_output(
        output,
        build_data_catalog(_SAMPLE),
    )
    assert any(
        violation.code == "derived_parameter_in_parameters"
        for violation in violations
    )
