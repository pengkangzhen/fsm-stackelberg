"""Deterministic Gurobi solver for ECR ground-truth generation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gurobipy as gp
from gurobipy import GRB


def _records_to_dict(records: list[dict[str, Any]], key_fields: list[str], value_field: str) -> dict:
    """Convert records into a tuple-keyed dictionary."""
    return {
        tuple(record[field] for field in key_fields): record[value_field]
        for record in records
    }


def _build_shared_hinterland_pairs(sample: dict[str, Any]) -> set[tuple[str, str]]:
    consignee_by_dryport: dict[str, set[str]] = {}
    shipper_by_dryport: dict[str, set[str]] = {}

    for record in sample.get("hinterland_consignee", []):
        consignee_by_dryport.setdefault(record["dryport"], set()).add(record["consignee"])
    for record in sample.get("hinterland_shipper", []):
        shipper_by_dryport.setdefault(record["dryport"], set()).add(record["shipper"])

    pairs: set[tuple[str, str]] = set()
    for dryport in set(consignee_by_dryport) | set(shipper_by_dryport):
        for consignee in consignee_by_dryport.get(dryport, set()):
            for shipper in shipper_by_dryport.get(dryport, set()):
                pairs.add((consignee, shipper))
    return pairs


@dataclass(frozen=True)
class TruthModelData:
    periods: list[int]
    transport_modes: list[str]
    storage_nodes: list[str]
    seaports: list[str]
    dryports: list[str]
    shippers: list[str]
    consignees: list[str]
    all_nodes: list[str]
    allowed_arcs: list[tuple[str, str, str]]
    outgoing_arcs: dict[str, list[tuple[str, str, str]]]
    incoming_arcs: dict[str, list[tuple[str, str, str]]]
    supply: dict[tuple[str, int], int]
    demand: dict[tuple[str, int], int]
    transport_cost: dict[tuple[str, str, str], float]
    transit_time: dict[tuple[str, str, str], int]
    holding_cost: dict[str, float]
    renting_cost: dict[str, float]
    storage_capacity: dict[str, int]
    initial_inventory: dict[str, int]


def build_truth_model_data(sample: dict[str, Any]) -> TruthModelData:
    """Adapt generated instance data into deterministic solver inputs."""
    sets = sample["sets"]
    periods = [int(period) for period in sets["periods"]]
    transport_modes = list(sets["transport_modes"])
    all_nodes = list(sets["all_nodes"])
    storage_nodes = list(sets["seaports"]) + list(sets["dryports"])
    seaports = list(sets["seaports"])
    dryports = list(sets["dryports"])
    shippers = list(sets["shippers"])
    consignees = list(sets["consignees"])

    allowed_arc_candidates = [
        (record["from"], record["to"], record["mode"])
        for record in sample.get("allowed_transport", [])
    ]
    shared_hinterland_pairs = _build_shared_hinterland_pairs(sample)

    allowed_arcs: list[tuple[str, str, str]] = []
    for from_node, to_node, mode in allowed_arc_candidates:
        from_type = sample["nodes"][from_node]["type"]
        to_type = sample["nodes"][to_node]["type"]
        if (
            from_type == "consignee"
            and to_type == "shipper"
            and (from_node, to_node) not in shared_hinterland_pairs
        ):
            continue
        allowed_arcs.append((from_node, to_node, mode))

    allowed_arc_set = set(allowed_arcs)
    transport_cost_all = _records_to_dict(
        sample.get("transport_cost", []),
        ["from", "to", "mode"],
        "cost",
    )
    transit_time_all = _records_to_dict(
        sample.get("transit_time_matrix", []),
        ["from", "to", "mode"],
        "time",
    )

    missing_costs = sorted(arc for arc in allowed_arcs if arc not in transport_cost_all)
    if missing_costs:
        raise ValueError(
            "transport_cost is missing allowed arcs, e.g. "
            + ", ".join(map(str, missing_costs[:5]))
        )

    transport_cost = {arc: float(transport_cost_all[arc]) for arc in allowed_arcs}
    transit_time = {arc: int(transit_time_all.get(arc, 0)) for arc in allowed_arcs}

    outgoing_arcs = {node: [] for node in all_nodes}
    incoming_arcs = {node: [] for node in all_nodes}
    for arc in allowed_arcs:
        outgoing_arcs[arc[0]].append(arc)
        incoming_arcs[arc[1]].append(arc)

    supply = _records_to_dict(sample.get("supply", []), ["node", "period"], "value")
    demand = _records_to_dict(sample.get("demand", []), ["node", "period"], "value")

    return TruthModelData(
        periods=periods,
        transport_modes=transport_modes,
        storage_nodes=storage_nodes,
        seaports=seaports,
        dryports=dryports,
        shippers=shippers,
        consignees=consignees,
        all_nodes=all_nodes,
        allowed_arcs=allowed_arcs,
        outgoing_arcs=outgoing_arcs,
        incoming_arcs=incoming_arcs,
        supply=supply,
        demand=demand,
        transport_cost=transport_cost,
        transit_time=transit_time,
        holding_cost={node: float(value) for node, value in sample["holding_cost"].items()},
        renting_cost={node: float(value) for node, value in sample["renting_cost"].items()},
        storage_capacity={node: int(value) for node, value in sample["storage_capacity"].items()},
        initial_inventory={node: int(value) for node, value in sample.get("initial_inventory", {}).items()},
    )


def solve_truth_model(
    model_data: TruthModelData,
    *,
    time_limit: float | None = None,
    mip_gap: float = 0.0,
) -> dict[str, Any]:
    """Solve the adapted ECR instance with Gurobi."""
    model = gp.Model("ecr_ground_truth")
    model.Params.OutputFlag = 0
    model.Params.MIPGap = mip_gap
    if time_limit is not None:
        model.Params.TimeLimit = time_limit

    x_index = [
        (arc[0], arc[1], arc[2], period)
        for arc in model_data.allowed_arcs
        for period in model_data.periods
    ]
    x = model.addVars(x_index, vtype=GRB.INTEGER, lb=0.0, name="x")

    rent_index = [
        (node, period)
        for node in model_data.storage_nodes
        for period in model_data.periods
    ]
    rent = model.addVars(rent_index, vtype=GRB.INTEGER, lb=0.0, name="rent")
    inventory = model.addVars(rent_index, vtype=GRB.INTEGER, lb=0.0, name="inventory")

    transport_cost_expr = gp.quicksum(
        model_data.transport_cost[(i, j, mode)] * x[i, j, mode, period]
        for i, j, mode, period in x_index
    )
    holding_cost_expr = gp.quicksum(
        model_data.holding_cost[node] * inventory[node, period]
        for node, period in rent_index
    )
    renting_cost_expr = gp.quicksum(
        model_data.renting_cost[node] * rent[node, period]
        for node, period in rent_index
    )
    model.setObjective(transport_cost_expr + holding_cost_expr + renting_cost_expr, GRB.MINIMIZE)

    first_period = min(model_data.periods)

    for period in model_data.periods:
        for node in model_data.storage_nodes:
            previous_inventory = (
                inventory[node, period - 1]
                if period != first_period
                else model_data.initial_inventory.get(node, 0)
            )
            arrivals = gp.quicksum(
                x[i, j, mode, period - model_data.transit_time[(i, j, mode)]]
                for i, j, mode in model_data.incoming_arcs[node]
                if period - model_data.transit_time[(i, j, mode)] >= first_period
            )
            departures = gp.quicksum(
                x[i, j, mode, period]
                for i, j, mode in model_data.outgoing_arcs[node]
            )
            model.addConstr(
                previous_inventory
                + model_data.supply.get((node, period), 0)
                + rent[node, period]
                + arrivals
                == model_data.demand.get((node, period), 0)
                + departures
                + inventory[node, period],
                name=f"storage_balance[{node},{period}]",
            )
            model.addConstr(
                inventory[node, period] <= model_data.storage_capacity[node],
                name=f"storage_capacity[{node},{period}]",
            )

        for node in model_data.consignees:
            arrivals = gp.quicksum(
                x[i, j, mode, period - model_data.transit_time[(i, j, mode)]]
                for i, j, mode in model_data.incoming_arcs[node]
                if period - model_data.transit_time[(i, j, mode)] >= first_period
            )
            departures = gp.quicksum(
                x[i, j, mode, period]
                for i, j, mode in model_data.outgoing_arcs[node]
            )
            model.addConstr(
                model_data.supply.get((node, period), 0) + arrivals == departures,
                name=f"consignee_balance[{node},{period}]",
            )

        for node in model_data.shippers:
            arrivals = gp.quicksum(
                x[i, j, mode, period - model_data.transit_time[(i, j, mode)]]
                for i, j, mode in model_data.incoming_arcs[node]
                if period - model_data.transit_time[(i, j, mode)] >= first_period
            )
            departures = gp.quicksum(
                x[i, j, mode, period]
                for i, j, mode in model_data.outgoing_arcs[node]
            )
            model.addConstr(
                arrivals == model_data.demand.get((node, period), 0) + departures,
                name=f"shipper_balance[{node},{period}]",
            )

    model.optimize()

    status_name = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.UNBOUNDED: "UNBOUNDED",
    }.get(model.Status, str(model.Status))

    has_solution = model.SolCount > 0
    objective = float(model.ObjVal) if has_solution else None
    best_bound = float(model.ObjBound) if model.Status not in {GRB.LOADED} else None
    reported_gap = None
    if has_solution and model.Status != GRB.OPTIMAL:
        reported_gap = float(model.MIPGap)

    truth_type = (
        "optimal" if model.Status == GRB.OPTIMAL else "best_known" if has_solution else "infeasible"
    )

    return {
        "status": status_name,
        "objective": objective,
        "best_bound": best_bound,
        "mip_gap": 0.0 if model.Status == GRB.OPTIMAL and has_solution else reported_gap,
        "runtime_s": float(model.Runtime),
        "truth_type": truth_type,
        "sol_count": int(model.SolCount),
    }


def solve_instance_path(
    instance_dir: str | Path,
    *,
    time_limit: float | None = None,
    mip_gap: float = 0.0,
) -> dict[str, Any]:
    """Load an instance directory, solve it, and return a serializable result."""
    instance_dir = Path(instance_dir)
    sample = json.loads((instance_dir / "sample.json").read_text(encoding="utf-8"))
    metadata_path = instance_dir / "metadata.json"
    metadata = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    model_data = build_truth_model_data(sample)
    result = solve_truth_model(model_data, time_limit=time_limit, mip_gap=mip_gap)
    result["instance_name"] = instance_dir.name
    result["periods"] = len(model_data.periods)
    result["num_nodes"] = len(model_data.all_nodes)
    result["num_allowed_arcs"] = len(model_data.allowed_arcs)
    result["metadata"] = metadata
    return result


def write_truth_result(instance_dir: str | Path, result: dict[str, Any]) -> Path:
    """Write a truth result to ``optimal.json`` inside the instance directory."""
    instance_dir = Path(instance_dir)
    output_path = instance_dir / "optimal.json"
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output_path
