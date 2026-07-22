"""TSLP instance export / load helpers for the agentic pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tslp_ecr_demand.dep import solve_dep
from tslp_ecr_demand.instance import (
    DemandInstance,
    build_verify_instance,
    sample_demand_scenarios,
)


def _kv_records(d: dict, key_names: list[str], value_name: str = "value") -> list[dict]:
    records = []
    for key, val in d.items():
        if not isinstance(key, tuple):
            key = (key,)
        rec = {name: key[i] for i, name in enumerate(key_names)}
        rec[value_name] = float(val) if isinstance(val, (int, float)) else val
        records.append(rec)
    return records


def instance_to_sample(
    inst: DemandInstance,
    scenarios: list[dict[tuple[str, int], float]],
    probabilities: list[float],
) -> dict[str, Any]:
    """Serialize a TSLP window instance + demand scenarios to agent-facing JSON."""
    hubs, spokes, dry = inst.hubs, inst.spokes, inst.dry_ports
    nodes = hubs + spokes + dry
    omega = list(range(len(scenarios)))

    lambda_records = [
        {"sea": s, "dry": d, "linked": int(v)}
        for (s, d), v in inst.lambda_hl.items()
        if v
    ]
    arc_records = [
        {
            "from": i,
            "to": j,
            "mode": k,
            "transit_time": int(inst.tau[i, j, k]),
            "distance_km": float(inst.dist_km[i, j]),
            "capacity": float(inst.F_cap[i, j, k]),
            "unit_cost": float(inst.c_land[i, j, k]),
        }
        for (i, j, k) in inst.arcs
    ]
    eta_records = []
    for w, eta in enumerate(scenarios):
        for (n, t), val in eta.items():
            eta_records.append(
                {"scenario": w, "node": n, "period": int(t), "value": float(val)}
            )

    return {
        "sets": {
            "hubs": list(hubs),
            "spokes": list(spokes),
            "dry_ports": list(dry),
            "seas": list(hubs + spokes),
            "nodes": list(nodes),
            "periods": [int(t) for t in inst.periods],
            "modes": list(inst.modes),
            "scenarios": omega,
        },
        "topology": {
            "lambda_hl": lambda_records,
            "arcs": arc_records,
        },
        "first_stage": {
            "vessel_calls": _kv_records(inst.V, ["hub", "period"], "calls"),
            "B_in": _kv_records(inst.B_in, ["hub"], "value"),
            "B_out": _kv_records(inst.B_out, ["hub"], "value"),
            "D_ext_eff": _kv_records(inst.D_ext_eff, ["hub", "period"], "value"),
            "c_sea_in": float(inst.c_sea_in),
            "c_sea_out": float(inst.c_sea_out),
        },
        "supply_demand": {
            "xi": _kv_records(inst.xi, ["node", "period"], "value"),
            "E": _kv_records(inst.E, ["node", "period"], "value"),
            "eta_bar": _kv_records(inst.eta_bar, ["node", "period"], "value"),
            "eta": eta_records,
            "scenario_probability": [
                {"scenario": w, "probability": float(p)} for w, p in enumerate(probabilities)
            ],
        },
        "inventory": {
            "I0": _kv_records(inst.I0, ["node"], "value"),
            "U_cap": _kv_records(inst.U_cap, ["node"], "value"),
            "c_hold": _kv_records(inst.c_hold, ["node"], "value"),
            "c_lease": _kv_records(inst.c_lease, ["node"], "value"),
            "c_spill": float(inst.c_spill),
        },
        "Q_bar": [],  # committed inland arrivals; empty for first-window smoke
        "_meta": {
            "problem": "tslp_ecr_demand",
            "formulation": "two_stage_DEP_D1_D2",
            "variable_domain": "continuous_nonnegative_TEU",
            "uncertainty": "export_demand_eta_omega",
            "supply": "exogenous_deterministic_xi",
            "instance": dict(inst.meta),
            "n_scenarios": len(scenarios),
        },
    }


def _records_to_dict(records: list[dict], key_fields: list[str], value_field: str = "value") -> dict:
    out = {}
    for rec in records:
        key = tuple(rec[f] if f != "period" else int(rec[f]) for f in key_fields)
        if len(key) == 1:
            out[key[0]] = rec[value_field]
        else:
            out[key] = rec[value_field]
    return out


def sample_to_instance(
    sample: dict[str, Any],
) -> tuple[DemandInstance, list[dict[tuple[str, int], float]], list[float]]:
    """Rebuild DemandInstance + scenarios from exported sample.json."""
    sets = sample["sets"]
    topo = sample["topology"]
    fs = sample["first_stage"]
    sd = sample["supply_demand"]
    inv = sample["inventory"]

    arcs = [(a["from"], a["to"], a["mode"]) for a in topo["arcs"]]
    tau = {(a["from"], a["to"], a["mode"]): int(a["transit_time"]) for a in topo["arcs"]}
    dist_km = {}
    F_cap = {}
    c_land = {}
    for a in topo["arcs"]:
        dist_km[a["from"], a["to"]] = float(a["distance_km"])
        F_cap[a["from"], a["to"], a["mode"]] = float(a["capacity"])
        c_land[a["from"], a["to"], a["mode"]] = float(a["unit_cost"])

    lambda_hl = {
        (r["sea"], r["dry"]): int(r["linked"]) for r in topo["lambda_hl"]
    }

    V = {}
    for r in fs["vessel_calls"]:
        V[r["hub"], int(r["period"])] = int(r["calls"])

    B_in = {r["hub"]: float(r["value"]) for r in fs["B_in"]}
    B_out = {r["hub"]: float(r["value"]) for r in fs["B_out"]}
    D_ext_eff = {
        (r["hub"], int(r["period"])): float(r["value"]) for r in fs["D_ext_eff"]
    }
    # D_ext unused in DEP beyond D_ext_eff; keep parallel
    D_ext = dict(D_ext_eff)

    xi = {(r["node"], int(r["period"])): float(r["value"]) for r in sd["xi"]}
    E = {(r["node"], int(r["period"])): float(r["value"]) for r in sd.get("E", [])}
    eta_bar = {(r["node"], int(r["period"])): float(r["value"]) for r in sd["eta_bar"]}

    I0 = {r["node"]: float(r["value"]) for r in inv["I0"]}
    U_cap = {r["node"]: float(r["value"]) for r in inv["U_cap"]}
    c_hold = {r["node"]: float(r["value"]) for r in inv["c_hold"]}
    c_lease = {r["node"]: float(r["value"]) for r in inv["c_lease"]}

    inst = DemandInstance(
        hubs=list(sets["hubs"]),
        spokes=list(sets["spokes"]),
        dry_ports=list(sets["dry_ports"]),
        periods=[int(t) for t in sets["periods"]],
        modes=list(sets["modes"]),
        lambda_hl=lambda_hl,
        arcs=arcs,
        tau=tau,
        dist_km=dist_km,
        V=V,
        B_in=B_in,
        B_out=B_out,
        D_ext=D_ext,
        D_ext_eff=D_ext_eff,
        E=E,
        xi=xi,
        eta_bar=eta_bar,
        I0=I0,
        U_cap=U_cap,
        F_cap=F_cap,
        c_sea_in=float(fs["c_sea_in"]),
        c_sea_out=float(fs["c_sea_out"]),
        c_land=c_land,
        c_hold=c_hold,
        c_lease=c_lease,
        c_spill=float(inv["c_spill"]),
        meta=dict(sample.get("_meta", {}).get("instance", {})),
    )

    scenarios: list[dict[tuple[str, int], float]] = [
        {} for _ in sets["scenarios"]
    ]
    for r in sd["eta"]:
        w = int(r["scenario"])
        scenarios[w][r["node"], int(r["period"])] = float(r["value"])

    probs = [0.0] * len(scenarios)
    for r in sd["scenario_probability"]:
        probs[int(r["scenario"])] = float(r["probability"])
    if abs(sum(probs) - 1.0) > 1e-6:
        probs = [1.0 / len(scenarios)] * len(scenarios)

    return inst, scenarios, probs


def build_smoke_export(
    *,
    T: int = 4,
    n_scenarios: int = 5,
    seed: int = 42,
    sdr: float = 1.0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build default smoke instance (single window = full horizon T=H)."""
    inst = build_verify_instance(T=T, seed=seed, sdr=sdr)
    scenarios, probs = sample_demand_scenarios(inst, n_scenarios, seed=seed)
    sample = instance_to_sample(inst, scenarios, probs)
    result = solve_dep(inst, scenarios, probs)
    optimal = {
        "status": result.status,
        "objective": result.objective,
        "solver": "pulp_CBC_DEP",
        "total_spill": result.total_spill,
        "cost_breakdown": result.cost_breakdown,
        "first_stage": result.first_stage,
        "num_variables": result.num_variables,
        "num_constraints": result.num_constraints,
        "meta": {
            "T": T,
            "n_scenarios": n_scenarios,
            "seed": seed,
            "sdr": sdr,
            "source": "tslp-ecr-demand",
        },
    }
    return sample, optimal


def write_instance(output_dir: str | Path, sample: dict, optimal: dict) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "sample.json", "w", encoding="utf-8") as f:
        json.dump(sample, f, indent=2, ensure_ascii=False)
    with open(out / "optimal.json", "w", encoding="utf-8") as f:
        json.dump(optimal, f, indent=2, ensure_ascii=False)
    with open(out / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "problem": "tslp_ecr_demand",
                "formulation": "two_stage_DEP",
                "optimal_status": optimal.get("status"),
                "objective": optimal.get("objective"),
                "instance_meta": sample.get("_meta", {}),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    return out
