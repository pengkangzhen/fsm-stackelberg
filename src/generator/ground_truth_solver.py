"""Ground-truth DEP solver for TSLP ECR demand-uncertainty instances."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tslp_ecr_demand.dep import solve_dep

from .serialize import sample_to_instance


def solve_ground_truth(sample: dict[str, Any]) -> dict[str, Any]:
    """Solve the extensive-form DEP and return status / objective."""
    inst, scenarios, probs = sample_to_instance(sample)
    result = solve_dep(inst, scenarios, probs)
    return {
        "status": result.status,
        "objective": float(result.objective),
        "total_spill": float(result.total_spill),
        "cost_breakdown": result.cost_breakdown,
        "first_stage": result.first_stage,
        "num_variables": result.num_variables,
        "num_constraints": result.num_constraints,
        "solver": "pulp_CBC_DEP",
    }


def write_optimal_json(instance_dir: str | Path) -> dict[str, Any]:
    """Load sample.json, solve DEP, write optimal.json."""
    path = Path(instance_dir)
    with open(path / "sample.json", encoding="utf-8") as f:
        sample = json.load(f)
    optimal = solve_ground_truth(sample)
    with open(path / "optimal.json", "w", encoding="utf-8") as f:
        json.dump(optimal, f, indent=2, ensure_ascii=False)
    return optimal
