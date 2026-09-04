"""Dataset loading and result serialization utilities."""

import glob
import logging
import os
import json
import shutil
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional

logger = logging.getLogger(__name__)


def read_file(dirpath: str, filename: str) -> str:
    filepath = os.path.join(dirpath, filename)
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


_AGENT_CLASS_NAMES = ["DataEngineer", "ModelExpert", "PyDeveloper", "SolverExecutor"]
_ARCHIVE_DIR = ".archive"


def _archive_existing_results(filepath: str):
    """Move existing result files to a timestamped archive subdirectory."""
    files_to_archive = []
    for agent in _AGENT_CLASS_NAMES:
        base = os.path.join(filepath, f"{agent}.txt")
        if os.path.exists(base):
            files_to_archive.append(base)
        for f in glob.glob(os.path.join(filepath, f"{agent}_round*.txt")):
            files_to_archive.append(f)

    er = os.path.join(filepath, "experiment_result.json")
    if os.path.exists(er):
        files_to_archive.append(er)

    if not files_to_archive:
        return

    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    archive_dir = os.path.join(filepath, _ARCHIVE_DIR, timestamp)
    os.makedirs(archive_dir, exist_ok=True)

    for f in files_to_archive:
        shutil.move(f, os.path.join(archive_dir, os.path.basename(f)))


def save_workflow_result(result: Dict[str, str], filepath: str):
    for agent, msg in result.items():
        filename = os.path.join(filepath, f"{agent}.txt")
        with open(filename, "w", encoding="utf-8") as f:
            if msg is None:
                f.write(f"No response from {agent}")
            else:
                f.write(msg)
    print("workflow result saved successfully")


# =============================================================================
# Output History Management
# =============================================================================

def record_agent_output(
    output_history: List[Dict],
    agent_name: str,
    output: Any,
    current_round: int,
    step_type: str = "forward"
) -> List[Dict]:
    """Record an agent output to history.

    Args:
        output_history: Existing history list
        agent_name: Name of the agent (e.g., "data_engineer")
        output: The output object (Pydantic model or string)
        current_round: Current round number
        step_type: "forward" or "backward"

    Returns:
        Updated history list
    """
    record = {
        "round": current_round,
        "agent": agent_name,
        "output": output,
        "timestamp": datetime.now().isoformat(),
        "step_type": step_type,
    }
    return output_history + [record]


def get_outputs_by_round(output_history: List[Dict], round_num: int) -> Dict[str, Any]:
    """Get all agent outputs for a specific round.

    Args:
        output_history: History list
        round_num: Round number to filter

    Returns:
        Dict mapping agent names to their outputs
    """
    return {
        record["agent"]: record["output"]
        for record in output_history
        if record["round"] == round_num
    }


def get_outputs_by_agent(output_history: List[Dict], agent_name: str) -> List[Dict]:
    """Get all outputs from a specific agent across rounds.

    Args:
        output_history: History list
        agent_name: Agent name to filter

    Returns:
        List of output records for the agent
    """
    return [
        record for record in output_history
        if record["agent"] == agent_name
    ]


def get_last_forward_output(output_history: List[Dict], agent_name: str) -> str:
    """Get the last forward output from a specific agent.

    Args:
        output_history: History list
        agent_name: Agent name to filter

    Returns:
        String representation of the last forward output, or "N/A" if not found
    """
    for record in reversed(output_history):
        if record["agent"] == agent_name and record.get("step_type") == "forward":
            output = record["output"]
            if hasattr(output, 'model_dump_json'):
                return output.model_dump_json(indent=2)
            return str(output)
    return "N/A"


def _agent_name_to_class(agent_name: str) -> str:
    """Convert agent name to class name for file naming."""
    mapping = {
        "data_engineer": "DataEngineer",
        "model_expert": "ModelExpert",
        "python_developer": "PyDeveloper",
        "solver_executor": "SolverExecutor",
    }
    return mapping.get(agent_name, agent_name.title())


def _format_output(output: Any) -> str:
    """Format agent output to string for saving."""
    if output is None:
        return "No output"
    if hasattr(output, 'model_dump_json'):
        return output.model_dump_json(indent=2)
    return str(output)


def save_workflow_result_with_history(state: dict, filepath: str):
    """Save workflow results with history support.

    Saves:
    - {Agent}_round{N}.txt for each round
    - {Agent}.txt for the final successful output
    """
    output_history = state.get("output_history", [])

    if not output_history:
        # No history, just save current outputs
        from ..main import _build_agent_outputs
        save_workflow_result(_build_agent_outputs(state), filepath)
        return

    # Archive existing results before saving new ones
    _archive_existing_results(filepath)

    # Group by round
    rounds: Dict[int, Dict[str, Any]] = {}
    for record in output_history:
        r = record["round"]
        if r not in rounds:
            rounds[r] = {}
        rounds[r][record["agent"]] = record["output"]

    if len(rounds) == 1:
        # Single round: save as plain {Agent}.txt
        from ..main import _build_agent_outputs
        save_workflow_result(_build_agent_outputs(state), filepath)
    else:
        # Multi-round: save each round as {Agent}_round{N}.txt only
        for round_num, agents_output in rounds.items():
            for agent_name, output in agents_output.items():
                content = _format_output(output)
                class_name = _agent_name_to_class(agent_name)
                filename = os.path.join(filepath, f"{class_name}_round{round_num}.txt")
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(content)

    print(f"workflow result saved with {len(rounds)} round(s) history")


def _load_multi_source(instance_dir: str) -> dict | None:
    """Load two-stage ECR multi-source files into a sample-equivalent dict.

    Reads the MAKO-style per-instance layout produced by
    ``sample_to_multi_source_files`` and reconstructs a dict structurally
    identical to ``sample.json``.  Returns ``None`` when the directory does
    not contain the multi-source layout so the caller can fall back to the
    consolidated ``sample.json``.
    """
    import csv as _csv

    meta_path = os.path.join(instance_dir, "metadata.json")
    if not os.path.exists(meta_path):
        return None
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    sets = meta.get("sets")
    if not sets:  # legacy metadata.json without the multi-source contract
        return None

    logger.info("Loading two-stage ECR instance from multi-source files")

    def _read_csv(name: str) -> list[dict]:
        path = os.path.join(instance_dir, name)
        if not os.path.exists(path):
            return []
        with open(path, encoding="utf-8") as fh:
            return list(_csv.DictReader(fh))

    def _num(val: Any) -> float | None:
        return float(val) if val not in (None, "") else None

    def _int(val: Any) -> int | None:
        return int(float(val)) if val not in (None, "") else None

    # --- topology ------------------------------------------------------------
    arcs = [
        {
            "from": r["from"],
            "to": r["to"],
            "mode": r["mode"],
            "distance_km": _num(r.get("distance_km")),
            "transit_time": _int(r.get("transit_time")),
            "capacity": _num(r.get("capacity")),
            "unit_cost": _num(r.get("unit_cost")),
        }
        for r in _read_csv("arcs.csv")
    ]
    lambda_hl = [
        {"sea": r["sea"], "dry": r["dry"], "linked": _int(r.get("linked"))}
        for r in _read_csv("lambda_hl.csv")
    ]

    # --- first_stage ---------------------------------------------------------
    fs_path = os.path.join(instance_dir, "first_stage.json")
    fs: dict = {}
    if os.path.exists(fs_path):
        with open(fs_path, encoding="utf-8") as f:
            fs = json.load(f)
    vessel_calls = [
        {"hub": r["hub"], "period": _int(r["period"]), "calls": float(r["calls"])}
        for r in _read_csv("vessel_calls.csv")
    ]

    # --- supply_demand (split the outer-joined table back out) --------------
    xi: list[dict] = []
    eta_bar: list[dict] = []
    e_records: list[dict] = []
    for r in _read_csv("supply_demand.csv"):
        node = r["node"]
        period = _int(r["period"])
        if _num(r.get("xi")) is not None:
            xi.append({"node": node, "period": period, "value": _num(r["xi"])})
        if _num(r.get("eta_bar")) is not None:
            eta_bar.append({"node": node, "period": period, "value": _num(r["eta_bar"])})
        if _num(r.get("E")) is not None:
            e_records.append({"node": node, "period": period, "value": _num(r["E"])})
    eta = [
        {
            "scenario": _int(r["scenario"]),
            "node": r["node"],
            "period": _int(r["period"]),
            "value": _num(r["eta"]),
        }
        for r in _read_csv("scenarios.csv")
    ]
    scenario_probability = [
        {"scenario": _int(r["scenario"]), "probability": _num(r["probability"])}
        for r in _read_csv("scenario_probability.csv")
    ]

    # --- inventory -----------------------------------------------------------
    inv_fields = ["I0", "U_cap", "c_hold", "c_lease"]
    inv_blocks: dict[str, list[dict]] = {f: [] for f in inv_fields}
    for r in _read_csv("inventory.csv"):
        for f in inv_fields:
            if _num(r.get(f)) is not None:
                inv_blocks[f].append({"node": r["node"], "value": _num(r[f])})

    return {
        "sets": sets,
        "topology": {"lambda_hl": lambda_hl, "arcs": arcs},
        "first_stage": {
            "vessel_calls": vessel_calls,
            "B_in": fs.get("B_in", []),
            "B_out": fs.get("B_out", []),
            "D_ext_eff": fs.get("D_ext_eff", []),
            "c_sea_in": fs.get("c_sea_in"),
            "c_sea_out": fs.get("c_sea_out"),
        },
        "supply_demand": {
            "xi": xi,
            "E": e_records,
            "eta_bar": eta_bar,
            "eta": eta,
            "scenario_probability": scenario_probability,
        },
        "inventory": {
            "I0": inv_blocks["I0"],
            "U_cap": inv_blocks["U_cap"],
            "c_hold": inv_blocks["c_hold"],
            "c_lease": inv_blocks["c_lease"],
            "c_spill": fs.get("c_spill"),
        },
        "Q_bar": [],
        "_meta": meta.get("_meta", {}),
    }


def dataset_loader(dataset: str, prob_name: str) -> dict:
    root_dir = f"./dataset/{dataset}"

    # Try instances subdirectory first
    instance_dir = f"./dataset/{dataset}/instances/{prob_name}"
    if os.path.exists(os.path.join(instance_dir, "sample.json")):
        sample_path = instance_dir
    else:
        sample_path = f"./dataset/{dataset}/{prob_name}"

    # description.txt always from problem root
    description: str = read_file(root_dir, "description.txt")

    # Try loading from multi-source CSV/JSON files first, fall back to sample.json
    sample_data = _load_multi_source(sample_path)

    if sample_data is None:
        # Fallback to full sample.json
        sample_data = json.loads(read_file(sample_path, "sample.json"))

    expected_value = None
    optimal_path = os.path.join(sample_path, "optimal.json")
    if os.path.exists(optimal_path):
        optimal_data = json.loads(read_file(sample_path, "optimal.json"))
        expected_value = optimal_data.get("objective")

    # Load _meta if not present in sample.json
    if isinstance(sample_data, dict) and "_meta" not in sample_data:
        meta_path = os.path.join(root_dir, "_meta.json")
        if os.path.exists(meta_path):
            sample_data["_meta"] = json.loads(read_file(root_dir, "_meta.json"))

    # Build lightweight schema for DataEngineer
    schema = {
        "sets": {k: {"description": f"Set of {k}"} for k in sample_data.get("sets", {}).keys()},
        "_meta": sample_data.get("_meta", {}),
    }

    return {
        "description": description,
        "sample": sample_data,
        "schema": schema,
        "dataset": dataset,
        "expected_value": expected_value,
    }
