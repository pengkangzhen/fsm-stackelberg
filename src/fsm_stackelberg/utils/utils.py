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


def _load_csv_data(instance_dir: str) -> dict | None:
    """Load fundamental data from CSV files if they exist.

    Returns a dict with the same structure as sample.json but containing
    only fundamental data (no derived parameters like distances, allowed_arcs, etc.)
    """
    import csv as _csv

    nodes_csv = os.path.join(instance_dir, "nodes.csv")
    sd_csv = os.path.join(instance_dir, "supply_demand.csv")
    tm_csv = os.path.join(instance_dir, "transport_modes.csv")

    # Only use CSV if all three files exist
    if not all(os.path.exists(p) for p in [nodes_csv, sd_csv, tm_csv]):
        return None

    logger.info("Loading fundamental data from CSV files (no derived parameters)")

    # Parse nodes.csv
    nodes_list = []
    storage_cap = {}
    init_inv = {}
    holding_cost = {}
    renting_cost = {}

    with open(nodes_csv, encoding="utf-8") as f:
        reader = _csv.DictReader(f)
        for row in reader:
            name = row["node_name"]
            ntype = row["type"]
            lng = float(row["lng"]) if row["lng"] else None
            lat = float(row["lat"]) if row["lat"] else None
            nodes_list.append({"name": name, "type": ntype, "lng": lng, "lat": lat})

            # Storage parameters — only filled for seaports/dryports
            if row["storage_capacity"]:
                storage_cap[name] = int(float(row["storage_capacity"]))
                init_inv[name] = int(float(row["initial_inventory"]))
                holding_cost[name] = int(float(row["holding_cost"]))
                renting_cost[name] = int(float(row["renting_cost"]))

    # Parse supply_demand.csv (optional "commodity" column → multi-commodity data;
    # backward-compatible: absent column keeps the single-commodity behaviour)
    supply_records = []
    demand_records = []
    multi_commodity = False
    with open(sd_csv, encoding="utf-8") as f:
        reader = _csv.DictReader(f)
        multi_commodity = "commodity" in (reader.fieldnames or [])
        for row in reader:
            rec = {"node": row["node"], "period": int(row["period"]), "value": int(float(row["quantity"]))}
            if multi_commodity and row.get("commodity"):
                # supply: commodity is a type g (e.g. 20S/40S/20F/40F);
                # demand: commodity is a size σ (e.g. 20/40), pooled and substitutable.
                rec["commodity"] = row["commodity"]
            if row["type"] == "supply":
                supply_records.append(rec)
            else:
                demand_records.append(rec)

    # Parse transport_modes.csv
    mode_records = []
    utc_list = []
    with open(tm_csv, encoding="utf-8") as f:
        reader = _csv.DictReader(f)
        for row in reader:
            mode_records.append(row["mode"])
            utc_list.append(float(row["unit_cost_per_km"]))

    # Parse arcs.csv if present (pre-derived feasible arcs + distances + transit;
    # when present these are used directly and coordinate-based derivation skipped)
    allowed_transport = None
    transport_cost = None
    transit_time_matrix = None
    arcs_csv = os.path.join(instance_dir, "arcs.csv")
    if os.path.exists(arcs_csv):
        unit_cost_by_mode = dict(zip(mode_records, utc_list))
        allowed_transport, transport_cost, transit_time_matrix = [], [], []
        with open(arcs_csv, encoding="utf-8") as f:
            for row in _csv.DictReader(f):
                i, j, mode = row["from"], row["to"], row["mode"]
                distance = float(row["distance"])
                tau = int(float(row.get("transit_time") or 0))
                allowed_transport.append({"from": i, "to": j, "mode": mode})
                transport_cost.append({"from": i, "to": j, "mode": mode,
                                       "cost": distance * unit_cost_by_mode.get(mode, 0.0)})
                transit_time_matrix.append({"from": i, "to": j, "mode": mode, "time": tau})

    # Infer sets from nodes
    seaports = [n["name"] for n in nodes_list if n["type"] == "seaport"]
    dryports = [n["name"] for n in nodes_list if n["type"] == "dryport"]
    shippers = [n["name"] for n in nodes_list if n["type"] == "shipper"]
    consignees = [n["name"] for n in nodes_list if n["type"] == "consignee"]
    all_nodes = [n["name"] for n in nodes_list]

    # Infer periods from supply/demand data
    periods = sorted(set(r["period"] for r in supply_records))

    # Build nodes dict (with coordinates)
    nodes_dict = {}
    for n in nodes_list:
        nodes_dict[n["name"]] = {"type": n["type"], "location": {"lng": n["lng"], "lat": n["lat"]}}

    # Build sets (add multi-commodity metadata when a commodity column is present)
    sets = {
        "periods": periods,
        "transport_modes": mode_records,
        "node_types": ["seaport", "dryport", "shipper", "consignee"],
        "seaports": seaports,
        "dryports": dryports,
        "shippers": shippers,
        "consignees": consignees,
        "all_nodes": all_nodes,
    }
    if multi_commodity:
        sets["commodity_types"] = sorted({r["commodity"] for r in supply_records})
        sets["demand_sizes"] = sorted({r["commodity"] for r in demand_records})

    # Assemble result (same structure as sample.json). When arcs.csv is absent,
    # derived data (distance_matrix, allowed_transport, transit_time_matrix,
    # transport_cost, hinterland_*, street_turn) must be computed from domain
    # knowledge; when present, the arcs/dists/transits are used directly.
    result = {
        "sets": sets,
        "nodes": nodes_dict,
        "supply": supply_records,
        "demand": demand_records,
        "storage_capacity": storage_cap,
        "initial_inventory": init_inv,
        "holding_cost": holding_cost,
        "renting_cost": renting_cost,
        "unit_transport_cost": utc_list,
    }
    if allowed_transport is not None:
        result["allowed_transport"] = allowed_transport
        result["transport_cost"] = transport_cost
        result["transit_time_matrix"] = transit_time_matrix

    # Merge extra model parameters from params.json if present (e.g. theta,
    # c_fold, size_of mapping for multi-commodity instances)
    params_json = os.path.join(instance_dir, "params.json")
    if os.path.exists(params_json):
        import json as _json
        with open(params_json, encoding="utf-8") as f:
            extra_params = _json.load(f)
        result.update(extra_params)
        logger.info("Merged %d extra model parameters from params.json", len(extra_params))

    return result


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

    # Try loading fundamental data from CSV files first
    sample_data = _load_csv_data(sample_path)

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
