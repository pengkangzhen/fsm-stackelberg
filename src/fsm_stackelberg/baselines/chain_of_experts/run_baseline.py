"""Chain-of-Experts baseline runner — integrated with MAKO experiment framework.

Usage:
    poetry run python -m fsm_stackelberg.baselines.chain_of_experts.run_baseline \
        --dataset prob_tslp_ecr_demand \
        --prob_name instances/small_5-3_5 \
        --provider DeepSeek --model deepseek-chat \
        --max_collaborate_nums 3 \
        --n_runs 20
"""

import argparse
import json
import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path

from langchain_community.callbacks import get_openai_callback

from fsm_stackelberg.utils.llm_config import DEFAULT_PROVIDER, DEFAULT_MODEL
from fsm_stackelberg.utils.utils import dataset_loader
from fsm_stackelberg.utils.experiment_result import ExperimentResult
from fsm_stackelberg.agents.solver_executor import sandbox_exec_code
from fsm_stackelberg.data.auto_preprocessor import auto_preprocess

from .main import chain_of_experts
from .utils import extract_code_from_string

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _build_data_description(sample: dict, schema: dict) -> str:
    """Build a human-readable description of the data structure."""
    lines = []
    sets = schema.get("sets", {})
    for key, value in sample.items():
        if key in sets:
            desc = sets[key].get("description", key)
            if isinstance(value, dict):
                # e.g. sets → {"periods": [0,1,2,...]}
                sub_keys = list(value.keys())
                lines.append(f"  data['{key}']: dict with keys {sub_keys}  ({desc})")
            elif isinstance(value, list) and value and isinstance(value[0], (int, float, str)):
                lines.append(f"  data['{key}']: list of {len(value)} items  ({desc})")
            elif isinstance(value, list) and value and isinstance(value[0], dict):
                cols = list(value[0].keys())
                lines.append(f"  data['{key}']: list of {len(value)} records, columns: {cols}  ({desc})")
            else:
                lines.append(f"  data['{key}']: {type(value).__name__}  ({desc})")
        else:
            if isinstance(value, dict):
                sub_keys = list(value.keys())[:5]
                lines.append(f"  data['{key}']: dict, sample keys: {sub_keys}")
            elif isinstance(value, list) and value and isinstance(value[0], dict):
                cols = list(value[0].keys())
                lines.append(f"  data['{key}']: list of {len(value)} records, columns: {cols}")
            elif isinstance(value, list):
                lines.append(f"  data['{key}']: list of {len(value)} items, e.g. {value[:3]}")
            else:
                lines.append(f"  data['{key}']: {value}")

    return "\n".join(lines)


def _build_code_template(sample: dict, schema: dict) -> str:
    """Build Gurobi starter code with actual data structure hints."""
    data_desc = _build_data_description(sample, schema)
    return f'''import gurobipy as gp
from gurobipy import GRB
import json

def optimize(data):
    """
    Solve the optimization problem using Gurobi.

    The `data` dict has the following structure:
{data_desc}

    Returns:
        dict with keys:
            - status: str, optimization status (e.g., 'OPTIMAL')
            - objective_value: float, optimal objective value
            - variables: dict, variable values
    """
    # Extract sets
    # periods = data["sets"]["periods"]
    # nodes = data["sets"]["all_nodes"]
    # ...

    # Extract parameters
    # supply = data["supply"]
    # demand = data["demand"]
    # ...

    model = gp.Model("optimization")

    # TODO: Define decision variables
    # TODO: Add constraints
    # TODO: Set objective function

    model.optimize()

    return {{
        "status": model.status,
        "objective_value": model.ObjVal,
        "variables": {{}},
    }}
'''


def extract_optimize_code(response: str) -> str | None:
    """Extract the optimize() function code from LLM response.

    Handles both markdown-wrapped code and raw code.
    """
    # First try extracting from code blocks
    code = extract_code_from_string(response)
    if "def optimize(" in code:
        return _clean_code(code)
    return None


def _clean_code(code: str) -> str:
    """Clean extracted code: remove markdown artifacts, ensure proper structure."""
    # Remove any remaining markdown markers
    code = re.sub(r"```python\s*", "", code)
    code = re.sub(r"```\s*", "", code)
    return code.strip()


def _setup_filepath(args) -> Path:
    """Generate output directory path matching MAKO's convention."""
    model_tag = args.model.replace("/", "-").replace(":", "-")
    filepath = (
        Path("results")
        / "coe"
        / f"{args.provider}_{model_tag}"
        / f"{args.dataset}_k{args.max_collaborate_nums}"
        / str(args.prob_name)
    )
    result_suffix = os.environ.get("MAKO_RESULT_SUFFIX", "").strip()
    if result_suffix:
        safe_suffix = re.sub(r"[^A-Za-z0-9._-]+", "_", result_suffix)
        filepath = filepath / safe_suffix
    filepath.mkdir(parents=True, exist_ok=True)
    return filepath


def run_single(
    problem: dict,
    provider: str,
    model: str,
    max_collaborate_nums: int,
) -> dict:
    """Run a single CoE baseline experiment.

    Args:
        problem: Problem data from dataset_loader().
        provider: LLM provider name.
        model: LLM model name.
        max_collaborate_nums: Number of expert collaboration rounds.

    Returns:
        Dict with experiment results.
    """
    start_time = time.time()

    description = problem.get("description", "")
    sample = problem.get("sample", {})

    result = {
        "algorithm": "coe",
        "dataset": problem.get("dataset", ""),
        "prob_name": problem.get("prob_name", ""),
        "provider": provider,
        "model": model,
        "status": False,
        "obj_value": None,
        "gurobi_status": "",
        "total_tokens": 0,
        "total_duration_s": 0.0,
        "error_msg": None,
        "steps": [],
        "max_collaborate_nums": max_collaborate_nums,
    }

    # Unified V2 data interface: auto_preprocess handles compression
    sample = problem.get("sample", {})
    processed_data, data_access_guide = auto_preprocess(sample)

    # Build code template with data guide
    code_template = f'''import gurobipy as gp
from gurobipy import GRB

def optimize(data):
    """
    Solve the optimization problem using Gurobi.

    The `data` dict has the following structure:
{data_access_guide}
    """
    # Extract sets and parameters from data dict above
    # TODO: Define decision variables
    # TODO: Add constraints
    # TODO: Set objective function

    model = gp.Model("optimization")
    model.optimize()

    return {{
        "status": model.status,
        "objective_value": model.ObjVal,
        "variables": {{}},
    }}
'''

    coe_problem = {
        "description": description + "\n\n## Problem Data\n\n" + data_access_guide,
        "code_example": code_template,
    }

    try:
        logger.info(f"Starting CoE baseline ({provider}/{model}, {max_collaborate_nums} rounds)")

        # Run CoE pipeline
        with get_openai_callback() as cb:
            raw_output = chain_of_experts(
                problem=coe_problem,
                max_collaborate_nums=max_collaborate_nums,
                model=model,
                provider=provider,
                enable_reflection=False,
            )
        pipeline_tokens = cb.total_tokens
        pipeline_duration = time.time() - start_time

        result["total_tokens"] += pipeline_tokens
        result["steps"].append({
            "step": "coe_pipeline",
            "duration_s": round(pipeline_duration, 3),
            "tokens": pipeline_tokens,
        })

        logger.info(f"CoE pipeline complete ({pipeline_tokens} tokens, {pipeline_duration:.2f}s)")

        # Extract code
        code = extract_optimize_code(raw_output)
        if not code:
            result["error_msg"] = "Could not extract optimize() function from CoE output"
            logger.error(result["error_msg"])
            # Save raw output for debugging
            result["raw_output"] = raw_output[:2000]
            return result

        # Execute code using processed data
        exec_start = time.time()
        exec_report = sandbox_exec_code(processed_data, code)
        exec_duration = time.time() - exec_start

        result["steps"].append({
            "step": "code_execution",
            "duration_s": round(exec_duration, 3),
            "tokens": 0,
        })

        # Process execution result
        if exec_report and exec_report.get("diagnosis_required") is False:
            result["status"] = True
            exec_result = exec_report.get("result") or {}
            result["obj_value"] = exec_result.get("objective_value")
            result["gurobi_status"] = exec_report.get("gurobi_status", "")
            logger.info(f"CoE: Success! Objective = {result['obj_value']}")
        else:
            if exec_report:
                result["gurobi_status"] = exec_report.get("gurobi_status", "")
                error_details = exec_report.get("error_details") or {}
                gurobi_status = exec_report.get("gurobi_status", "unknown")
                if isinstance(error_details, dict):
                    result["error_msg"] = error_details.get("error_message") or f"Gurobi status: {gurobi_status}"
                else:
                    result["error_msg"] = str(error_details) if error_details else f"Gurobi status: {gurobi_status}"
            else:
                result["error_msg"] = "Execution returned no report"
            logger.error(f"CoE: Failed - {result['error_msg']}")

    except Exception as e:
        result["error_msg"] = str(e)
        logger.error(f"CoE: Exception - {e}", exc_info=True)

    result["total_duration_s"] = round(time.time() - start_time, 3)
    return result


def main():
    parser = argparse.ArgumentParser(description="Chain-of-Experts Baseline Runner")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--prob_name", type=str, default=".", help="Problem instance path")
    parser.add_argument("--provider", type=str, default=DEFAULT_PROVIDER, help="LLM provider")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="LLM model")
    parser.add_argument("--max_collaborate_nums", type=int, default=3,
                        help="Number of expert collaboration rounds")
    parser.add_argument("--n_runs", type=int, default=1, help="Number of runs for statistical analysis")
    args = parser.parse_args()

    # Load problem
    problem = dataset_loader(args.dataset, args.prob_name)
    problem["prob_name"] = args.prob_name

    filepath = _setup_filepath(args)
    logger.info(f"Results will be saved to: {filepath}")

    # Run experiments
    all_results = []
    for run_idx in range(args.n_runs):
        logger.info(f"=== Run {run_idx + 1}/{args.n_runs} ===")
        run_result = run_single(
            problem=problem,
            provider=args.provider,
            model=args.model,
            max_collaborate_nums=args.max_collaborate_nums,
        )
        all_results.append(run_result)

        # Build and save ExperimentResult
        exp = ExperimentResult(
            timestamp=datetime.now().isoformat(),
            algorithm="coe",
            dataset=args.dataset,
            prob_name=str(args.prob_name),
            max_collaborate_nums=args.max_collaborate_nums,
            is_backtrack=False,
            provider=args.provider,
            model=args.model,
            orchestrator_mode="",
            knowledge_enabled=False,
            knowledge_mode="disable",
            knowledge_excluded_modules=[],
            agent_configs=[],
            steps=run_result.get("steps", []),
            total_duration_s=run_result.get("total_duration_s", 0.0),
            total_tokens=run_result.get("total_tokens", 0),
            num_forward_steps=args.max_collaborate_nums + 1,  # experts + reducer
            num_backward_steps=0,
            status=bool(run_result.get("status")),
            error_msg=run_result.get("error_msg") or "",
            obj_value=run_result.get("obj_value"),
            expected_value=problem.get("expected_value"),
            is_model_valid=None,
            gurobi_status=run_result.get("gurobi_status", ""),
            error_agent="",
            error_details={"error_message": run_result.get("error_msg")} if run_result.get("error_msg") else {},
            error_category="none",
            constraints=[],
            agent_metrics=[],
            node_metrics=[],
            backtrack_history=[],
            result_path=str(filepath),
        )
        exp.save_json(str(filepath))
        exp.append_summary_json("results/experiment_summary.json")

    # Print summary
    success_count = sum(1 for r in all_results if r["status"])
    total_runs = len(all_results)
    avg_duration = sum(r["total_duration_s"] for r in all_results) / total_runs
    avg_tokens = sum(r["total_tokens"] for r in all_results) / total_runs

    obj_values = [r["obj_value"] for r in all_results if r["obj_value"] is not None]
    expected = problem.get("expected_value")

    print(f"\n{'='*60}")
    print(f"Chain-of-Experts Baseline Summary")
    print(f"{'='*60}")
    print(f"Problem: {args.dataset}/{args.prob_name}")
    print(f"LLM: {args.provider}/{args.model}")
    print(f"Collaboration rounds: {args.max_collaborate_nums}")
    print(f"Runs: {total_runs}")
    print(f"SSR (Solution Success Rate): {success_count}/{total_runs} = {success_count/total_runs*100:.1f}%")
    print(f"Avg Runtime: {avg_duration:.2f}s")
    print(f"Avg Tokens: {avg_tokens:.0f}")
    if obj_values and expected:
        gaps = [abs(v - expected) / abs(expected) * 100 for v in obj_values]
        print(f"Obj Gap: {sum(gaps)/len(gaps):.2f}% (avg over successful runs)")
    print(f"Results saved to: {filepath}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
