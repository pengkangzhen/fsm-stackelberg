"""OptiMUS baseline runner — integrated with MAKO experiment framework.

Usage:
    poetry run python -m fsm_stackelberg.baselines.optimus.run_baseline \
        --dataset prob_ecr_shipper_consignee \
        --prob_name instances/small_5-3_5 \
        --provider DeepSeek --model deepseek-chat \
        --max_selections 8 \
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

from fsm_stackelberg.utils.llm_config import DEFAULT_PROVIDER, DEFAULT_MODEL
from fsm_stackelberg.utils.utils import dataset_loader
from fsm_stackelberg.utils.experiment_result import ExperimentResult
from fsm_stackelberg.data.auto_preprocessor import auto_preprocess

from .agents import Agent
from .formulator import Formulator
from .programmer import Programmer
from .evaluator import Evaluator
from .manager import GroupChatManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _build_state(problem: dict) -> dict:
    """Build OptiMUS-compatible state from MAKO problem dict.

    Uses unified V2 data interface (auto_preprocessor).
    """
    description = problem.get("description", "")
    sample = problem.get("sample", {})

    # Unified V2 data interface
    processed_data, data_access_guide = auto_preprocess(sample)

    # Extract parameter names for the Formulator
    parameters = {}
    if isinstance(sample, dict):
        if "sets" in sample:
            parameters["sets"] = {k: f"Set of {k}" for k in sample["sets"].keys()}
        for key in sample:
            if not key.startswith("_") and key != "sets":
                parameters[key] = f"Parameter: {key}"

    # Build initial constraint/objective list
    constraints = [{
        "description": description,
        "status": "not_formulated",
    }]
    objective = [{
        "description": description,
        "status": "not_formulated",
    }]

    return {
        "background": description,
        "problem_type": "LP",
        "parameters": parameters,
        "data_description": data_access_guide,
        "constraint": constraints,
        "objective": objective,
        "variables": [],
        "solution_status": None,
        "solver_output_status": None,
        "error_message": None,
        "obj_val": None,
        "code": None,
        "data": processed_data,
        "execution_result": None,
    }


def _setup_filepath(args) -> Path:
    """Generate output directory path matching MAKO's convention."""
    model_tag = args.model.replace("/", "-").replace(":", "-")
    filepath = (
        Path("results")
        / "optimus"
        / f"{args.provider}_{model_tag}"
        / f"{args.dataset}_k{args.max_selections}"
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
    max_selections: int,
    filepath: Path | None = None,
) -> dict:
    """Run a single OptiMUS baseline experiment."""
    start_time = time.time()

    result = {
        "algorithm": "optimus",
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
        "max_selections": max_selections,
    }

    try:
        # Build state from problem
        state = _build_state(problem)

        # Create agents and manager
        formulator = Formulator(model=model, provider=provider)
        programmer = Programmer(model=model, provider=provider)
        evaluator = Evaluator(model=model, provider=provider)

        manager = GroupChatManager(
            agents=[formulator, programmer, evaluator],
            model=model,
            provider=provider,
            max_selections=max_selections,
        )

        logger.info(f"Starting OptiMUS baseline ({provider}/{model}, max {max_selections} selections)")

        # Run the manager loop
        from langchain_community.callbacks import get_openai_callback

        with get_openai_callback() as cb:
            final_state = manager.solve(state=state)

        total_tokens = cb.total_tokens
        duration = time.time() - start_time

        result["total_tokens"] = total_tokens
        result["total_duration_s"] = round(duration, 3)
        result["steps"].append({
            "step": "optimus_pipeline",
            "duration_s": round(duration, 3),
            "tokens": total_tokens,
            "num_selections": len(final_state.get("history", [])),
        })

        # Check final state — accept both "optimal" (Programmer) and "correct" (Evaluator)
        if final_state.get("solution_status") in ("optimal", "correct"):
            result["status"] = True
            result["obj_value"] = final_state.get("obj_val")
            result["gurobi_status"] = final_state.get("solver_output_status", "")
            logger.info(f"OptiMUS: Success! Objective = {result['obj_value']}")
        else:
            result["error_msg"] = final_state.get("error_message", "Failed to find solution")
            result["gurobi_status"] = final_state.get("solver_output_status", "")
            logger.warning(f"OptiMUS: Failed - {result['error_msg']}")

        # Save generated code for debugging
        if final_state.get("code"):
            code_path = filepath / "generated_code.py"
            try:
                code_path.parent.mkdir(parents=True, exist_ok=True)
                code_path.write_text(final_state["code"])
                logger.info(f"Generated code saved to {code_path}")
            except Exception as e:
                logger.warning(f"Failed to save generated code: {e}")

    except Exception as e:
        result["error_msg"] = str(e)
        result["total_duration_s"] = round(time.time() - start_time, 3)
        logger.error(f"OptiMUS: Exception - {e}", exc_info=True)

    return result


def main():
    parser = argparse.ArgumentParser(description="OptiMUS Baseline Runner")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--prob_name", type=str, default=".", help="Problem instance path")
    parser.add_argument("--provider", type=str, default=DEFAULT_PROVIDER, help="LLM provider")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="LLM model")
    parser.add_argument("--max_selections", type=int, default=8,
                        help="Max agent selections (iterations)")
    parser.add_argument("--n_runs", type=int, default=1, help="Number of runs")
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
            max_selections=args.max_selections,
            filepath=filepath,
        )
        all_results.append(run_result)

        # Build and save ExperimentResult
        exp = ExperimentResult(
            timestamp=datetime.now().isoformat(),
            algorithm="optimus",
            dataset=args.dataset,
            prob_name=str(args.prob_name),
            max_collaborate_nums=args.max_selections,
            is_backtrack=False,
            provider=args.provider,
            model=args.model,
            orchestrator_mode="manager",
            knowledge_enabled=False,
            knowledge_mode="disable",
            knowledge_excluded_modules=[],
            agent_configs=[],
            steps=run_result.get("steps", []),
            total_duration_s=run_result.get("total_duration_s", 0.0),
            total_tokens=run_result.get("total_tokens", 0),
            num_forward_steps=len(run_result.get("steps", [])),
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
    print(f"OptiMUS Baseline Summary")
    print(f"{'='*60}")
    print(f"Problem: {args.dataset}/{args.prob_name}")
    print(f"LLM: {args.provider}/{args.model}")
    print(f"Max selections: {args.max_selections}")
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
