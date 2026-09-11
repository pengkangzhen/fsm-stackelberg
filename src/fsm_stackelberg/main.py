"""MAKO LangChain - Main entry point.

A LangChain/LangGraph implementation of the MAKO multi-agent optimization system.
"""

import argparse
import json
import logging
import os
import re
import time
import traceback
from datetime import datetime
from pathlib import Path

from fsm_stackelberg.graph import run_mako
from fsm_stackelberg.utils.utils import dataset_loader, save_workflow_result_with_history, _archive_existing_results
from fsm_stackelberg.utils.llm_config import DEFAULT_PROVIDER, DEFAULT_MODEL
from fsm_stackelberg.utils.experiment_result import ExperimentResult
from fsm_stackelberg.utils.workflow_failure import WorkflowNodeError
from fsm_stackelberg.utils.run_log import (
    build_run_manifest,
    clear_run_logger,
    get_git_commit,
    init_run_logger,
    new_run_id,
    resolve_api_base,
    save_run_manifest,
    synthesize_events_from_steps,
    utc_now_iso,
)
from fsm_stackelberg.logger import config_logger

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def _knowledge_enabled(mode: str) -> bool:
    return (mode or "").strip().lower() not in {"disable", "off", "none", "0", "false"}


def _setup_filepath(args) -> Path:
    """Generate output directory path."""
    model_tag = args.model.replace("/", "-").replace(":", "-")
    algorithm = getattr(args, 'algorithm', 'mako')
    filepath = Path("results") / algorithm / f"{args.provider}_{model_tag}" \
                / f"{args.dataset}_k{args.max_retries}" / str(args.prob_name)
    result_suffix = os.environ.get("MAKO_RESULT_SUFFIX", "").strip()
    if result_suffix:
        safe_suffix = re.sub(r"[^A-Za-z0-9._-]+", "_", result_suffix)
        filepath = filepath / safe_suffix
    filepath.mkdir(parents=True, exist_ok=True)
    return filepath


def _convert_keys_to_str(obj):
    """Recursively convert tuple keys to string keys for JSON serialization."""
    if isinstance(obj, dict):
        return {str(k) if isinstance(k, tuple) else k: _convert_keys_to_str(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_keys_to_str(item) for item in obj]
    return obj


def _build_agent_outputs(state: dict) -> dict:
    """Extract agent outputs from state for saving."""
    outputs = {}
    if state.get("data_engineer_output"):
        outputs["DataEngineer"] = state["data_engineer_output"].model_dump_json(indent=2)
    if state.get("model_expert_output"):
        outputs["ModelExpert"] = state["model_expert_output"].model_dump_json(indent=2)
    if state.get("python_code"):
        outputs["PyDeveloper"] = state["python_code"]
    if state.get("execution_result"):
        outputs["SolverExecutor"] = json.dumps(_convert_keys_to_str(state["execution_result"]), indent=2, ensure_ascii=False)
    return outputs


def _extract_usage_from_error_message(message: str) -> tuple[int, int, int]:
    """Extract token usage embedded in provider exception messages, if present."""
    match = re.search(
        r"completion_tokens=(\d+), prompt_tokens=(\d+), total_tokens=(\d+)",
        message,
    )
    if not match:
        return 0, 0, 0
    completion_tokens = int(match.group(1))
    prompt_tokens = int(match.group(2))
    total_tokens = int(match.group(3))
    return prompt_tokens, completion_tokens, total_tokens


def _extract_failed_task_name(message: str) -> str:
    """Extract LangGraph task name from an exception message when available."""
    match = re.search(r"During task with name '([^']+)'", message)
    return match.group(1) if match else ""


def _build_crash_state(exc: Exception, started_at: float) -> dict:
    """Convert an uncaught workflow exception into a failure-shaped final state."""
    error_message = str(exc)
    prompt_tokens, completion_tokens, total_tokens = _extract_usage_from_error_message(error_message)
    failed_task = _extract_failed_task_name(error_message)
    duration = time.time() - started_at

    step_metrics = []
    node_metrics = {}
    agent_metrics = {}
    if failed_task:
        step_metrics.append(
            {
                "node": failed_task,
                "step_type": "forward",
                "duration_s": round(duration, 3),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            }
        )
        node_metrics[failed_task] = {
            "total_duration_s": duration,
            "total_tokens": total_tokens,
            "num_calls": 1,
        }
        if failed_task in {"data_engineer", "model_expert", "python_developer", "solver_executor"}:
            agent_metrics[failed_task] = {
                "total_duration_s": duration,
                "total_tokens": total_tokens,
                "num_calls": 1,
            }

    return {
        "execution_result": {
            "diagnosis_required": True,
            "gurobi_status": "",
            "result": None,
        },
        "error_info": json.dumps(
            {
                "error_type": type(exc).__name__,
                "error_message": error_message,
                "stack_trace": traceback.format_exc(),
            },
            ensure_ascii=False,
        ),
        "error_category": "execution",
        "error_agent": failed_task,
        "step_metrics": step_metrics,
        "node_metrics": node_metrics,
        "agent_metrics": agent_metrics,
        "total_tokens": total_tokens,
        "total_duration_s": duration,
        "backtrack_history": [],
        "retry_count": 0,
        "current_round": 1,
        "output_history": [],
    }


def _build_experiment_result(state: dict, args, filepath: Path, expected_value=None) -> ExperimentResult:
    """Build ExperimentResult object from final state."""
    from .game.payoff import finalize_episode_payoffs

    execution_result = state.get("execution_result", {})
    is_optimal = execution_result.get("diagnosis_required") is False
    knowledge_mode = getattr(args, "knowledge", "enable")

    # Get objective value if available (always extract for reporting)
    obj_value = None
    if execution_result.get("result"):
        obj_value = execution_result["result"].get("objective_value")

    # Success: OPTIMAL + relative gap <= 1%
    is_success = is_optimal
    if is_success and obj_value is not None and expected_value is not None and expected_value != 0:
        is_success = abs(obj_value - expected_value) / abs(expected_value) <= 0.01

    # Ensure episode payoffs exist (crash paths may skip run_mako finalization)
    episode_payoff = state.get("episode_payoff")
    if not episode_payoff:
        if expected_value is not None and state.get("expected_value") is None:
            state = {**state, "expected_value": expected_value}
        if getattr(args, "true_root_cause", None) and not state.get("true_root_cause"):
            state = {**state, "true_root_cause": args.true_root_cause}
        episode_payoff = finalize_episode_payoffs(state)

    attributed_layer = state.get("attributed_layer") or episode_payoff.get("attributed_layer") or ""

    # Build step records
    steps = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    for sm in state.get("step_metrics", []):
        pt = int(sm.get("prompt_tokens", 0) or 0)
        ct = int(sm.get("completion_tokens", 0) or 0)
        total_prompt_tokens += pt
        total_completion_tokens += ct
        steps.append({
            "node": sm.get("node"),
            "step_type": sm.get("step_type"),
            "duration_s": sm.get("duration_s"),
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "total_tokens": sm.get("total_tokens", 0),
            "event_id": sm.get("event_id"),
            "artifact_refs": sm.get("artifact_refs"),
        })

    # Convert agent_metrics dict to list format expected by ExperimentResult
    agent_metrics_list = []
    for agent_name, metrics in state.get("agent_metrics", {}).items():
        agent_metrics_list.append({
            "agent_name": agent_name,
            "total_duration_s": metrics.get("total_duration_s", 0),
            "total_tokens": metrics.get("total_tokens", 0),
            "num_calls": metrics.get("num_calls", 0),
        })

    node_metrics_list = []
    for node_name, metrics in state.get("node_metrics", {}).items():
        node_metrics_list.append({
            "node_name": node_name,
            "total_duration_s": metrics.get("total_duration_s", 0),
            "total_tokens": metrics.get("total_tokens", 0),
            "num_calls": metrics.get("num_calls", 0),
        })

    return ExperimentResult(
        timestamp=datetime.now().isoformat(),
        algorithm="mako",
        dataset=args.dataset,
        prob_name=str(args.prob_name),
        max_collaborate_nums=args.max_retries,
        is_backtrack=True,
        provider=args.provider,
        model=args.model,
        orchestrator_mode=args.diagnosis_mode,
        knowledge_enabled=_knowledge_enabled(knowledge_mode),
        knowledge_mode=knowledge_mode,
        knowledge_excluded_modules=[],
        agent_configs=[],
        steps=steps,
        total_duration_s=state.get("total_duration_s", 0.0),
        total_prompt_tokens=total_prompt_tokens,
        total_completion_tokens=total_completion_tokens,
        total_tokens=state.get("total_tokens", 0),
        num_forward_steps=len([s for s in steps if s.get("step_type") == "forward"]),
        num_backward_steps=len([s for s in steps if s.get("step_type") == "backward"]),
        status=is_success,
        error_msg=state.get("error_info") if not is_success else "",
        obj_value=obj_value,
        expected_value=expected_value,
        is_model_valid=None,
        gurobi_status=execution_result.get("gurobi_status", ""),
        gurobi_structure=execution_result.get("gurobi_structure", {}),
        error_agent=state.get("error_agent") if not is_success else "",
        error_details=state.get("error_info") if not is_success else {},
        error_category=state.get("error_category") if not is_success else "none",
        constraints=[],
        agent_metrics=agent_metrics_list,
        node_metrics=node_metrics_list,
        backtrack_history=state.get("backtrack_history", []),
        episode_payoff=episode_payoff or {},
        attributed_layer=attributed_layer or "",
        true_root_cause=state.get("true_root_cause")
        or getattr(args, "true_root_cause", None)
        or "",
        probe_order=getattr(args, "probe_order", "") or state.get("probe_order") or "",
        probe_seed=getattr(args, "probe_seed", None)
        if getattr(args, "probe_seed", None) is not None
        else state.get("probe_seed"),
        inject_id=state.get("fault_plant_id")
        or state.get("inject_id")
        or getattr(args, "inject_id", None)
        or "",
        temperature=float(getattr(args, "temperature", 0.0) or state.get("temperature") or 0.0),
        result_path=str(filepath),
    )


def main():
    parser = argparse.ArgumentParser(description="MAKO LangChain - Multi-Agent Optimization")
    parser.add_argument("--algorithm", type=str, default="mako",
                        choices=["mako", "spm", "cot"],
                        help="Algorithm to use (mako, spm, cot)")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--prob_name", type=str, default=".", help="Problem name (default: . for default case)")
    parser.add_argument("--provider", type=str, default=DEFAULT_PROVIDER, help="LLM provider")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="LLM model to use")
    parser.add_argument("--max_retries", type=int, default=3, help="Maximum backtrack retries")
    parser.add_argument("--diagnosis_mode", type=str, default="stackelberg",
                        choices=["stackelberg", "adversarial", "sequential"],
                        help="Diagnosis mode: stackelberg (inspection game, default), "
                             "adversarial (single-judge baseline), or sequential (reverse-order baseline)")
    parser.add_argument("--probe_order", type=str, default="causal",
                        choices=["causal", "reverse", "random"],
                        help="Stackelberg commitment-order ablation: causal (default), reverse, or random")
    parser.add_argument("--true_root_cause", type=str, default=None,
                        choices=["data_engineer", "model_expert", "python_developer"],
                        help="Ground-truth failing layer for follower attribution payoff u_F")
    parser.add_argument(
        "--inject",
        type=str,
        default=None,
        dest="inject_id",
        help="Fault-plant id (Exp-I). Applied once on ME→PD; seeds true_root_cause if unset. "
             "Known: me_drop_stage2_balance, me_force_zero_sea",
    )
    parser.add_argument(
        "--probe_seed",
        type=int,
        default=None,
        help="RNG seed for --probe_order random (reproducible shuffle; ignored otherwise)",
    )
    parser.add_argument(
        "--omega_source",
        type=str,
        default="evidence_rank",
        choices=["evidence_rank", "status_prior"],
        help="How Stackelberg commits ω: evidence_rank (default) or legacy status_prior",
    )
    parser.add_argument(
        "--rank_method",
        type=str,
        default="llm_rank",
        choices=["llm_rank", "heuristic", "hybrid"],
        help="Layer ranking method for evidence_rank (default: llm_rank)",
    )
    parser.add_argument("--knowledge", type=str, default="progressive",
                        choices=["progressive", "enable", "disable"],
                        help="Knowledge injection: progressive (catalog→request→load; default), "
                             "enable (alias of progressive), or disable")
    parser.add_argument(
        "--log_prompts",
        action="store_true",
        help="Dump full LLM prompts under results/.../prompts/ (off by default)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="LLM temperature (default 0; recorded in run_manifest)",
    )
    parser.add_argument(
        "--snapshot_dir",
        type=str,
        default=None,
        help="Freeze the failure blackboard to this directory at the "
             "solver→diagnosis boundary instead of diagnosing (freeze run; "
             "branch arms later via --resume_from)",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Resume a frozen failure snapshot: only the diagnosis–repair "
             "loop runs live; diagnosis-side flags act as resume overrides "
             "(provider/model/temperature default to the CLI defaults, not "
             "the snapshot's — pass them explicitly to pin)",
    )

    args = parser.parse_args()

    if getattr(args, "inject_id", None):
        from fsm_stackelberg.injection import get_plant
        plant = get_plant(args.inject_id)
        if not args.true_root_cause:
            args.true_root_cause = plant.true_root_cause

    # Setup output directory
    filepath = _setup_filepath(args)
    logger.info(f"Results will be saved to: {filepath}")

    # Archive existing results from previous runs
    _archive_existing_results(str(filepath))

    # Configure logging to file
    config_logger(filepath)

    run_id = new_run_id()
    run_started_at = utc_now_iso()
    run_logger = init_run_logger(
        filepath,
        run_id=run_id,
        provider=args.provider,
        model=args.model,
        temperature=float(args.temperature),
        log_prompts=bool(args.log_prompts),
        api_base=resolve_api_base(args.provider),
    )
    logger.info("Run logging: run_id=%s events=%s", run_id, run_logger.events_path)

    # Load problem
    logger.info(f"Loading problem from dataset: {args.dataset}, prob_name: {args.prob_name}")
    problem = dataset_loader(args.dataset, args.prob_name)

    # Fail fast on an unusable snapshot before any spend (e.g. the freeze run
    # crashed in forward and never wrote a blackboard).
    if getattr(args, "resume_from", None):
        from fsm_stackelberg.graph.snapshot import SNAPSHOT_STATE_FILE
        if not (Path(args.resume_from) / SNAPSHOT_STATE_FILE).exists():
            logger.error(
                "Resume aborted: %s/%s not found — the freeze run may have "
                "failed before diagnosis (check its workflow.log).",
                args.resume_from,
                SNAPSHOT_STATE_FILE,
            )
            return None

    # Run algorithm based on selection
    if args.algorithm == "spm":
        logger.info(f"Running SPM algorithm (provider={args.provider}, model={args.model})...")
        from fsm_stackelberg.experiments.spm import run_spm
        result = run_spm(problem, provider=args.provider, model=args.model)

        # Log results
        if result.get("status"):
            logger.info(f"✓ SPM successful! Objective = {result.get('obj_value')}")
        else:
            logger.warning(f"✗ SPM failed: {result.get('error_msg')}")

        logger.info(f"Total tokens: {result.get('total_tokens', 0)}")
        logger.info(f"Total duration: {result.get('total_duration_s', 0):.2f}s")

        # Save result
        import json
        result_file = filepath / "spm_result.json"
        with open(result_file, "w") as f:
            json.dump(result, f, indent=2, default=str)

        # Append to summary
        from fsm_stackelberg.utils.experiment_result import ExperimentResult
        exp = ExperimentResult(
            timestamp=datetime.now().isoformat(),
            algorithm="spm",
            dataset=args.dataset,
            prob_name=str(args.prob_name),
            max_collaborate_nums=0,
            is_backtrack=False,
            provider=args.provider,
            model=args.model,
            orchestrator_mode="",
            knowledge_enabled=False,
            knowledge_mode="disable",
            knowledge_excluded_modules=[],
            agent_configs=[],
            steps=result.get("steps", []),
            total_duration_s=result.get("total_duration_s", 0.0),
            total_tokens=result.get("total_tokens", 0),
            num_forward_steps=1,
            num_backward_steps=0,
            status=bool(result.get("status")),
            error_msg=result.get("error_msg") or "",
            obj_value=result.get("obj_value"),
            expected_value=problem.get("expected_value"),
            is_model_valid=result.get("is_model_valid"),
            gurobi_status=result.get("gurobi_status", ""),
            error_agent="",
            error_details={"error_message": result.get("error_msg")} if result.get("error_msg") else {},
            error_category="none",
            constraints=[],
            agent_metrics=[],
            node_metrics=[],
            backtrack_history=[],
            result_path=str(filepath),
        )
        exp.save_json(str(filepath))
        exp.append_summary_json("results/experiment_summary.json")
        synthesize_events_from_steps(
            filepath,
            exp.steps,
            run_id=run_id,
            provider=args.provider,
            model=args.model,
        )
        save_run_manifest(
            filepath,
            build_run_manifest(
                exp=exp,
                state={},
                run_id=run_id,
                started_at=run_started_at,
                ended_at=utc_now_iso(),
                temperature=float(args.temperature),
                api_base=resolve_api_base(args.provider),
                git_commit=get_git_commit(),
            ),
        )
        clear_run_logger()

        return result

    elif args.algorithm == "cot":
        logger.info(f"Running CoT algorithm (provider={args.provider}, model={args.model})...")
        from fsm_stackelberg.experiments.cot import run_cot
        result = run_cot(problem, provider=args.provider, model=args.model)

        # Log results
        if result.get("status"):
            logger.info(f"✓ CoT successful! Objective = {result.get('obj_value')}")
        else:
            logger.warning(f"✗ CoT failed: {result.get('error_msg')}")

        logger.info(f"Total tokens: {result.get('total_tokens', 0)}")
        logger.info(f"Total duration: {result.get('total_duration_s', 0):.2f}s")

        # Save result
        import json
        result_file = filepath / "cot_result.json"
        with open(result_file, "w") as f:
            json.dump(result, f, indent=2, default=str)

        # Append to summary
        from fsm_stackelberg.utils.experiment_result import ExperimentResult
        exp = ExperimentResult(
            timestamp=datetime.now().isoformat(),
            algorithm="cot",
            dataset=args.dataset,
            prob_name=str(args.prob_name),
            max_collaborate_nums=0,
            is_backtrack=False,
            provider=args.provider,
            model=args.model,
            orchestrator_mode="",
            knowledge_enabled=False,
            knowledge_mode="disable",
            knowledge_excluded_modules=[],
            agent_configs=[],
            steps=result.get("steps", []),
            total_duration_s=result.get("total_duration_s", 0.0),
            total_tokens=result.get("total_tokens", 0),
            num_forward_steps=3,
            num_backward_steps=0,
            status=bool(result.get("status")),
            error_msg=result.get("error_msg") or "",
            obj_value=result.get("obj_value"),
            expected_value=problem.get("expected_value"),
            is_model_valid=result.get("is_model_valid"),
            gurobi_status=result.get("gurobi_status", ""),
            error_agent="",
            error_details={"error_message": result.get("error_msg")} if result.get("error_msg") else {},
            error_category="none",
            constraints=[],
            agent_metrics=[],
            node_metrics=[],
            backtrack_history=[],
            result_path=str(filepath),
        )
        exp.save_json(str(filepath))
        exp.append_summary_json("results/experiment_summary.json")
        synthesize_events_from_steps(
            filepath,
            exp.steps,
            run_id=run_id,
            provider=args.provider,
            model=args.model,
        )
        save_run_manifest(
            filepath,
            build_run_manifest(
                exp=exp,
                state={},
                run_id=run_id,
                started_at=run_started_at,
                ended_at=utc_now_iso(),
                temperature=float(args.temperature),
                api_base=resolve_api_base(args.provider),
                git_commit=get_git_commit(),
            ),
        )
        clear_run_logger()

        return result

    else:  # mako
        logger.info(f"Starting FSM-Stackelberg workflow (provider={args.provider}, model={args.model}, "
                    f"diagnosis_mode={args.diagnosis_mode}, probe_order={args.probe_order}, "
                    f"inject={getattr(args, 'inject_id', None)})...")
        workflow_started_at = time.time()
        try:
            final_state = run_mako(
                problem_description=problem["description"],
                sample=problem["sample"],
                schema=problem.get("schema"),
                provider=args.provider,
                model=args.model,
                max_retries=args.max_retries,
                diagnosis_mode=args.diagnosis_mode,
                probe_order=args.probe_order,
                expected_value=problem.get("expected_value"),
                knowledge_mode=args.knowledge,
                true_root_cause=args.true_root_cause,
                result_dir=str(filepath),
                run_id=run_id,
                log_prompts=bool(args.log_prompts),
                temperature=float(args.temperature),
                probe_seed=args.probe_seed,
                inject_id=args.inject_id,
                omega_source=getattr(args, "omega_source", "evidence_rank"),
                rank_method=getattr(args, "rank_method", "llm_rank"),
                snapshot_dir=getattr(args, "snapshot_dir", None),
                resume_from=getattr(args, "resume_from", None),
            )
        except Exception as exc:
            logger.exception("MAKO workflow crashed; recording failure result.")
            if isinstance(exc, WorkflowNodeError):
                final_state = exc.state_snapshot
            else:
                final_state = _build_crash_state(exc, workflow_started_at)

        # Print results
        execution_result = final_state.get("execution_result", {})
        if execution_result.get("diagnosis_required") is False:
            logger.info("✓ Optimization successful!")
            result = execution_result.get("result", {})
            logger.info(f"  Status: {result.get('status')}")
            logger.info(f"  Objective value: {result.get('objective_value')}")
        else:
            logger.warning("✗ Optimization failed after all retries")
            error_info = final_state.get("error_info", "Unknown error")
            logger.warning(f"  Error: {error_info}")

        # Print metrics
        logger.info(f"Total tokens: {final_state.get('total_tokens', 0)}")
        logger.info(f"Total duration: {final_state.get('total_duration_s', 0):.2f}s")
        episode_payoff = final_state.get("episode_payoff") or {}
        if episode_payoff:
            logger.info(
                "Episode payoff: u_L=%s u_F=%s attributed=%s",
                episode_payoff.get("u_L"),
                episode_payoff.get("u_F"),
                episode_payoff.get("attributed_layer"),
            )

        # Print backtrack history
        backtrack_history = final_state.get("backtrack_history", [])
        if backtrack_history:
            logger.info(f"Diagnosis history: {len(backtrack_history)} attempts")
            for event in backtrack_history:
                logger.info(f"  - Retry {event['retry_count']}: mode={event.get('diagnosis_mode')}, "
                           f"agent={event.get('error_agent', 'unknown')}")

        # Save results with history
        logger.info("Saving results...")
        save_workflow_result_with_history(final_state, str(filepath))

        exp = _build_experiment_result(
            final_state,
            args,
            filepath,
            expected_value=problem.get("expected_value"),
        )
        exp.save_json(str(filepath))
        exp.append_summary_json("results/experiment_summary.json")

        synthesize_events_from_steps(
            filepath,
            exp.steps,
            run_id=run_id,
            provider=args.provider,
            model=args.model,
        )
        snapshot_forward_tokens = None
        if getattr(args, "resume_from", None):
            from fsm_stackelberg.graph.snapshot import read_snapshot_manifest
            snap_manifest = read_snapshot_manifest(args.resume_from)
            snapshot_forward_tokens = (
                snap_manifest.get("forward_totals") or {}
            ).get("total_tokens")
        manifest = build_run_manifest(
            exp=exp,
            state=final_state,
            run_id=run_id,
            started_at=run_started_at,
            ended_at=utc_now_iso(),
            temperature=float(args.temperature),
            api_base=resolve_api_base(args.provider),
            probe_seed=args.probe_seed,
            inject_id=args.inject_id,
            knowledge_max_rounds=final_state.get("knowledge_max_rounds"),
            git_commit=get_git_commit(),
            resumed_from=getattr(args, "resume_from", None),
            snapshot_dir=getattr(args, "snapshot_dir", None),
            snapshot_forward_tokens=snapshot_forward_tokens,
        )
        save_run_manifest(filepath, manifest)
        clear_run_logger()

        logger.info(f"Results saved to: {filepath}")
        logger.info("Structured logs: run_manifest.json, events.jsonl, artifacts/")

        return final_state


if __name__ == "__main__":
    main()
