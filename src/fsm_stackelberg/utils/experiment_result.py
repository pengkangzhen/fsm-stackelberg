"""
ExperimentResult — structured experiment metadata for result tracking.

Captures algorithm config, per-node step metrics (tokens, timing),
and final solve status in a single JSON-serializable object.
"""

import json
import os
import fcntl
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional


@dataclass
class ExperimentResult:
    # --- Metadata ---
    timestamp: str = ""
    algorithm: str = ""
    dataset: str = ""
    prob_name: str = ""
    max_collaborate_nums: int = 0
    is_backtrack: bool = False

    # --- Config info ---
    provider: str = ""
    model: str = ""
    orchestrator_mode: str = ""
    knowledge_enabled: bool = False
    knowledge_mode: str = "enable"
    knowledge_excluded_modules: List[str] = field(default_factory=list)

    # --- Agent configuration snapshot ---
    agent_configs: List[Dict] = field(default_factory=list)

    # --- Per-step metrics ---
    steps: List[Dict] = field(default_factory=list)

    # --- Aggregated metrics ---
    total_duration_s: float = 0.0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    num_forward_steps: int = 0
    num_backward_steps: int = 0

    # --- Result ---
    status: bool = False
    error_msg: str = ""
    obj_value: Optional[float] = None
    expected_value: Optional[float] = None

    # --- Error details ---
    error_agent: str = ""
    result_path: str = ""
    error_details: Dict = field(default_factory=dict)

    # --- Solver validation ---
    is_model_valid: Optional[bool] = None
    gurobi_status: str = ""

    # --- Error classification ---
    error_category: str = "none"

    # --- Constraints ---
    constraints: List[Dict] = field(default_factory=list)

    # --- Agent level metrics ---
    agent_metrics: List[Dict] = field(default_factory=list)

    # --- Node level metrics ---
    node_metrics: List[Dict] = field(default_factory=list)

    # --- Command used to run this experiment ---
    command: str = ""

    # --- Backtrack history ---
    backtrack_history: List[Dict] = field(default_factory=list)

    # --- Stackelberg episode payoffs (analysis-mode) ---
    episode_payoff: Dict = field(default_factory=dict)
    attributed_layer: str = ""
    true_root_cause: str = ""
    probe_order: str = ""
    probe_seed: Optional[int] = None
    inject_id: str = ""
    temperature: float = 0.0

    # --- Solver structural fingerprint (§5.5 formulation stability) ---
    gurobi_structure: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def save_json(self, filepath: str):
        """Save full experiment result as JSON."""
        path = os.path.join(filepath, "experiment_result.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    def _build_command(self) -> str:
        """Generate the reproduction command for this run."""
        cmd_parts = [
            "poetry run python -m fsm_stackelberg.main",
            f"--dataset {self.dataset}",
            f"--prob_name {self.prob_name}",
            f"--max_retries {self.max_collaborate_nums}",
            f"--diagnosis_mode {self.orchestrator_mode}",
            f"--provider {self.provider}",
            f"--model {self.model}",
            f"--knowledge {self.knowledge_mode}",
        ]
        if self.probe_order:
            cmd_parts.append(f"--probe_order {self.probe_order}")
        if self.true_root_cause:
            cmd_parts.append(f"--true_root_cause {self.true_root_cause}")
        if self.inject_id:
            cmd_parts.append(f"--inject {self.inject_id}")
        if self.probe_seed is not None:
            cmd_parts.append(f"--probe_seed {self.probe_seed}")
        return " \\\n  ".join(cmd_parts)

    def _build_run_record(self) -> Dict:
        """Build a single run record for the summary file."""
        import uuid
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6]

        gap_percent = None
        if self.obj_value is not None and self.expected_value is not None and self.expected_value != 0:
            gap_percent = round(abs(self.obj_value - self.expected_value) / abs(self.expected_value) * 100, 2)

        error_summary = None
        if not self.status:
            error_summary = {
                "error_agent": self.error_agent,
                "error_msg": self.error_msg,
                "error_category": self.error_category,
            }

        command = self.command if self.command else self._build_command()

        return {
            "run_id": run_id,
            "timestamp": self.timestamp,
            "config": {
                "algorithm": self.algorithm,
                "provider": self.provider,
                "model": self.model,
                "dataset": self.dataset,
                "prob_name": self.prob_name,
                "max_collaborate_nums": self.max_collaborate_nums,
                "is_backtrack": self.is_backtrack,
                "orchestrator_mode": self.orchestrator_mode,
                "knowledge_enabled": self.knowledge_enabled,
                "knowledge_mode": self.knowledge_mode,
                "knowledge_excluded_modules": self.knowledge_excluded_modules,
                "probe_order": self.probe_order,
                "probe_seed": self.probe_seed,
                "true_root_cause": self.true_root_cause or None,
                "inject_id": self.inject_id or None,
                "temperature": self.temperature,
            },
            "command": command,
            "results": {
                "status": self.status,
                "duration_s": round(self.total_duration_s, 3),
                "obj_value": self.obj_value,
                "expected_value": self.expected_value,
                "gap_percent": gap_percent,
                "total_tokens": self.total_tokens,
                "prompt_tokens": self.total_prompt_tokens,
                "completion_tokens": self.total_completion_tokens,
                "num_forward_steps": self.num_forward_steps,
                "num_backward_steps": self.num_backward_steps,
                "total_steps": self.num_forward_steps + self.num_backward_steps,
                "is_model_valid": self.is_model_valid,
                "gurobi_status": self.gurobi_status,
                "gurobi_structure": self.gurobi_structure,
                "result_path": self.result_path,
                "attributed_layer": self.attributed_layer or None,
                "episode_payoff": self.episode_payoff or None,
            },
            "error_summary": error_summary,
            "backtrack_history": self.backtrack_history,
            "agent_metrics": [
                {
                    "agent_name": m.get("agent_name", ""),
                    "total_duration_s": round(m.get("total_duration_s", 0), 3),
                    "total_tokens": m.get("total_tokens", 0),
                    "num_calls": m.get("num_calls", 0),
                }
                for m in self.agent_metrics
            ],
            "node_metrics": [
                {
                    "node_name": m.get("node_name", ""),
                    "total_duration_s": round(m.get("total_duration_s", 0), 3),
                    "total_tokens": m.get("total_tokens", 0),
                    "num_calls": m.get("num_calls", 0),
                }
                for m in self.node_metrics
            ],
        }

    def _compute_statistics(self, runs: List[Dict]) -> Dict:
        """Compute aggregate statistics across runs."""
        from collections import defaultdict

        if not runs:
            return {}

        total_runs = len(runs)
        success_count = sum(1 for r in runs if r.get("results", {}).get("status", False))
        failure_count = total_runs - success_count
        success_rate = round(success_count / total_runs, 4) if total_runs > 0 else 0

        durations = [r.get("results", {}).get("duration_s", 0) for r in runs]
        tokens = [r.get("results", {}).get("total_tokens", 0) for r in runs]

        by_algorithm = defaultdict(lambda: {"runs": 0, "success": 0, "total_duration": 0})
        by_model = defaultdict(lambda: {"runs": 0, "success": 0})
        by_dataset = defaultdict(lambda: {"runs": 0, "success": 0})

        for r in runs:
            cfg = r.get("config", {})
            res = r.get("results", {})
            algo = cfg.get("algorithm", "unknown")
            model = cfg.get("model", "unknown")
            dataset = cfg.get("dataset", "unknown")

            by_algorithm[algo]["runs"] += 1
            by_algorithm[algo]["success"] += 1 if res.get("status") else 0
            by_algorithm[algo]["total_duration"] += res.get("duration_s", 0)

            by_model[model]["runs"] += 1
            by_model[model]["success"] += 1 if res.get("status") else 0

            by_dataset[dataset]["runs"] += 1
            by_dataset[dataset]["success"] += 1 if res.get("status") else 0

        for algo_data in by_algorithm.values():
            algo_data["success_rate"] = round(algo_data["success"] / algo_data["runs"], 4) if algo_data["runs"] > 0 else 0
            algo_data["avg_duration"] = round(algo_data["total_duration"] / algo_data["runs"], 3) if algo_data["runs"] > 0 else 0

        for model_data in by_model.values():
            model_data["success_rate"] = round(model_data["success"] / model_data["runs"], 4) if model_data["runs"] > 0 else 0

        for dataset_data in by_dataset.values():
            dataset_data["success_rate"] = round(dataset_data["success"] / dataset_data["runs"], 4) if dataset_data["runs"] > 0 else 0

        return {
            "total_runs": total_runs,
            "success_count": success_count,
            "failure_count": failure_count,
            "success_rate": success_rate,
            "avg_duration_s": round(sum(durations) / total_runs, 3) if total_runs > 0 else 0,
            "total_tokens": sum(tokens),
            "avg_tokens_per_run": round(sum(tokens) / total_runs) if total_runs > 0 else 0,
            "by_algorithm": dict(by_algorithm),
            "by_model": dict(by_model),
            "by_dataset": dict(by_dataset),
        }

    def append_summary_json(self, json_path: str):
        """Append run summary to central JSON file with a process-safe read-modify-write."""
        os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
        if not os.path.exists(json_path):
            with open(json_path, "w", encoding="utf-8") as f:
                f.write("")

        with open(json_path, "r+", encoding="utf-8") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                f.seek(0)
                raw = f.read().strip()
                if raw:
                    try:
                        data = json.loads(raw)
                    except json.JSONDecodeError:
                        data = {"metadata": {}, "runs": [], "statistics": {}}
                else:
                    data = {"metadata": {}, "runs": [], "statistics": {}}

                new_run = self._build_run_record()

                now = datetime.now().isoformat()
                if not data.get("metadata", {}).get("created_at"):
                    data.setdefault("metadata", {})["created_at"] = now
                data["metadata"]["last_updated"] = now
                data.setdefault("runs", []).append(new_run)
                data["metadata"]["total_runs"] = len(data["runs"])
                data["statistics"] = self._compute_statistics(data["runs"])

                f.seek(0)
                f.truncate()
                json.dump(data, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
