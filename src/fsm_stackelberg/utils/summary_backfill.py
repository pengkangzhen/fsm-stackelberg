"""Offline backfill helpers for experiment_summary.json ground-truth fields."""

from __future__ import annotations

import argparse
import fcntl
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .experiment_result import ExperimentResult


@dataclass
class BackfillStats:
    total_runs: int = 0
    matched_runs: int = 0
    updated_expected_value: int = 0
    updated_gap_percent: int = 0
    missing_optimal: int = 0
    missing_obj_value: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "total_runs": self.total_runs,
            "matched_runs": self.matched_runs,
            "updated_expected_value": self.updated_expected_value,
            "updated_gap_percent": self.updated_gap_percent,
            "missing_optimal": self.missing_optimal,
            "missing_obj_value": self.missing_obj_value,
        }


def normalize_instance_name(prob_name: str | None) -> str | None:
    if not prob_name:
        return None
    cleaned = str(prob_name).strip().strip("/")
    if cleaned.startswith("instances/"):
        cleaned = cleaned[len("instances/") :]
    return cleaned or None


def compute_gap_percent(obj_value: Any, expected_value: Any) -> float | None:
    if obj_value is None or expected_value in (None, 0):
        return None
    return round(abs(float(obj_value) - float(expected_value)) / abs(float(expected_value)) * 100, 2)


def load_optimal_objectives(instances_dir: Path) -> dict[str, float]:
    objectives: dict[str, float] = {}
    for instance_dir in sorted(path for path in instances_dir.iterdir() if path.is_dir()):
        optimal_path = instance_dir / "optimal.json"
        if not optimal_path.exists():
            continue
        optimal_data = json.loads(optimal_path.read_text(encoding="utf-8"))
        objective = optimal_data.get("objective")
        if objective is not None:
            objectives[instance_dir.name] = float(objective)
    return objectives


def backfill_summary_data(
    data: dict[str, Any],
    optimal_objectives: dict[str, float],
    dataset: str = "prob_ecr_shipper_consignee",
) -> BackfillStats:
    runs = data.get("runs", [])
    stats = BackfillStats(total_runs=len(runs))

    for run in runs:
        config = run.get("config", {})
        results = run.get("results", {})
        if config.get("dataset") != dataset:
            continue

        instance_name = normalize_instance_name(config.get("prob_name"))
        if not instance_name:
            continue

        objective = optimal_objectives.get(instance_name)
        if objective is None:
            stats.missing_optimal += 1
            continue

        stats.matched_runs += 1
        if results.get("expected_value") != objective:
            results["expected_value"] = objective
            stats.updated_expected_value += 1

        gap_percent = compute_gap_percent(results.get("obj_value"), objective)
        if gap_percent is None:
            if results.get("obj_value") is None:
                stats.missing_obj_value += 1
            if results.get("gap_percent") is not None:
                results["gap_percent"] = None
                stats.updated_gap_percent += 1
            continue

        if results.get("gap_percent") != gap_percent:
            results["gap_percent"] = gap_percent
            stats.updated_gap_percent += 1

    data.setdefault("metadata", {})
    data["metadata"]["last_updated"] = datetime.now().isoformat()
    data["metadata"]["ground_truth_backfilled_at"] = datetime.now().isoformat()
    data["statistics"] = ExperimentResult()._compute_statistics(runs)
    return stats


def write_summary_file(summary_path: Path, data: dict[str, Any]) -> None:
    with open(summary_path, "w", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            json.dump(data, handle, indent=2, ensure_ascii=False)
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def create_backup(summary_path: Path) -> Path:
    backup_path = summary_path.with_name(
        f"{summary_path.stem}_before_truth_backfill_{datetime.now().strftime('%Y%m%d_%H%M%S')}{summary_path.suffix}"
    )
    backup_path.write_text(summary_path.read_text(encoding="utf-8"), encoding="utf-8")
    return backup_path


def backfill_summary_file(
    summary_path: Path,
    instances_dir: Path,
    dataset: str = "prob_ecr_shipper_consignee",
    create_backup_file: bool = True,
) -> tuple[BackfillStats, Path | None]:
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    optimal_objectives = load_optimal_objectives(instances_dir)
    backup_path = create_backup(summary_path) if create_backup_file else None
    stats = backfill_summary_data(data, optimal_objectives, dataset=dataset)
    write_summary_file(summary_path, data)
    return stats, backup_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Backfill expected_value and gap_percent from optimal.json")
    parser.add_argument("--summary", default="results/experiment_summary.json", help="Path to experiment_summary.json")
    parser.add_argument(
        "--instances-dir",
        default="dataset/prob_ecr_shipper_consignee/instances",
        help="Directory containing instance optimal.json files",
    )
    parser.add_argument("--dataset", default="prob_ecr_shipper_consignee", help="Dataset name to backfill")
    parser.add_argument("--no-backup", action="store_true", help="Do not create a backup file before overwriting")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    stats, backup_path = backfill_summary_file(
        Path(args.summary),
        Path(args.instances_dir),
        dataset=args.dataset,
        create_backup_file=not args.no_backup,
    )
    if backup_path is not None:
        print(f"Backup created: {backup_path}")
    print(json.dumps(stats.to_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
