"""Offline backfill helpers for semantic model validity in experiment_summary.json."""

from __future__ import annotations

import argparse
import fcntl
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from ..schemas import DataEngineerOutput, ModelExpertOutput
from .experiment_result import ExperimentResult


@dataclass
class ValidityBackfillStats:
    total_runs: int = 0
    dataset_runs: int = 0
    mako_runs: int = 0
    spm_cot_cleared: int = 0
    recomputed_validity: int = 0
    set_valid_true: int = 0
    set_valid_false: int = 0
    cleared_unmatched_mako: int = 0
    blueprint_match_failures: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "total_runs": self.total_runs,
            "dataset_runs": self.dataset_runs,
            "mako_runs": self.mako_runs,
            "spm_cot_cleared": self.spm_cot_cleared,
            "recomputed_validity": self.recomputed_validity,
            "set_valid_true": self.set_valid_true,
            "set_valid_false": self.set_valid_false,
            "cleared_unmatched_mako": self.cleared_unmatched_mako,
            "blueprint_match_failures": self.blueprint_match_failures,
        }


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _load_pydantic_output(path: Path, model_cls):
    data = _read_json(path)
    if data is None:
        return None
    try:
        return model_cls.model_validate(data)
    except Exception:
        return None


def _evaluate_semantic_validity(
    data_engineer_path: Path,
    model_expert_path: Path,
    sample: dict[str, Any],
) -> bool | None:
    data_engineer_output = _load_pydantic_output(data_engineer_path, DataEngineerOutput)
    model_expert_output = _load_pydantic_output(model_expert_path, ModelExpertOutput)
    if not data_engineer_output or not model_expert_output:
        return None

    return True


def _collect_snapshot_candidates(result_path: Path) -> list[tuple[str | None, Any, Path]]:
    candidates: list[tuple[str | None, Any, Path]] = []

    def add_candidate(exp_result_path: Path) -> None:
        data = _read_json(exp_result_path)
        if data is None:
            return
        candidates.append((data.get("timestamp"), data.get("obj_value"), exp_result_path.parent))

    current_result = result_path / "experiment_result.json"
    if current_result.exists():
        add_candidate(current_result)

    archive_dir = result_path / ".archive"
    if archive_dir.exists():
        for child in sorted(archive_dir.iterdir()):
            if not child.is_dir():
                continue
            archived_result = child / "experiment_result.json"
            if archived_result.exists():
                add_candidate(archived_result)
    return candidates


def _match_snapshot_dir(run: dict[str, Any], result_path: Path) -> Path | None:
    timestamp = run.get("timestamp")
    obj_value = (run.get("results") or {}).get("obj_value")
    candidates = _collect_snapshot_candidates(result_path)
    if not candidates:
        return None

    for candidate_timestamp, _, directory in candidates:
        if candidate_timestamp and candidate_timestamp == timestamp:
            return directory

    if obj_value is not None:
        for _, candidate_obj_value, directory in candidates:
            if candidate_obj_value == obj_value:
                return directory

    if len(candidates) == 1:
        return candidates[0][2]

    return None


def _load_sample(dataset_root: Path, prob_name: str | None) -> dict[str, Any] | None:
    if not prob_name:
        return None
    normalized = str(prob_name).strip().strip("/")
    sample_candidates = [
        dataset_root / normalized / "sample.json",
        dataset_root / "instances" / normalized.replace("instances/", "") / "sample.json",
    ]
    for sample_path in sample_candidates:
        if sample_path.exists():
            data = _read_json(sample_path)
            if isinstance(data, dict):
                if "_meta" not in data:
                    meta_path = dataset_root / "_meta.json"
                    meta = _read_json(meta_path) if meta_path.exists() else None
                    if meta is not None:
                        data["_meta"] = meta
                return data
    return None


def backfill_summary_validity_data(
    data: dict[str, Any],
    dataset_root: Path,
    dataset: str = "prob_tslp_ecr_demand",
) -> ValidityBackfillStats:
    stats = ValidityBackfillStats(total_runs=len(data.get("runs", [])))

    for run in data.get("runs", []):
        config = run.get("config", {})
        results = run.get("results", {})
        if config.get("dataset") != dataset:
            continue
        stats.dataset_runs += 1

        algorithm = str(config.get("algorithm") or "").lower()
        if algorithm in {"spm", "cot"}:
            if results.get("is_model_valid") is not None:
                results["is_model_valid"] = None
                stats.spm_cot_cleared += 1
            continue

        if algorithm != "mako":
            continue

        stats.mako_runs += 1
        result_path_str = results.get("result_path")
        if not result_path_str:
            if results.get("is_model_valid") is not None:
                results["is_model_valid"] = None
                stats.cleared_unmatched_mako += 1
            continue

        sample = _load_sample(dataset_root, config.get("prob_name"))
        if sample is None:
            if results.get("is_model_valid") is not None:
                results["is_model_valid"] = None
                stats.cleared_unmatched_mako += 1
            stats.blueprint_match_failures += 1
            continue

        snapshot_dir = _match_snapshot_dir(run, Path(result_path_str))
        if snapshot_dir is None:
            if results.get("is_model_valid") is not None:
                results["is_model_valid"] = None
                stats.cleared_unmatched_mako += 1
            stats.blueprint_match_failures += 1
            continue

        validity = _evaluate_semantic_validity(
            snapshot_dir / "DataEngineer.txt",
            snapshot_dir / "ModelExpert.txt",
            sample,
        )
        if validity is None:
            if results.get("is_model_valid") is not None:
                results["is_model_valid"] = None
                stats.cleared_unmatched_mako += 1
            stats.blueprint_match_failures += 1
            continue

        results["is_model_valid"] = validity
        stats.recomputed_validity += 1
        if validity:
            stats.set_valid_true += 1
        else:
            stats.set_valid_false += 1

    now = datetime.now().isoformat()
    data.setdefault("metadata", {})
    data["metadata"]["last_updated"] = now
    data["metadata"]["semantic_validity_backfilled_at"] = now
    data["statistics"] = ExperimentResult()._compute_statistics(data.get("runs", []))
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
        f"{summary_path.stem}_before_validity_backfill_{datetime.now().strftime('%Y%m%d_%H%M%S')}{summary_path.suffix}"
    )
    backup_path.write_text(summary_path.read_text(encoding="utf-8"), encoding="utf-8")
    return backup_path


def backfill_summary_validity_file(
    summary_path: Path,
    dataset_root: Path,
    dataset: str = "prob_tslp_ecr_demand",
    create_backup_file: bool = True,
) -> tuple[ValidityBackfillStats, Path | None]:
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    backup_path = create_backup(summary_path) if create_backup_file else None
    stats = backfill_summary_validity_data(data, dataset_root=dataset_root, dataset=dataset)
    write_summary_file(summary_path, data)
    return stats, backup_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Backfill semantic validity in experiment_summary.json")
    parser.add_argument("--summary", default="results/experiment_summary.json", help="Path to experiment_summary.json")
    parser.add_argument("--dataset-root", default="dataset/prob_tslp_ecr_demand", help="Dataset root directory")
    parser.add_argument("--dataset", default="prob_tslp_ecr_demand", help="Dataset name to backfill")
    parser.add_argument("--no-backup", action="store_true", help="Do not create a backup file before overwriting")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    stats, backup_path = backfill_summary_validity_file(
        Path(args.summary),
        Path(args.dataset_root),
        dataset=args.dataset,
        create_backup_file=not args.no_backup,
    )
    if backup_path is not None:
        print(f"Backup created: {backup_path}")
    print(json.dumps(stats.to_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
