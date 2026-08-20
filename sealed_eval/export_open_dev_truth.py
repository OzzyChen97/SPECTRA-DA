#!/usr/bin/env python3
"""Export candidate-level truth for the declared open-development tasks.

This script is evaluator-only.  It reads sealed labels, but only for the four
Gate-1 tasks that have already been inspected and are now explicitly open
development.  The output is the candidate-truth report consumed by
``selector/objective_v2.py``.  It must not be used for the final 12 sealed
transfers.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from metrics import macro_f1  # noqa: E402
from protocol.tasks import PILOT_TASKS  # noqa: E402
from sealed_eval.access import load_target_labels  # noqa: E402
from scripts.trajectory_export.schema import (  # noqa: E402
    atomic_json,
    candidate_bank_hash,
    discover_candidate_records,
)

OPEN_DEVELOPMENT_TASKS = tuple(task.id for task in PILOT_TASKS)


def validate_open_development_tasks(tasks: list[str]) -> None:
    allowed = set(OPEN_DEVELOPMENT_TASKS)
    unknown = sorted(set(tasks) - allowed)
    if unknown:
        raise ValueError(
            "open-development truth export is restricted to Gate-1 tasks; "
            f"refusing {unknown}"
        )


def _labels_to_numpy(labels: Any) -> np.ndarray:
    if hasattr(labels, "numpy"):
        values = labels.numpy()
    else:
        values = np.asarray(labels)
    values = np.asarray(values, dtype=np.int64)
    if values.ndim != 1:
        raise ValueError("target labels must be one-dimensional")
    return values


def export_task_truth(
    *,
    candidate_root: Path,
    task: str,
    purpose: str,
    verify_hashes: bool,
) -> dict[str, Any]:
    records = discover_candidate_records(
        candidate_root,
        task,
        verify_hashes=verify_hashes,
    )
    bank_hash = candidate_bank_hash(records)
    by_id = {record["metadata"]["candidate_id"]: record for record in records}
    identifiers = sorted(by_id)
    first_target_path = by_id[identifiers[0]]["path"] / "target_public.npz"
    with np.load(first_target_path, allow_pickle=False) as first_target:
        expected_num_nodes = int(first_target["hard_predictions"].shape[0])
    labels = _labels_to_numpy(
        load_target_labels(
            task,
            purpose=purpose,
            candidate_bank_sha256=bank_hash,
            expected_num_nodes=expected_num_nodes,
        )
    )

    candidate_truth: dict[str, dict[str, float]] = {}
    target_errors: list[float] = []
    for identifier in identifiers:
        target_path = by_id[identifier]["path"] / "target_public.npz"
        with np.load(target_path, allow_pickle=False) as artifact:
            predictions = artifact["hard_predictions"].astype(np.int64, copy=False)
        if predictions.shape != labels.shape:
            raise ValueError(f"prediction shape mismatch for {task}/{identifier}")
        accuracy = float(np.mean(predictions == labels))
        target_error = 1.0 - accuracy
        target_errors.append(target_error)
        candidate_truth[identifier] = {
            "target_error": target_error,
            "target_micro_f1": accuracy,
            "target_macro_f1": macro_f1(labels, predictions),
        }

    oracle_index = int(np.argmin(np.asarray(target_errors, dtype=np.float64)))
    return {
        "task": task,
        "candidate_bank_sha256": bank_hash,
        "candidate_count": len(identifiers),
        "oracle_candidate_id": identifiers[oracle_index],
        "oracle_target_error": float(target_errors[oracle_index]),
        "candidate_truth": candidate_truth,
    }


def export_open_development_truth(args: argparse.Namespace) -> dict[str, Any]:
    if not args.trusted_evaluator:
        raise RuntimeError("refusing target-label access without --trusted-evaluator")
    tasks = args.tasks or list(OPEN_DEVELOPMENT_TASKS)
    validate_open_development_tasks(tasks)
    reports = [
        export_task_truth(
            candidate_root=args.candidate_root.resolve(),
            task=task,
            purpose=args.purpose,
            verify_hashes=not args.metadata_only_candidate_check,
        )
        for task in tasks
    ]
    if args.expected_candidates_per_task is not None:
        bad = [
            {
                "task": report["task"],
                "candidate_count": report["candidate_count"],
            }
            for report in reports
            if int(report["candidate_count"]) != args.expected_candidates_per_task
        ]
        if bad:
            raise ValueError(
                "candidate count mismatch for open-development export: "
                + json.dumps(bad, sort_keys=True)
            )
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": args.purpose,
        "scope": "open_development_gate1_candidate_truth",
        "open_development_tasks": tasks,
        "task_count": len(reports),
        "candidate_count_total": sum(int(report["candidate_count"]) for report in reports),
        "evaluator_target_label_read_count": len(reports),
        "label_access_count": 0,
        "protocol_violation_count": 0,
        "final_sealed_tasks_exposed": 0,
        "tasks": reports,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", action="append", dest="tasks")
    parser.add_argument("--purpose", default="development_open_truth")
    parser.add_argument("--expected-candidates-per-task", type=int)
    parser.add_argument(
        "--metadata-only-candidate-check",
        action="store_true",
        help="skip candidate artifact hash verification when hashes are unavailable",
    )
    parser.add_argument(
        "--trusted-evaluator",
        action="store_true",
        help="required acknowledgement that this process reads sealed target labels",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = export_open_development_truth(args)
    atomic_json(report, args.output.resolve())
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "task_count": report["task_count"],
                "candidate_count_total": report["candidate_count_total"],
                "evaluator_target_label_read_count": report["evaluator_target_label_read_count"],
                "final_sealed_tasks_exposed": report["final_sealed_tasks_exposed"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
