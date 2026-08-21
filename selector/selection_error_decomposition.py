#!/usr/bin/env python3
"""Decompose open-development selector regret by method, trajectory, checkpoint.

The tool reads only selector JSON files and the already exported open-dev
candidate-truth report.  It rejects sealed/final-label paths and never imports
the sealed evaluator.  For a selected candidate h in trajectory g and method
m, the exact decompositions are

    R(h) - R(h*) = [min_{j in g} R(j) - R(h*)]
                   + [R(h) - min_{j in g} R(j)]

and the analogous method-level decomposition.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from selector.objective_v2 import (
    OPEN_DEVELOPMENT_TASKS,
    discover_selection_paths,
    load_open_dev_truth,
    load_selection,
)

FORBIDDEN_PATH_PARTS = {".sealed", "sealed_eval", "final_12_labels"}


def reject_forbidden_path(path: Path) -> None:
    forbidden = sorted(set(path.resolve().parts) & FORBIDDEN_PATH_PARTS)
    if forbidden:
        raise RuntimeError(f"forbidden path components: {forbidden}")


def atomic_json(document: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(document, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def parse_candidate_id(candidate_id: str) -> dict[str, str | int]:
    parts = str(candidate_id).split("__")
    if len(parts) < 4:
        raise ValueError(f"candidate id has no method/config fields: {candidate_id}")
    seed = next((part.removeprefix("seed-") for part in parts if part.startswith("seed-")), None)
    epoch_text = next(
        (part.removeprefix("epoch-") for part in parts if part.startswith("epoch-")),
        None,
    )
    if seed is None or epoch_text is None:
        raise ValueError(f"candidate id has no seed/epoch fields: {candidate_id}")
    return {
        "method": parts[1],
        "config": parts[2],
        "seed": seed,
        "epoch": int(epoch_text),
    }


def trajectory_key(metadata: dict[str, str | int]) -> tuple[str, str, str]:
    return (
        str(metadata["method"]),
        str(metadata["config"]),
        str(metadata["seed"]),
    )


def decompose_selection(
    selection: dict[str, Any],
    truth: dict[str, Any],
) -> dict[str, Any]:
    task = str(selection["task"])
    if task != truth["task"]:
        raise ValueError("selection task does not match truth task")
    risks = truth["risks"]
    if set(selection["candidate_scores"]) != set(risks):
        raise ValueError(f"candidate coverage mismatch for {task}/{selection['selector']}")
    selected = str(selection["selected_candidate_id"])
    if selected not in risks:
        raise ValueError(f"selected candidate is absent from truth: {selected}")

    metadata = {candidate: parse_candidate_id(candidate) for candidate in risks}
    selected_metadata = metadata[selected]
    selected_trajectory = trajectory_key(selected_metadata)
    selected_method = str(selected_metadata["method"])
    oracle_candidate = min(risks, key=lambda candidate: (float(risks[candidate]), candidate))
    trajectory_candidates = [
        candidate
        for candidate, candidate_metadata in metadata.items()
        if trajectory_key(candidate_metadata) == selected_trajectory
    ]
    method_candidates = [
        candidate
        for candidate, candidate_metadata in metadata.items()
        if str(candidate_metadata["method"]) == selected_method
    ]
    trajectory_oracle = min(
        trajectory_candidates,
        key=lambda candidate: (float(risks[candidate]), candidate),
    )
    method_oracle = min(
        method_candidates,
        key=lambda candidate: (float(risks[candidate]), candidate),
    )

    selected_risk = float(risks[selected])
    oracle_risk = float(risks[oracle_candidate])
    trajectory_oracle_risk = float(risks[trajectory_oracle])
    method_oracle_risk = float(risks[method_oracle])
    risk_range = float(max(risks.values()) - min(risks.values()))
    denominator = max(risk_range, 1.0e-12)
    total_gap = selected_risk - oracle_risk
    trajectory_gap = trajectory_oracle_risk - oracle_risk
    checkpoint_gap = selected_risk - trajectory_oracle_risk
    method_gap = method_oracle_risk - oracle_risk
    within_method_gap = selected_risk - method_oracle_risk
    if not np.isclose(total_gap, trajectory_gap + checkpoint_gap, atol=1.0e-12):
        raise AssertionError("trajectory decomposition is not exact")
    if not np.isclose(total_gap, method_gap + within_method_gap, atol=1.0e-12):
        raise AssertionError("method decomposition is not exact")

    return {
        "task": task,
        "selector": str(selection["selector"]),
        "selected_candidate_id": selected,
        "selected_candidate_metadata": selected_metadata,
        "oracle_candidate_id": oracle_candidate,
        "trajectory_oracle_candidate_id": trajectory_oracle,
        "method_oracle_candidate_id": method_oracle,
        "selected_target_error": selected_risk,
        "oracle_target_error": oracle_risk,
        "trajectory_oracle_target_error": trajectory_oracle_risk,
        "method_oracle_target_error": method_oracle_risk,
        "risk_range": risk_range,
        "total_gap": total_gap,
        "trajectory_gap": trajectory_gap,
        "checkpoint_gap": checkpoint_gap,
        "method_gap": method_gap,
        "within_method_gap": within_method_gap,
        "normalized_total_gap": total_gap / denominator,
        "normalized_trajectory_gap": trajectory_gap / denominator,
        "normalized_checkpoint_gap": checkpoint_gap / denominator,
        "normalized_method_gap": method_gap / denominator,
        "normalized_within_method_gap": within_method_gap / denominator,
        "trajectory_candidate_count": len(trajectory_candidates),
        "method_candidate_count": len(method_candidates),
    }


def aggregate_selector(reports: list[dict[str, Any]]) -> dict[str, Any]:
    normalized_total = sum(float(report["normalized_total_gap"]) for report in reports)
    normalized_trajectory = sum(
        float(report["normalized_trajectory_gap"]) for report in reports
    )
    normalized_checkpoint = sum(
        float(report["normalized_checkpoint_gap"]) for report in reports
    )
    normalized_method = sum(float(report["normalized_method_gap"]) for report in reports)
    normalized_within_method = sum(
        float(report["normalized_within_method_gap"]) for report in reports
    )
    count = len(reports)
    return {
        "selector": reports[0]["selector"],
        "task_count": count,
        "mean_normalized_total_gap": normalized_total / count,
        "mean_normalized_trajectory_gap": normalized_trajectory / count,
        "mean_normalized_checkpoint_gap": normalized_checkpoint / count,
        "mean_normalized_method_gap": normalized_method / count,
        "mean_normalized_within_method_gap": normalized_within_method / count,
        "trajectory_gap_share": (
            normalized_trajectory / normalized_total if normalized_total > 1.0e-12 else 0.0
        ),
        "checkpoint_gap_share": (
            normalized_checkpoint / normalized_total if normalized_total > 1.0e-12 else 0.0
        ),
        "method_gap_share": (
            normalized_method / normalized_total if normalized_total > 1.0e-12 else 0.0
        ),
        "within_method_gap_share": (
            normalized_within_method / normalized_total
            if normalized_total > 1.0e-12
            else 0.0
        ),
        "tasks": reports,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    tasks = args.tasks or list(OPEN_DEVELOPMENT_TASKS)
    reject_forbidden_path(args.dev_truth_report)
    for root in args.selection_root:
        reject_forbidden_path(root)
    truth = load_open_dev_truth(args.dev_truth_report.resolve())
    paths = discover_selection_paths(
        args.selection_root,
        selectors=args.selector,
        tasks=tasks,
    )
    by_selector: dict[str, list[dict[str, Any]]] = {}
    for path in paths:
        selection = load_selection(path)
        task = str(selection["task"])
        if task not in tasks:
            continue
        report = decompose_selection(selection, truth[task])
        by_selector.setdefault(report["selector"], []).append(report)

    expected = set(tasks)
    aggregates: dict[str, dict[str, Any]] = {}
    for selector, reports in sorted(by_selector.items()):
        covered = {report["task"] for report in reports}
        if covered != expected or len(reports) != len(tasks):
            raise ValueError(
                f"selector {selector} does not cover tasks exactly once: "
                f"covered={sorted(covered)} report_count={len(reports)}"
            )
        aggregates[selector] = aggregate_selector(reports)

    result = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "open_development_tasks": tasks,
        "selectors": aggregates,
        "label_access_count": 0,
        "protocol_violation_count": 0,
    }
    if args.output is not None:
        atomic_json(result, args.output.resolve())
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-truth-report", type=Path, required=True)
    parser.add_argument("--selection-root", type=Path, action="append", required=True)
    parser.add_argument("--selector", action="append", required=True)
    parser.add_argument("--tasks", nargs="*")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    result = run(parse_args())
    print(
        json.dumps(
            {
                "selector_count": len(result["selectors"]),
                "label_access_count": result["label_access_count"],
                "protocol_violation_count": result["protocol_violation_count"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
