#!/usr/bin/env python3
"""Cross-fitted Agreement Reference excluding a candidate's own group.

For every candidate, the target pseudo-reference is formed after removing all
checkpoints from the same method/config/seed trajectory (LOTO) or the same
adaptation method (LOMO).  The selector reads only ``target_public.npz`` hard
predictions and candidate metadata; no target labels are loaded.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.trajectory_export.schema import (  # noqa: E402
    atomic_json,
    candidate_bank_hash,
    discover_candidate_records,
)
from selector.baselines import _hard_prediction_matrix  # noqa: E402

CROSSFIT_MODES = ("trajectory", "method")


def record_group(record: dict[str, Any], mode: str) -> str:
    if mode not in CROSSFIT_MODES:
        raise ValueError(f"unknown crossfit mode: {mode}")
    metadata = record["metadata"]
    if mode == "method":
        return str(metadata["method"])
    return "__".join(
        (
            str(metadata["method"]),
            str(metadata["config_id"]),
            f"seed-{int(metadata['seed'])}",
        )
    )


def crossfit_agreement_scores(
    predictions: np.ndarray,
    groups: list[str],
) -> np.ndarray:
    values = np.asarray(predictions, dtype=np.int64)
    if values.ndim != 2:
        raise ValueError("predictions must have shape [models, nodes]")
    model_count, node_count = values.shape
    if len(groups) != model_count:
        raise ValueError("group count must match model count")
    if model_count < 2 or node_count < 1:
        raise ValueError("cross-fitted agreement needs at least two models and one node")
    class_count = int(values.max()) + 1
    node_indices = np.arange(node_count)
    global_counts = np.zeros((node_count, class_count), dtype=np.int32)
    for prediction in values:
        global_counts[node_indices, prediction] += 1

    scores = np.empty(model_count, dtype=np.float64)
    for group in sorted(set(groups)):
        members = np.asarray(
            [index for index, candidate_group in enumerate(groups) if candidate_group == group],
            dtype=np.int64,
        )
        if members.size >= model_count:
            raise ValueError("crossfit group cannot contain the full candidate bank")
        excluded_counts = np.zeros_like(global_counts)
        for member in members:
            excluded_counts[node_indices, values[member]] += 1
        reference = (global_counts - excluded_counts).argmax(axis=1)
        scores[members] = np.mean(values[members] == reference[None, :], axis=1)
    return scores


def build_selection(
    records: list[dict[str, Any]],
    *,
    task: str,
    mode: str,
    selector_name: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    predictions = _hard_prediction_matrix(records)
    groups = [record_group(record, mode) for record in records]
    values = crossfit_agreement_scores(predictions, groups)
    identifiers = [record["metadata"]["candidate_id"] for record in records]
    scores = {
        identifier: float(values[index]) for index, identifier in enumerate(identifiers)
    }
    optimum = max(scores.values())
    selected = min(candidate for candidate, score in scores.items() if score == optimum)
    group_sizes = {
        group: int(sum(candidate_group == group for candidate_group in groups))
        for group in sorted(set(groups))
    }
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task": task,
        "selector": selector_name,
        "candidate_bank_sha256": candidate_bank_hash(records),
        "candidate_count": len(records),
        "candidate_scores": scores,
        "score_direction": "maximize",
        "score_semantics": "cross_fitted_prediction_agreement",
        "selected_candidate_id": selected,
        "label_access_count": 0,
        "protocol_violation_count": 0,
        "crossfit_config": {
            "mode": mode,
            "group_count": len(group_sizes),
            "group_sizes": group_sizes,
            "reference_rule": "node-wise majority vote excluding candidate group",
        },
        "selector_runtime_seconds": time.perf_counter() - started,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--task", action="append", required=True, dest="tasks")
    parser.add_argument("--mode", choices=CROSSFIT_MODES, required=True)
    parser.add_argument("--output-selector", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidate_root = args.candidate_root.resolve()
    output_root = args.output_root.resolve()
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "candidate_root": str(candidate_root),
        "output_root": str(output_root),
        "tasks": args.tasks,
        "mode": args.mode,
        "selector": args.output_selector,
        "outputs": [],
        "label_access_count": 0,
        "protocol_violation_count": 0,
    }
    for task in args.tasks:
        records = discover_candidate_records(candidate_root, task)
        result = build_selection(
            records,
            task=task,
            mode=args.mode,
            selector_name=args.output_selector,
        )
        output = output_root / task / f"{args.output_selector}.json"
        atomic_json(result, output)
        manifest["outputs"].append(
            {
                "task": task,
                "output": str(output),
                "selector_runtime_seconds": result["selector_runtime_seconds"],
            }
        )
    atomic_json(manifest, output_root / f"{args.output_selector}_manifest.json")
    print(
        json.dumps(
            {
                "selector": args.output_selector,
                "task_count": len(args.tasks),
                "label_access_count": 0,
                "protocol_violation_count": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
