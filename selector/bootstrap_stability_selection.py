#!/usr/bin/env python3
"""Build a bootstrap-stable Agreement shortlist from public target predictions.

For each task, the node-wise committee reference is fixed from all candidate
hard predictions. The selector then performs deterministic 80% node
subsampling, recomputes Agreement scores, and ranks candidates by top-budget
inclusion frequency. Transfer Score reranks the resulting fixed shortlist.
The diagnostic also records shortlist Jaccard, selection frequencies, entropy,
and boundary-margin stability without reading target labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections import Counter
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
from selector.consensus_selection import (  # noqa: E402
    load_task_selection,
    reject_forbidden_path,
)
from selector.reliable_selection import (  # noqa: E402
    SHORTLIST_EXCLUSION_SCORE,
    choose,
    percentile_ranks,
)
from selector.trajectory_shortlist_selection import trajectory_key  # noqa: E402


def _stable_task_seed(base_seed: int, task: str) -> int:
    digest = hashlib.sha256(task.encode("utf-8")).digest()
    return (int(base_seed) + int.from_bytes(digest[:4], "little")) % (2**32)


def _top_budget(
    scores: np.ndarray,
    candidate_ids: list[str],
    *,
    budget: int,
) -> set[int]:
    order = sorted(
        range(len(candidate_ids)),
        key=lambda index: (-float(scores[index]), candidate_ids[index]),
    )
    return set(order[:budget])


def _entropy(counter: Counter[str], total: int) -> float:
    if total <= 0:
        return 0.0
    probabilities = np.asarray([count / total for count in counter.values()], dtype=np.float64)
    return float(-np.sum(probabilities * np.log(np.clip(probabilities, 1.0e-12, 1.0))))


def bootstrap_agreement_diagnostics(
    predictions: np.ndarray,
    candidate_ids: list[str],
    reranker_ranks: dict[str, float],
    *,
    budget: int,
    bootstrap_count: int,
    node_fraction: float,
    seed: int,
) -> dict[str, Any]:
    values = np.asarray(predictions, dtype=np.int64)
    if values.ndim != 2:
        raise ValueError("predictions must have shape [models, nodes]")
    model_count, node_count = values.shape
    if len(candidate_ids) != model_count:
        raise ValueError("candidate ids do not align with prediction rows")
    if set(candidate_ids) != set(reranker_ranks):
        raise ValueError("reranker ranks do not align with candidate ids")
    if not 0 < budget <= model_count:
        raise ValueError("budget must lie in [1, model_count]")
    if bootstrap_count <= 0:
        raise ValueError("bootstrap_count must be positive")
    if not 0.0 < node_fraction <= 1.0:
        raise ValueError("node_fraction must lie in (0, 1]")

    class_count = int(values.max()) + 1
    node_indices = np.arange(node_count)
    counts = np.zeros((node_count, class_count), dtype=np.int32)
    for prediction in values:
        counts[node_indices, prediction] += 1
    reference = counts.argmax(axis=1)
    agreements = values == reference[None, :]
    full_scores = agreements.mean(axis=1, dtype=np.float64)
    full_shortlist = _top_budget(full_scores, candidate_ids, budget=budget)
    full_shortlist_trajectories = {
        trajectory_key(candidate_ids[index]) for index in full_shortlist
    }

    sample_size = max(1, int(round(node_fraction * node_count)))
    rng = np.random.default_rng(seed)
    inclusion_counts = np.zeros(model_count, dtype=np.int64)
    jaccards: list[float] = []
    trajectory_jaccards: list[float] = []
    margins: list[float] = []
    selected_candidates: Counter[str] = Counter()
    selected_trajectories: Counter[str] = Counter()
    selected_methods: Counter[str] = Counter()

    for _ in range(bootstrap_count):
        sample = rng.choice(node_count, size=sample_size, replace=False)
        scores = agreements[:, sample].mean(axis=1, dtype=np.float64)
        shortlist = _top_budget(scores, candidate_ids, budget=budget)
        for index in shortlist:
            inclusion_counts[index] += 1
        union = len(full_shortlist | shortlist)
        jaccards.append(len(full_shortlist & shortlist) / union if union else 1.0)
        shortlist_trajectories = {
            trajectory_key(candidate_ids[index]) for index in shortlist
        }
        trajectory_union = len(
            full_shortlist_trajectories | shortlist_trajectories
        )
        trajectory_jaccards.append(
            len(full_shortlist_trajectories & shortlist_trajectories)
            / trajectory_union
            if trajectory_union
            else 1.0
        )
        ordered_scores = sorted((float(score) for score in scores), reverse=True)
        margins.append(
            ordered_scores[budget - 1] - ordered_scores[budget]
            if budget < model_count
            else math.inf
        )
        selected_index = min(
            shortlist,
            key=lambda index: (reranker_ranks[candidate_ids[index]], candidate_ids[index]),
        )
        selected_id = candidate_ids[selected_index]
        selected_candidates[selected_id] += 1
        trajectory = trajectory_key(selected_id)
        selected_trajectories[trajectory] += 1
        selected_methods[selected_id.split("__")[1]] += 1

    inclusion_frequency = inclusion_counts.astype(np.float64) / bootstrap_count
    stable_order = sorted(
        range(model_count),
        key=lambda index: (
            -float(inclusion_frequency[index]),
            -float(full_scores[index]),
            candidate_ids[index],
        ),
    )
    stable_shortlist = set(stable_order[:budget])
    return {
        "stable_shortlist_indices": sorted(stable_shortlist),
        "full_agreement_scores": full_scores,
        "inclusion_frequency": inclusion_frequency,
        "mean_shortlist_jaccard": float(np.mean(jaccards)),
        "std_shortlist_jaccard": float(np.std(jaccards)),
        "mean_trajectory_shortlist_jaccard": float(
            np.mean(trajectory_jaccards)
        ),
        "std_trajectory_shortlist_jaccard": float(
            np.std(trajectory_jaccards)
        ),
        "mean_boundary_margin": float(np.mean(margins)),
        "std_boundary_margin": float(np.std(margins)),
        "selected_candidate_frequency": dict(sorted(selected_candidates.items())),
        "selected_trajectory_frequency": dict(sorted(selected_trajectories.items())),
        "selected_method_frequency": dict(sorted(selected_methods.items())),
        "selected_candidate_entropy": _entropy(selected_candidates, bootstrap_count),
        "selected_trajectory_entropy": _entropy(selected_trajectories, bootstrap_count),
        "selected_method_entropy": _entropy(selected_methods, bootstrap_count),
        "sample_size": sample_size,
    }


def build_bootstrap_selection(
    records: list[dict[str, Any]],
    reranker: dict[str, Any],
    *,
    task: str,
    selector_name: str,
    budget: int,
    bootstrap_count: int,
    node_fraction: float,
    seed: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    predictions = _hard_prediction_matrix(records)
    candidate_ids = [record["metadata"]["candidate_id"] for record in records]
    if set(candidate_ids) != set(reranker["candidate_scores"]):
        raise ValueError("candidate bank does not align with reranker selection")
    reranker_ranks = percentile_ranks(
        reranker["candidate_scores"], reranker["score_direction"]
    )
    diagnostics = bootstrap_agreement_diagnostics(
        predictions,
        candidate_ids,
        reranker_ranks,
        budget=budget,
        bootstrap_count=bootstrap_count,
        node_fraction=node_fraction,
        seed=seed,
    )
    stable_indices = set(diagnostics.pop("stable_shortlist_indices"))
    full_scores = diagnostics.pop("full_agreement_scores")
    inclusion_frequency = diagnostics.pop("inclusion_frequency")
    full_ranks = percentile_ranks(
        {
            candidate: float(full_scores[index])
            for index, candidate in enumerate(candidate_ids)
        },
        "maximize",
    )
    fused_scores = {
        candidate: float(
            reranker_ranks[candidate]
            if index in stable_indices
            else SHORTLIST_EXCLUSION_SCORE
            + (1.0 - float(inclusion_frequency[index]))
            + 1.0e-3 * full_ranks[candidate]
        )
        for index, candidate in enumerate(candidate_ids)
    }
    selected = choose(fused_scores, "minimize")
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task": task,
        "selector": selector_name,
        "candidate_bank_sha256": candidate_bank_hash(records),
        "candidate_count": len(records),
        "candidate_scores": fused_scores,
        "score_direction": "minimize",
        "score_semantics": "bootstrap_stable_agreement_shortlist_then_rerank",
        "selected_candidate_id": selected,
        "label_access_count": 0,
        "protocol_violation_count": 0,
        "selector_runtime_seconds": time.perf_counter() - started,
        "fusion_config": {
            "fusion_mode": "bootstrap_stable_shortlist_rerank",
            "shortlist_owner": "agreement_reference",
            "shortlist_size": len(stable_indices),
            "shortlist_exclusion_score": SHORTLIST_EXCLUSION_SCORE,
            "budget": int(budget),
            "bootstrap_count": int(bootstrap_count),
            "node_fraction": float(node_fraction),
            "seed": int(seed),
            "rerank_selector": reranker["selector"],
        },
        "candidate_inclusion_frequency": {
            candidate: float(inclusion_frequency[index])
            for index, candidate in enumerate(candidate_ids)
        },
        "bootstrap_diagnostics": diagnostics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--rerank-root", type=Path, required=True)
    parser.add_argument("--rerank-selector", default="transfer_score")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-selector", default="stable_agreement20_transfer_rerank")
    parser.add_argument("--budget", type=int, default=135)
    parser.add_argument("--bootstrap-count", type=int, default=32)
    parser.add_argument("--node-fraction", type=float, default=0.80)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--task", action="append", required=True, dest="tasks")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidate_root = args.candidate_root.resolve()
    rerank_root = args.rerank_root.resolve()
    output_root = args.output_root.resolve()
    for root in (candidate_root, rerank_root, output_root):
        reject_forbidden_path(root)
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selector": args.output_selector,
        "tasks": args.tasks,
        "candidate_root": str(candidate_root),
        "rerank_root": str(rerank_root),
        "rerank_selector": args.rerank_selector,
        "budget": int(args.budget),
        "bootstrap_count": int(args.bootstrap_count),
        "node_fraction": float(args.node_fraction),
        "base_seed": int(args.seed),
        "outputs": [],
        "label_access_count": 0,
        "protocol_violation_count": 0,
        "contains_target_labels": False,
    }
    for task in args.tasks:
        records = discover_candidate_records(candidate_root, task)
        reranker = load_task_selection(rerank_root, task, args.rerank_selector)
        task_seed = _stable_task_seed(args.seed, task)
        result = build_bootstrap_selection(
            records,
            reranker,
            task=task,
            selector_name=args.output_selector,
            budget=args.budget,
            bootstrap_count=args.bootstrap_count,
            node_fraction=args.node_fraction,
            seed=task_seed,
        )
        output = output_root / task / f"{args.output_selector}.json"
        atomic_json(result, output)
        manifest["outputs"].append(
            {
                "task": task,
                "output": str(output),
                "seed": task_seed,
                "selector_runtime_seconds": result["selector_runtime_seconds"],
            }
        )
    atomic_json(manifest, output_root / f"{args.output_selector}_manifest.json")
    print(
        json.dumps(
            {
                "task_count": len(args.tasks),
                "selector": args.output_selector,
                "label_access_count": 0,
                "protocol_violation_count": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
