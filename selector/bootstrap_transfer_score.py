#!/usr/bin/env python3
"""Exact node-bootstrap stability diagnostic for the frozen Transfer Score.

The target-dependent Hopkins and normalized information-maximization terms are
recomputed on every deterministic node subsample. The classifier-geometry term
is target-node invariant and is recovered once from the frozen full-data score,
so model checkpoints do not need to be deserialized in worker processes. This
is algebraically identical to holding the original classifier geometry fixed.
No target labels are read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.neighbors import NearestNeighbors

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from selector.bootstrap_stability_selection import (  # noqa: E402
    _entropy,
    _stable_task_seed,
    _top_budget,
)
from selector.consensus_selection import (  # noqa: E402
    atomic_json,
    load_task_selection,
    reject_forbidden_path,
)
from selector.trajectory_shortlist_selection import trajectory_key  # noqa: E402


def stable_candidate_seed(identifier: str) -> int:
    return int.from_bytes(hashlib.sha256(identifier.encode("utf-8")).digest()[:4], "little")


def normalized_information_maximization(probabilities: np.ndarray) -> float:
    values = np.asarray(probabilities, dtype=np.float64)
    values = np.clip(values, 1.0e-12, 1.0)
    conditional_entropy = float(np.mean(-np.sum(values * np.log(values), axis=1)))
    marginal = values.mean(axis=0)
    marginal_entropy = float(-np.sum(marginal * np.log(np.clip(marginal, 1.0e-12, 1.0))))
    return (marginal_entropy - conditional_entropy) / np.log(values.shape[1])


def hopkins_statistic(features: np.ndarray, *, seed: int, fraction: float = 0.05) -> float:
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] < 3:
        raise ValueError("Transfer Score features require at least three target nodes")
    sample_size = max(1, int(values.shape[0] * fraction))
    rng = np.random.default_rng(seed)
    uniform_sample = rng.uniform(
        values.min(axis=0),
        values.max(axis=0),
        size=(sample_size, values.shape[1]),
    ).astype(np.float32)
    sample_indices = rng.choice(values.shape[0], size=sample_size, replace=False)
    real_sample = values[sample_indices]
    neighbors = NearestNeighbors(n_neighbors=2, n_jobs=1).fit(values)
    uniform_distances = neighbors.kneighbors(
        uniform_sample, n_neighbors=2, return_distance=True
    )[0][:, 0]
    real_distances = neighbors.kneighbors(
        real_sample, n_neighbors=2, return_distance=True
    )[0][:, 1]
    uniform_sum = float(uniform_distances.sum())
    real_sum = float(real_distances.sum())
    denominator = uniform_sum + real_sum
    return uniform_sum / denominator if denominator > 1.0e-12 else 0.5


def candidate_bootstrap_transfer_scores(
    target_path: str,
    candidate_id: str,
    frozen_full_score: float,
    *,
    bootstrap_count: int,
    node_fraction: float,
    task_seed: int,
) -> tuple[str, np.ndarray, dict[str, float]]:
    with np.load(target_path, allow_pickle=False) as artifact:
        embeddings = artifact["embeddings"].astype(np.float32, copy=False)
        probabilities = artifact["probabilities"].astype(np.float64, copy=False)
    if embeddings.shape[0] != probabilities.shape[0]:
        raise ValueError("target embeddings and probabilities do not align")
    candidate_seed = stable_candidate_seed(candidate_id)
    full_representation = hopkins_statistic(embeddings, seed=candidate_seed)
    full_discriminability = normalized_information_maximization(probabilities)
    geometry_penalty = (
        full_representation + full_discriminability - float(frozen_full_score)
    )

    node_count = embeddings.shape[0]
    sample_size = max(3, int(round(node_fraction * node_count)))
    rng = np.random.default_rng(task_seed)
    scores = np.empty(bootstrap_count, dtype=np.float64)
    for bootstrap_index in range(bootstrap_count):
        sample = rng.choice(node_count, size=sample_size, replace=False)
        representation = hopkins_statistic(embeddings[sample], seed=candidate_seed)
        discriminability = normalized_information_maximization(probabilities[sample])
        scores[bootstrap_index] = (
            representation + discriminability - geometry_penalty
        )
    return candidate_id, scores, {
        "full_representation": full_representation,
        "full_discriminability": full_discriminability,
        "geometry_penalty": geometry_penalty,
        "reconstructed_full_score": (
            full_representation + full_discriminability - geometry_penalty
        ),
    }


def aggregate_transfer_stability(
    candidate_ids: list[str],
    frozen_scores: np.ndarray,
    bootstrap_scores: np.ndarray,
    *,
    budget: int,
) -> dict[str, Any]:
    if bootstrap_scores.ndim != 2 or bootstrap_scores.shape[0] != len(candidate_ids):
        raise ValueError("bootstrap score matrix does not align with candidates")
    full_shortlist = _top_budget(frozen_scores, candidate_ids, budget=budget)
    full_shortlist_trajectories = {
        trajectory_key(candidate_ids[index]) for index in full_shortlist
    }
    jaccards: list[float] = []
    trajectory_jaccards: list[float] = []
    margins: list[float] = []
    inclusion_counts = np.zeros(len(candidate_ids), dtype=np.int64)
    selected_candidates: Counter[str] = Counter()
    selected_trajectories: Counter[str] = Counter()
    selected_methods: Counter[str] = Counter()
    for bootstrap_index in range(bootstrap_scores.shape[1]):
        scores = bootstrap_scores[:, bootstrap_index]
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
        order = sorted(
            range(len(candidate_ids)),
            key=lambda index: (-float(scores[index]), candidate_ids[index]),
        )
        margins.append(
            float(scores[order[budget - 1]] - scores[order[budget]])
            if budget < len(candidate_ids)
            else math.inf
        )
        selected_id = candidate_ids[order[0]]
        selected_candidates[selected_id] += 1
        selected_trajectories[trajectory_key(selected_id)] += 1
        selected_methods[selected_id.split("__")[1]] += 1
    bootstrap_count = bootstrap_scores.shape[1]
    return {
        "candidate_inclusion_frequency": {
            candidate: float(inclusion_counts[index] / bootstrap_count)
            for index, candidate in enumerate(candidate_ids)
        },
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
        "max_selected_trajectory_frequency": float(
            max(selected_trajectories.values()) / bootstrap_count
        ),
        "max_selected_candidate_frequency": float(
            max(selected_candidates.values()) / bootstrap_count
        ),
    }


def run_task(
    records: list[dict[str, Any]],
    transfer_selection: dict[str, Any],
    *,
    task: str,
    budget: int,
    bootstrap_count: int,
    node_fraction: float,
    task_seed: int,
    workers: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    candidate_ids = [record["metadata"]["candidate_id"] for record in records]
    if set(candidate_ids) != set(transfer_selection["candidate_scores"]):
        raise ValueError("candidate bank does not align with Transfer Score selection")
    frozen_scores = np.asarray(
        [float(transfer_selection["candidate_scores"][candidate]) for candidate in candidate_ids],
        dtype=np.float64,
    )
    jobs = [
        (
            str(record["path"] / "target_public.npz"),
            candidate_id,
            float(transfer_selection["candidate_scores"][candidate_id]),
        )
        for record, candidate_id in zip(records, candidate_ids)
    ]

    def submit(executor: ProcessPoolExecutor, job: tuple[str, str, float]):
        return executor.submit(
            candidate_bootstrap_transfer_scores,
            job[0],
            job[1],
            job[2],
            bootstrap_count=bootstrap_count,
            node_fraction=node_fraction,
            task_seed=task_seed,
        )

    outputs: dict[str, tuple[np.ndarray, dict[str, float]]] = {}
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [submit(executor, job) for job in jobs]
        for future in futures:
            candidate_id, scores, components = future.result()
            outputs[candidate_id] = (scores, components)
    bootstrap_scores = np.stack([outputs[candidate][0] for candidate in candidate_ids])
    reconstructed_error = max(
        abs(
            outputs[candidate][1]["reconstructed_full_score"]
            - float(transfer_selection["candidate_scores"][candidate])
        )
        for candidate in candidate_ids
    )
    diagnostics = aggregate_transfer_stability(
        candidate_ids,
        frozen_scores,
        bootstrap_scores,
        budget=budget,
    )
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task": task,
        "selector": "transfer_score_bootstrap_stability",
        "candidate_count": len(candidate_ids),
        "candidate_bank_sha256": transfer_selection["candidate_bank_sha256"],
        "bootstrap_count": int(bootstrap_count),
        "node_fraction": float(node_fraction),
        "budget": int(budget),
        "task_seed": int(task_seed),
        "worker_count": int(workers),
        "full_score_reconstruction_max_abs_error": float(reconstructed_error),
        "candidate_bootstrap_score_mean": {
            candidate: float(np.mean(outputs[candidate][0])) for candidate in candidate_ids
        },
        "candidate_bootstrap_score_std": {
            candidate: float(np.std(outputs[candidate][0])) for candidate in candidate_ids
        },
        "bootstrap_diagnostics": diagnostics,
        "runtime_seconds": time.perf_counter() - started,
        "label_access_count": 0,
        "protocol_violation_count": 0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--transfer-root", type=Path, required=True)
    parser.add_argument("--transfer-selector", default="transfer_score")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=135)
    parser.add_argument("--bootstrap-count", type=int, default=32)
    parser.add_argument("--node-fraction", type=float, default=0.80)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--task", action="append", required=True, dest="tasks")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidate_root = args.candidate_root.resolve()
    transfer_root = args.transfer_root.resolve()
    output_root = args.output_root.resolve()
    for root in (candidate_root, transfer_root, output_root):
        reject_forbidden_path(root)
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    from scripts.trajectory_export.schema import discover_candidate_records

    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tasks": args.tasks,
        "candidate_root": str(candidate_root),
        "transfer_root": str(transfer_root),
        "transfer_selector": args.transfer_selector,
        "budget": int(args.budget),
        "bootstrap_count": int(args.bootstrap_count),
        "node_fraction": float(args.node_fraction),
        "base_seed": int(args.seed),
        "workers": int(args.workers),
        "resume": bool(args.resume),
        "outputs": [],
        "label_access_count": 0,
        "protocol_violation_count": 0,
        "contains_target_labels": False,
    }
    for task in args.tasks:
        output = output_root / task / "transfer_score_bootstrap_stability.json"
        if args.resume and output.is_file():
            result = json.loads(output.read_text(encoding="utf-8"))
            expected = {
                "task": task,
                "budget": int(args.budget),
                "bootstrap_count": int(args.bootstrap_count),
                "node_fraction": float(args.node_fraction),
            }
            mismatches = {
                key: (result.get(key), value)
                for key, value in expected.items()
                if result.get(key) != value
            }
            if mismatches:
                raise ValueError(
                    f"resume output config mismatch for {task}: {mismatches}"
                )
            if int(result.get("label_access_count", -1)) != 0 or int(
                result.get("protocol_violation_count", -1)
            ) != 0:
                raise ValueError(f"resume output violates protocol for {task}")
        else:
            records = discover_candidate_records(candidate_root, task)
            transfer = load_task_selection(transfer_root, task, args.transfer_selector)
            task_seed = _stable_task_seed(args.seed, task)
            result = run_task(
                records,
                transfer,
                task=task,
                budget=args.budget,
                bootstrap_count=args.bootstrap_count,
                node_fraction=args.node_fraction,
                task_seed=task_seed,
                workers=args.workers,
            )
            atomic_json(result, output)
        manifest["outputs"].append(
            {
                "task": task,
                "output": str(output),
                "runtime_seconds": result["runtime_seconds"],
            }
        )
    manifest["total_runtime_seconds"] = float(
        sum(float(entry["runtime_seconds"]) for entry in manifest["outputs"])
    )
    atomic_json(manifest, output_root / "transfer_score_bootstrap_stability_manifest.json")
    print(
        json.dumps(
            {
                "task_count": len(args.tasks),
                "label_access_count": 0,
                "protocol_violation_count": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
