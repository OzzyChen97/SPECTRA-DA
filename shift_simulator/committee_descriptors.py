#!/usr/bin/env python3
"""Label-free descriptors of candidate-committee behavior on a target graph.

Graph descriptors answer whether two graphs look similar.  Committee
descriptors answer whether the same frozen candidate bank behaves similarly:
entropy, margins, class-prior concentration, disagreement geometry, effective
prediction rank, method/trajectory diversity, and checkpoint drift.  The
extractor reads only ``target_public.npz`` and candidate metadata by default.
Transfer Score component summaries are optional because they require reading
classifier heads from checkpoints.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.trajectory_export.schema import (  # noqa: E402
    atomic_json,
    candidate_bank_hash,
    discover_candidate_records,
)

QUANTILE_NAMES = ("q10", "q50", "q90")
QUANTILES = np.asarray([0.10, 0.50, 0.90], dtype=np.float64)

COMMITTEE_DESCRIPTOR_NAMES = (
    "candidate_entropy_mean",
    "candidate_entropy_std",
    "candidate_entropy_q10",
    "candidate_entropy_q50",
    "candidate_entropy_q90",
    "candidate_margin_mean",
    "candidate_margin_std",
    "candidate_margin_q10",
    "candidate_margin_q50",
    "candidate_margin_q90",
    "class_prior_entropy_mean",
    "class_prior_entropy_std",
    "class_prior_max_mean",
    "class_prior_max_std",
    "pair_disagreement_mean",
    "pair_disagreement_std",
    "pair_disagreement_q10",
    "pair_disagreement_q50",
    "pair_disagreement_q90",
    "prediction_kernel_effective_rank",
    "prediction_kernel_lambda1",
    "prediction_kernel_lambda2",
    "prediction_kernel_lambda3",
    "prediction_kernel_lambda4",
    "prediction_kernel_lambda5",
    "within_method_disagreement",
    "between_method_disagreement",
    "same_trajectory_disagreement",
    "cross_method_disagreement",
    "checkpoint_drift_mean",
    "checkpoint_drift_std",
)

TRANSFER_COMPONENT_DESCRIPTOR_NAMES = (
    "transfer_hopkins_mean",
    "transfer_hopkins_std",
    "transfer_infomax_mean",
    "transfer_infomax_std",
    "transfer_geometry_penalty_mean",
    "transfer_geometry_penalty_std",
)


def _load_target(record: dict[str, Any]) -> dict[str, np.ndarray]:
    with np.load(record["path"] / "target_public.npz", allow_pickle=False) as artifact:
        keys = set(artifact.files)
        forbidden = keys & {
            "label",
            "labels",
            "target_label",
            "target_labels",
            "target_y",
            "y",
            "ground_truth",
        }
        if forbidden:
            raise RuntimeError(f"target_public contains label-like fields: {sorted(forbidden)}")
        return {key: artifact[key] for key in artifact.files}


def _summary(values: np.ndarray) -> list[float]:
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    if vector.size == 0 or not np.isfinite(vector).all():
        raise ValueError("summary values must be finite and non-empty")
    quantiles = np.quantile(vector, QUANTILES)
    return [
        float(np.mean(vector)),
        float(np.std(vector)),
        *[float(value) for value in quantiles],
    ]


def _prediction_arrays(records: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    probabilities = []
    hard_predictions = []
    for record in records:
        target = _load_target(record)
        probabilities.append(target["probabilities"].astype(np.float64, copy=False))
        hard_predictions.append(target["hard_predictions"].astype(np.int64, copy=False))
    probability_shapes = {array.shape for array in probabilities}
    prediction_shapes = {array.shape for array in hard_predictions}
    if len(probability_shapes) != 1 or len(prediction_shapes) != 1:
        raise ValueError("candidate target prediction shapes do not match")
    stacked_probabilities = np.stack(probabilities, axis=0)
    stacked_predictions = np.stack(hard_predictions, axis=0)
    if stacked_probabilities.ndim != 3 or stacked_predictions.ndim != 2:
        raise ValueError("invalid target prediction arrays")
    if stacked_probabilities.shape[:2] != stacked_predictions.shape:
        raise ValueError("probabilities and hard predictions are not aligned")
    return stacked_probabilities, stacked_predictions


def candidate_entropy(probabilities: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(probabilities, dtype=np.float64), 1.0e-12, 1.0)
    return -np.sum(values * np.log(values), axis=2).mean(axis=1)


def candidate_margin(probabilities: np.ndarray) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.shape[2] < 2:
        return np.ones(values.shape[0], dtype=np.float64)
    partition = np.partition(values, kth=values.shape[2] - 2, axis=2)
    top1 = partition[:, :, -1]
    top2 = partition[:, :, -2]
    return np.mean(top1 - top2, axis=1)


def class_prior_summaries(probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    priors = np.asarray(probabilities, dtype=np.float64).mean(axis=1)
    entropy = -np.sum(np.clip(priors, 1.0e-12, 1.0) * np.log(np.clip(priors, 1.0e-12, 1.0)), axis=1)
    max_prior = np.max(priors, axis=1)
    return entropy, max_prior


def pairwise_disagreement_matrix(predictions: np.ndarray, class_count: int | None = None) -> np.ndarray:
    labels = np.asarray(predictions, dtype=np.int64)
    if labels.ndim != 2:
        raise ValueError("predictions must have shape [candidates, nodes]")
    model_count, node_count = labels.shape
    if model_count < 2:
        raise ValueError("at least two candidates are required")
    classes = int(class_count or (labels.max(initial=0) + 1))
    agreements = np.zeros((model_count, model_count), dtype=np.float64)
    for class_index in range(classes):
        membership = (labels == class_index).astype(np.float64, copy=False)
        agreements += membership @ membership.T
    agreements /= max(1, node_count)
    disagreements = 1.0 - agreements
    np.fill_diagonal(disagreements, 0.0)
    return disagreements


def prediction_kernel_spectrum(probabilities: np.ndarray, top_k: int = 5) -> tuple[float, list[float]]:
    values = np.asarray(probabilities, dtype=np.float64)
    flattened = values.reshape(values.shape[0], -1)
    flattened -= flattened.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(flattened, axis=1, keepdims=True)
    normalized = flattened / np.clip(norms, 1.0e-12, None)
    gram = normalized @ normalized.T
    eigenvalues = np.linalg.eigvalsh(gram)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    total = float(np.sum(eigenvalues))
    if total <= 1.0e-12:
        return 1.0, [1.0] + [0.0] * (top_k - 1)
    normalized_eigenvalues = eigenvalues[::-1] / total
    effective_rank = total**2 / float(np.sum(eigenvalues**2) + 1.0e-12)
    top = normalized_eigenvalues[:top_k].tolist()
    top.extend([0.0] * (top_k - len(top)))
    return float(effective_rank), [float(value) for value in top]


def _pair_values(matrix: np.ndarray, pairs: list[tuple[int, int]]) -> np.ndarray:
    if not pairs:
        return np.asarray([], dtype=np.float64)
    return np.asarray([matrix[i, j] for i, j in pairs], dtype=np.float64)


def _mean_or_zero(values: np.ndarray) -> float:
    return float(np.mean(values)) if values.size else 0.0


def _trajectory_key(metadata: dict[str, Any]) -> tuple[str, str, int]:
    return (
        str(metadata.get("method", "")),
        str(metadata.get("config_id", "")),
        int(metadata.get("seed", -1)),
    )


def group_disagreement_features(
    records: list[dict[str, Any]],
    disagreement: np.ndarray,
) -> tuple[float, float, float, float, float, float]:
    methods = [str(record["metadata"].get("method", "")) for record in records]
    trajectories = [_trajectory_key(record["metadata"]) for record in records]
    within_method: list[tuple[int, int]] = []
    between_method: list[tuple[int, int]] = []
    same_trajectory: list[tuple[int, int]] = []
    cross_method: list[tuple[int, int]] = []
    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            if methods[i] == methods[j]:
                within_method.append((i, j))
            else:
                between_method.append((i, j))
                cross_method.append((i, j))
            if trajectories[i] == trajectories[j]:
                same_trajectory.append((i, j))
    drift_values = []
    trajectory_to_indices: dict[tuple[str, str, int], list[int]] = defaultdict(list)
    for index, key in enumerate(trajectories):
        trajectory_to_indices[key].append(index)
    for indices in trajectory_to_indices.values():
        ordered = sorted(indices, key=lambda index: int(records[index]["metadata"].get("epoch", 0)))
        for left, right in zip(ordered, ordered[1:]):
            drift_values.append(disagreement[left, right])
    drift = np.asarray(drift_values, dtype=np.float64)
    return (
        _mean_or_zero(_pair_values(disagreement, within_method)),
        _mean_or_zero(_pair_values(disagreement, between_method)),
        _mean_or_zero(_pair_values(disagreement, same_trajectory)),
        _mean_or_zero(_pair_values(disagreement, cross_method)),
        _mean_or_zero(drift),
        float(np.std(drift)) if drift.size else 0.0,
    )


def committee_descriptor(records: list[dict[str, Any]]) -> tuple[np.ndarray, list[str]]:
    """Compute label-free committee descriptors for aligned candidate records."""

    if len(records) < 2:
        raise ValueError("committee descriptor requires at least two candidates")
    probabilities, predictions = _prediction_arrays(records)
    class_count = probabilities.shape[2]
    entropy_values = candidate_entropy(probabilities)
    margin_values = candidate_margin(probabilities)
    prior_entropy, prior_max = class_prior_summaries(probabilities)
    disagreement = pairwise_disagreement_matrix(predictions, class_count=class_count)
    first, second = np.triu_indices(disagreement.shape[0], k=1)
    pair_values = disagreement[first, second]
    effective_rank, eigenvalues = prediction_kernel_spectrum(probabilities, top_k=5)
    group_values = group_disagreement_features(records, disagreement)
    descriptor = np.asarray(
        [
            *_summary(entropy_values),
            *_summary(margin_values),
            float(np.mean(prior_entropy)),
            float(np.std(prior_entropy)),
            float(np.mean(prior_max)),
            float(np.std(prior_max)),
            *_summary(pair_values),
            effective_rank,
            *eigenvalues,
            *group_values,
        ],
        dtype=np.float64,
    )
    if descriptor.shape != (len(COMMITTEE_DESCRIPTOR_NAMES),) or not np.isfinite(descriptor).all():
        raise FloatingPointError("committee descriptor contains invalid values")
    return descriptor, list(COMMITTEE_DESCRIPTOR_NAMES)


def transfer_component_descriptor(records: list[dict[str, Any]]) -> tuple[np.ndarray, list[str]]:
    """Summarize Transfer Score components without using target labels."""

    from selector.transfer_score import (  # noqa: WPS433
        _stable_seed,
        classifier_head_weight,
        classifier_uniformity,
        hopkins_statistic,
        normalized_information_maximization,
    )

    hopkins_values = []
    infomax_values = []
    geometry_values = []
    for record in records:
        target = _load_target(record)
        identifier = str(record["metadata"]["candidate_id"])
        embeddings = target["embeddings"]
        probabilities = target["probabilities"]
        head = classifier_head_weight(
            record["path"] / "model_state.pt",
            class_count=probabilities.shape[1],
            embedding_dim=embeddings.shape[1],
        )
        hopkins_values.append(
            hopkins_statistic(embeddings, seed=_stable_seed(identifier))
        )
        infomax_values.append(normalized_information_maximization(probabilities))
        geometry_values.append(classifier_uniformity(head))
    descriptor = np.asarray(
        [
            float(np.mean(hopkins_values)),
            float(np.std(hopkins_values)),
            float(np.mean(infomax_values)),
            float(np.std(infomax_values)),
            float(np.mean(geometry_values)),
            float(np.std(geometry_values)),
        ],
        dtype=np.float64,
    )
    if not np.isfinite(descriptor).all():
        raise FloatingPointError("transfer component descriptor contains invalid values")
    return descriptor, list(TRANSFER_COMPONENT_DESCRIPTOR_NAMES)


def build_descriptor_report(
    records: list[dict[str, Any]],
    *,
    task: str,
    include_transfer_components: bool = False,
) -> dict[str, Any]:
    descriptor, names = committee_descriptor(records)
    if include_transfer_components:
        transfer_descriptor, transfer_names = transfer_component_descriptor(records)
        descriptor = np.concatenate([descriptor, transfer_descriptor])
        names = names + transfer_names
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task": task,
        "candidate_count": len(records),
        "candidate_bank_sha256": candidate_bank_hash(records),
        "descriptor_family": (
            "committee_plus_transfer_components"
            if include_transfer_components
            else "committee"
        ),
        "descriptor_names": names,
        "descriptor": [float(value) for value in descriptor],
        "label_access_count": 0,
        "protocol_violation_count": 0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-transfer-components", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = discover_candidate_records(args.candidate_root.resolve(), args.task)
    report = build_descriptor_report(
        records,
        task=args.task,
        include_transfer_components=args.include_transfer_components,
    )
    atomic_json(report, args.output.resolve())
    print(
        json.dumps(
            {
                "task": report["task"],
                "candidate_count": report["candidate_count"],
                "descriptor_dimension": len(report["descriptor"]),
                "descriptor_family": report["descriptor_family"],
                "output": str(args.output.resolve()),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
