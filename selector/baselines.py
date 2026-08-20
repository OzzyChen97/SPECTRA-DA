#!/usr/bin/env python3
"""Run pre-registered label-free selectors over a trajectory candidate bank."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from scipy.stats import norm

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.trajectory_export.schema import (  # noqa: E402
    atomic_json,
    candidate_bank_hash,
    discover_candidate_records,
)
from selector.dev import adapted_dev_score  # noqa: E402
from selector.transfer_score import classifier_head_weight, transfer_score  # noqa: E402

SELECTION_SCHEMA_VERSION = 1


def _load_target(record: dict) -> dict[str, np.ndarray]:
    with np.load(record["path"] / "target_public.npz", allow_pickle=False) as artifact:
        return {key: artifact[key] for key in artifact.files}


def _load_source_validation(record: dict) -> dict[str, np.ndarray]:
    with np.load(record["path"] / "source_val.npz", allow_pickle=False) as artifact:
        return {key: artifact[key] for key in artifact.files}


def source_validation(records: list[dict]) -> dict[str, float]:
    return {
        record["metadata"]["candidate_id"]: float(record["metadata"]["source_val_micro_f1"])
        for record in records
    }


def last_source_validation(records: list[dict]) -> dict[str, float]:
    terminal_epoch: dict[tuple[str, str, int], int] = {}
    for record in records:
        metadata = record["metadata"]
        trajectory = (metadata["method"], metadata["config_id"], int(metadata["seed"]))
        terminal_epoch[trajectory] = max(terminal_epoch.get(trajectory, -1), int(metadata["epoch"]))

    scores = {}
    for record in records:
        metadata = record["metadata"]
        trajectory = (metadata["method"], metadata["config_id"], int(metadata["seed"]))
        is_terminal = int(metadata["epoch"]) == terminal_epoch[trajectory]
        source_score = float(metadata["source_val_micro_f1"])
        scores[metadata["candidate_id"]] = source_score if is_terminal else source_score - 2.0
    return scores


def target_entropy(records: list[dict]) -> dict[str, float]:
    return {
        record["metadata"]["candidate_id"]: float(record["metadata"]["target_entropy"])
        for record in records
    }


def information_maximization(records: list[dict]) -> dict[str, float]:
    scores = {}
    for record in records:
        probabilities = _load_target(record)["probabilities"].astype(np.float64, copy=False)
        probabilities = np.clip(probabilities, 1e-12, 1.0)
        conditional_entropy = float(np.mean(-np.sum(probabilities * np.log(probabilities), axis=1)))
        marginal = probabilities.mean(axis=0)
        marginal_entropy = float(-np.sum(marginal * np.log(np.clip(marginal, 1e-12, 1.0))))
        scores[record["metadata"]["candidate_id"]] = marginal_entropy - conditional_entropy
    return scores


def _hard_prediction_matrix(records: list[dict]) -> np.ndarray:
    predictions = [_load_target(record)["hard_predictions"].astype(np.int64, copy=False) for record in records]
    rows = {prediction.shape for prediction in predictions}
    if len(rows) != 1:
        raise ValueError("candidate target-prediction shapes do not match")
    return np.stack(predictions, axis=0)


def agreement_reference(records: list[dict]) -> dict[str, float]:
    predictions = _hard_prediction_matrix(records)
    num_classes = int(predictions.max()) + 1
    counts = np.zeros((predictions.shape[1], num_classes), dtype=np.int32)
    node_indices = np.arange(predictions.shape[1])
    for candidate_predictions in predictions:
        counts[node_indices, candidate_predictions] += 1
    reference = counts.argmax(axis=1)
    return {
        record["metadata"]["candidate_id"]: float(np.mean(predictions[index] == reference))
        for index, record in enumerate(records)
    }


def global_disagreement(records: list[dict]) -> dict[str, float]:
    predictions = _hard_prediction_matrix(records)
    if predictions.shape[0] < 2:
        return {records[0]["metadata"]["candidate_id"]: 0.0}
    num_classes = int(predictions.max()) + 1
    counts = np.zeros((predictions.shape[1], num_classes), dtype=np.int32)
    node_indices = np.arange(predictions.shape[1])
    for candidate_predictions in predictions:
        counts[node_indices, candidate_predictions] += 1

    denominator = predictions.shape[1] * (predictions.shape[0] - 1)
    scores = {}
    for index, record in enumerate(records):
        agreements = counts[node_indices, predictions[index]] - 1
        scores[record["metadata"]["candidate_id"]] = float(1.0 - agreements.sum() / denominator)
    return scores


def generalization_disagreement_equality(records: list[dict]) -> dict[str, float]:
    """Estimate error from disagreement between independent training replicas."""

    groups: dict[tuple[str, str, int], list[tuple[dict, np.ndarray]]] = {}
    for record in records:
        metadata = record["metadata"]
        key = (metadata["method"], metadata["config_id"], int(metadata["epoch"]))
        prediction = _load_target(record)["hard_predictions"].astype(np.int64, copy=False)
        groups.setdefault(key, []).append((record, prediction))

    scores: dict[str, float] = {}
    for key, members in groups.items():
        seeds = [int(record["metadata"]["seed"]) for record, _ in members]
        if len(members) < 2 or len(seeds) != len(set(seeds)):
            raise ValueError(f"GDE requires at least two distinct seeds for trajectory {key}")
        for index, (record, prediction) in enumerate(members):
            disagreements = [
                float(np.mean(prediction != peer_prediction))
                for peer_index, (_, peer_prediction) in enumerate(members)
                if peer_index != index
            ]
            scores[record["metadata"]["candidate_id"]] = float(np.mean(disagreements))
    return scores


def _snd_device() -> torch.device:
    """Use GPU only when the benchmark's pre-registered physical GPU is visible."""

    if os.environ.get("CUDA_VISIBLE_DEVICES") == "7" and torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def soft_neighborhood_density(
    features: np.ndarray,
    *,
    temperature: float = 0.05,
    block_size: int = 2048,
    device: torch.device | str | None = None,
) -> float:
    """Compute the exact SND entropy without materializing the full matrix.

    This follows the public SND implementation: row-normalize target softmax
    predictions, divide pairwise cosine similarities by ``T=0.05``, replace
    self-similarities with ``-1/T``, and maximize the mean row entropy.  Row
    blocking changes only memory use, not the statistic.
    """

    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("SND features must have shape [nodes, dimensions] with at least two nodes")
    if temperature <= 0.0:
        raise ValueError("SND temperature must be positive")
    if block_size <= 0:
        raise ValueError("SND block size must be positive")

    target_device = torch.device(device) if device is not None else _snd_device()
    tensor = torch.from_numpy(values).to(target_device)
    tensor = torch.nn.functional.normalize(tensor, p=2, dim=1)
    node_count = tensor.shape[0]
    entropy_sum = torch.zeros((), dtype=torch.float64, device=target_device)
    with torch.inference_mode():
        for start in range(0, node_count, block_size):
            stop = min(start + block_size, node_count)
            similarities = tensor[start:stop] @ tensor.T
            similarities.div_(temperature)
            local = torch.arange(stop - start, device=target_device)
            similarities[local, local + start] = -1.0 / temperature
            probabilities = torch.softmax(similarities, dim=1)
            row_entropy = -(probabilities * torch.log(probabilities + 1e-5)).sum(dim=1)
            entropy_sum += row_entropy.double().sum()
    return float((entropy_sum / node_count).item())


def soft_neighborhood_density_selector(records: list[dict]) -> dict[str, float]:
    """Official SND variant over target softmax predictions (larger is better)."""

    device = _snd_device()
    scores: dict[str, float] = {}
    for record in records:
        probabilities = _load_target(record)["probabilities"]
        identifier = record["metadata"]["candidate_id"]
        scores[identifier] = soft_neighborhood_density(probabilities, device=device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return scores


def _source_prediction_matrix(records: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    predictions: list[np.ndarray] = []
    reference_indices: np.ndarray | None = None
    reference_labels: np.ndarray | None = None
    for record in records:
        artifact = _load_source_validation(record)
        indices = artifact["indices"].astype(np.int64, copy=False)
        labels = artifact["labels"].astype(np.int64, copy=False)
        if reference_indices is None:
            reference_indices = indices
            reference_labels = labels
        elif not np.array_equal(indices, reference_indices) or not np.array_equal(labels, reference_labels):
            raise ValueError("AoL requires aligned source-validation nodes and labels across candidates")
        predictions.append(artifact["hard_predictions"].astype(np.int64, copy=False))
    assert reference_labels is not None
    return np.stack(predictions, axis=0), reference_labels


def _pairwise_agreements(predictions: np.ndarray) -> np.ndarray:
    if predictions.ndim != 2:
        raise ValueError("predictions must have shape [models, nodes]")
    model_count = predictions.shape[0]
    agreements = np.eye(model_count, dtype=np.float64)
    for left in range(model_count - 1):
        values = np.mean(predictions[left + 1 :] == predictions[left], axis=1, dtype=np.float64)
        agreements[left, left + 1 :] = values
        agreements[left + 1 :, left] = values
    return agreements


def agreement_on_the_line_scores(
    source_predictions: np.ndarray,
    source_labels: np.ndarray,
    target_predictions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the official ALine-S and ALine-D estimated accuracies.

    Only source-validation labels are used.  Target inputs are hard predictions,
    and pair filtering follows the released implementation's [0.05, 0.98]
    agreement interval before probit-space regression.
    """

    source_predictions = np.asarray(source_predictions, dtype=np.int64)
    target_predictions = np.asarray(target_predictions, dtype=np.int64)
    source_labels = np.asarray(source_labels, dtype=np.int64)
    if source_predictions.ndim != 2 or target_predictions.ndim != 2:
        raise ValueError("AoL prediction arrays must have shape [models, nodes]")
    if source_predictions.shape[0] != target_predictions.shape[0]:
        raise ValueError("AoL source and target model counts do not match")
    if source_predictions.shape[1] != source_labels.shape[0]:
        raise ValueError("AoL source labels do not align with source predictions")
    model_count = source_predictions.shape[0]
    if model_count < 3:
        raise ValueError("AoL requires at least three candidate models")

    source_accuracy = np.mean(source_predictions == source_labels[None, :], axis=1, dtype=np.float64)
    source_agreement = _pairwise_agreements(source_predictions)
    target_agreement = _pairwise_agreements(target_predictions)
    left, right = np.triu_indices(model_count, k=1)
    source_pairs = source_agreement[left, right]
    target_pairs = target_agreement[left, right]
    keep = (
        (source_pairs >= 0.05)
        & (source_pairs <= 0.98)
        & (target_pairs >= 0.05)
        & (target_pairs <= 0.98)
    )
    left = left[keep]
    right = right[keep]
    source_pairs = source_pairs[keep]
    target_pairs = target_pairs[keep]
    if source_pairs.size < model_count:
        raise ValueError("AoL has too few non-degenerate candidate pairs after agreement filtering")

    source_pair_probit = norm.ppf(source_pairs)
    target_pair_probit = norm.ppf(target_pairs)
    design = np.column_stack((np.ones_like(source_pair_probit), source_pair_probit))
    (bias, slope), _, rank, _ = np.linalg.lstsq(design, target_pair_probit, rcond=None)
    if rank < 2:
        raise ValueError("AoL source agreements have insufficient variation for line fitting")

    epsilon = np.finfo(np.float64).eps
    source_accuracy_probit = norm.ppf(np.clip(source_accuracy, epsilon, 1.0 - epsilon))
    aline_s = norm.cdf(slope * source_accuracy_probit + bias)

    pair_accuracy_probit = 0.5 * (source_accuracy_probit[left] + source_accuracy_probit[right])
    observations = target_pair_probit + slope * (pair_accuracy_probit - source_pair_probit)
    direct_design = np.zeros((source_pairs.size, model_count), dtype=np.float64)
    rows = np.arange(source_pairs.size)
    direct_design[rows, left] = 0.5
    direct_design[rows, right] = 0.5
    direct_solution, _, _, _ = np.linalg.lstsq(direct_design, observations, rcond=None)
    aline_d = norm.cdf(direct_solution)
    if not np.isfinite(aline_s).all() or not np.isfinite(aline_d).all():
        raise ValueError("AoL produced non-finite accuracy estimates")
    return aline_s, aline_d


def _agreement_on_the_line(records: list[dict]) -> tuple[dict[str, float], dict[str, float]]:
    source_predictions, source_labels = _source_prediction_matrix(records)
    target_predictions = _hard_prediction_matrix(records)
    aline_s, aline_d = agreement_on_the_line_scores(
        source_predictions,
        source_labels,
        target_predictions,
    )
    identifiers = [record["metadata"]["candidate_id"] for record in records]
    return (
        {identifier: float(aline_s[index]) for index, identifier in enumerate(identifiers)},
        {identifier: float(aline_d[index]) for index, identifier in enumerate(identifiers)},
    )


def agreement_on_the_line_s(records: list[dict]) -> dict[str, float]:
    return _agreement_on_the_line(records)[0]


def agreement_on_the_line_d(records: list[dict]) -> dict[str, float]:
    return _agreement_on_the_line(records)[1]


def deep_embedded_validation(records: list[dict]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for record in records:
        source = _load_source_validation(record)
        target = _load_target(record)
        errors = source["hard_predictions"].astype(np.int64, copy=False) != source["labels"].astype(
            np.int64, copy=False
        )
        identifier = record["metadata"]["candidate_id"]
        scores[identifier] = adapted_dev_score(
            source["embeddings"],
            target["embeddings"],
            errors,
            candidate_id=identifier,
        )
    return scores


def transfer_score_selector(records: list[dict]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for record in records:
        target = _load_target(record)
        identifier = record["metadata"]["candidate_id"]
        embeddings = target["embeddings"]
        probabilities = target["probabilities"]
        head = classifier_head_weight(
            record["path"] / "model_state.pt",
            class_count=probabilities.shape[1],
            embedding_dim=embeddings.shape[1],
        )
        scores[identifier] = transfer_score(
            embeddings,
            probabilities,
            head,
            candidate_id=identifier,
        )
    return scores


SELECTORS: dict[str, tuple[Callable[[list[dict]], dict[str, float]], str]] = {
    "source_val": (source_validation, "maximize"),
    "last_source_val": (last_source_validation, "maximize"),
    "entropy": (target_entropy, "minimize"),
    "infomax": (information_maximization, "maximize"),
    "agreement_reference": (agreement_reference, "maximize"),
    "global_disagreement": (global_disagreement, "minimize"),
    "gde": (generalization_disagreement_equality, "minimize"),
    "snd": (soft_neighborhood_density_selector, "maximize"),
    "aol_s": (agreement_on_the_line_s, "maximize"),
    "aol_d": (agreement_on_the_line_d, "maximize"),
    "dev": (deep_embedded_validation, "minimize"),
    "transfer_score": (transfer_score_selector, "maximize"),
}


def choose(scores: dict[str, float], direction: str) -> str:
    if not scores or not all(math.isfinite(score) for score in scores.values()):
        raise ValueError("selector produced empty or non-finite scores")
    optimum = min(scores.values()) if direction == "minimize" else max(scores.values())
    return min(candidate for candidate, score in scores.items() if score == optimum)


def build_selection_result(
    records: list[dict],
    task: str,
    selector: str,
    scores: dict[str, float],
    direction: str,
    *,
    bank_hash: str | None = None,
) -> dict:
    if selector not in SELECTORS or direction not in {"minimize", "maximize"}:
        raise ValueError("unknown selector or score direction")
    selected = choose(scores, direction)
    return {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task": task,
        "selector": selector,
        "candidate_bank_sha256": bank_hash or candidate_bank_hash(records),
        "candidate_count": len(records),
        "candidate_scores": scores,
        "score_direction": direction,
        "score_semantics": "estimated_error" if selector in {"dev", "gde"} else "ranking",
        "selected_candidate_id": selected,
        "label_access_count": 0,
        "protocol_violation_count": 0,
    }


def run_selector(candidate_root: Path, task: str, selector: str) -> dict:
    records = discover_candidate_records(candidate_root, task)
    function, direction = SELECTORS[selector]
    scores = function(records)
    return build_selection_result(records, task, selector, scores, direction)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--selector", choices=sorted(SELECTORS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_selector(args.candidate_root.resolve(), args.task, args.selector)
    atomic_json(result, args.output.resolve())
    print(json.dumps({key: value for key, value in result.items() if key != "candidate_scores"}, indent=2))


if __name__ == "__main__":
    main()
