"""Deterministic artifact-level implementation of Transfer Score."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.core.multiarray import _reconstruct
from sklearn.neighbors import NearestNeighbors


def _stable_seed(identifier: str) -> int:
    return int.from_bytes(hashlib.sha256(identifier.encode("utf-8")).digest()[:4], "little")


def _safe_checkpoint(path: Path) -> dict[str, Any]:
    safe_types = [_reconstruct, np.ndarray, np.dtype, np.dtypes.Float64DType]
    with torch.serialization.safe_globals(safe_types):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "backbone_state_dict" not in checkpoint:
        raise ValueError(f"unsupported trajectory checkpoint: {path}")
    return checkpoint


def classifier_head_weight(path: Path, class_count: int, embedding_dim: int) -> np.ndarray:
    state = _safe_checkpoint(path)["backbone_state_dict"]
    priorities = (
        "cls.lin.weight",
        "cls.weight",
        "mlp_classify.1.weight",
        "classifier.weight",
        "head.weight",
    )
    for key in priorities:
        value = state.get(key)
        if isinstance(value, torch.Tensor) and tuple(value.shape) == (class_count, embedding_dim):
            return value.detach().cpu().numpy().astype(np.float64, copy=False)
    candidates = [
        (key, value)
        for key, value in state.items()
        if isinstance(value, torch.Tensor)
        and value.ndim == 2
        and tuple(value.shape) == (class_count, embedding_dim)
        and any(token in key.lower() for token in ("cls", "class", "head"))
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"could not uniquely identify classifier head in {path}: {[key for key, _ in candidates]}"
        )
    return candidates[0][1].detach().cpu().numpy().astype(np.float64, copy=False)


def classifier_uniformity(weight: np.ndarray) -> float:
    values = np.asarray(weight, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("classifier weights must have shape [classes, dimensions]")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    normalized = values / np.clip(norms, 1e-12, None)
    cosine = np.clip(normalized @ normalized.T, -1.0, 1.0)
    first, second = np.triu_indices(values.shape[0], k=1)
    angles = np.arccos(cosine[first, second])
    simplex_angle = float(np.arccos(-1.0 / (values.shape[0] - 1)))
    return float(np.mean(np.square(angles - simplex_angle)))


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
    real_distances = neighbors.kneighbors(real_sample, n_neighbors=2, return_distance=True)[0][:, 1]
    uniform_sum = float(uniform_distances.sum())
    real_sum = float(real_distances.sum())
    denominator = uniform_sum + real_sum
    return uniform_sum / denominator if denominator > 1e-12 else 0.5


def normalized_information_maximization(probabilities: np.ndarray) -> float:
    values = np.asarray(probabilities, dtype=np.float64)
    values = np.clip(values, 1e-12, 1.0)
    conditional_entropy = float(np.mean(-np.sum(values * np.log(values), axis=1)))
    marginal = values.mean(axis=0)
    marginal_entropy = float(-np.sum(marginal * np.log(np.clip(marginal, 1e-12, 1.0))))
    return (marginal_entropy - conditional_entropy) / np.log(values.shape[1])


def transfer_score(
    embeddings: np.ndarray,
    probabilities: np.ndarray,
    classifier_weight: np.ndarray,
    *,
    candidate_id: str,
) -> float:
    """Combine representation transferability, discriminability and geometry."""

    representation = hopkins_statistic(embeddings, seed=_stable_seed(candidate_id))
    discriminability = normalized_information_maximization(probabilities)
    geometry_penalty = classifier_uniformity(classifier_weight)
    return float(representation + discriminability - geometry_penalty)
