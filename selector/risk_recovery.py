"""Band-wise disagreement computation and individual-risk recovery."""

from __future__ import annotations

import itertools
from functools import lru_cache
from typing import Any

import numpy as np
import torch
from scipy import sparse
from scipy.optimize import lsq_linear


def pairwise_band_disagreement(filtered: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Return ``[bands, models, models]`` squared spectral disagreement."""

    if filtered.ndim != 4:
        raise ValueError("filtered predictions must have shape [models, bands, nodes, classes]")
    model_count, band_count, observed_nodes, _ = filtered.shape
    if observed_nodes != num_nodes:
        raise ValueError("filtered prediction node count mismatch")
    matrices = []
    for band in range(band_count):
        flattened = filtered[:, band].reshape(model_count, -1)
        norms = torch.sum(flattened * flattened, dim=1)
        distances = (
            norms[:, None]
            + norms[None, :]
            - 2.0 * torch.matmul(flattened, flattened.T)
        ) / num_nodes
        matrices.append(distances.clamp_min(0.0))
    return torch.stack(matrices, dim=0)


def closed_form_recovery(disagreement: np.ndarray) -> np.ndarray:
    """Recover non-negative risks under zero pairwise error covariance."""

    model_count = disagreement.shape[0]
    if disagreement.shape != (model_count, model_count):
        raise ValueError("disagreement must be square")
    if model_count < 3:
        raise ValueError("risk recovery requires at least three candidates")
    row_sums = disagreement.sum(axis=1)
    upper_sum = np.triu(disagreement, k=1).sum()
    risks = (row_sums - upper_sum / (model_count - 1)) / (model_count - 2)
    return np.maximum(risks, 0.0)


@lru_cache(maxsize=None)
def pair_design(model_count: int) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray]:
    """Return the immutable complete-pair design, cached by committee size."""

    if model_count < 2:
        raise ValueError("pair design requires at least two candidates")
    pairs = np.asarray(list(itertools.combinations(range(model_count), 2)), dtype=np.int64)
    rows = np.repeat(np.arange(pairs.shape[0]), 2)
    cols = pairs.reshape(-1)
    values = np.ones(rows.shape[0], dtype=np.float64)
    design = sparse.coo_matrix(
        (values, (rows, cols)),
        shape=(pairs.shape[0], model_count),
    ).tocsr()
    pairs.setflags(write=False)
    return design, pairs[:, 0], pairs[:, 1]


def nnls_recovery(disagreement: np.ndarray, ridge: float = 0.0) -> np.ndarray:
    model_count = disagreement.shape[0]
    if model_count < 3:
        raise ValueError("risk recovery requires at least three candidates")
    if ridge < 0:
        raise ValueError("ridge must be non-negative")
    design, first, second = pair_design(model_count)
    observations = disagreement[first, second].astype(np.float64, copy=False)
    if ridge > 0:
        design = sparse.vstack(
            [design, math_sqrt(ridge) * sparse.eye(model_count, format="csr")],
            format="csr",
        )
        observations = np.concatenate([observations, np.zeros(model_count, dtype=np.float64)])
    result = lsq_linear(design, observations, bounds=(0.0, np.inf), lsmr_tol="auto")
    if not result.success:
        raise RuntimeError(f"NNLS risk recovery failed: {result.message}")
    return result.x


def math_sqrt(value: float) -> float:
    return float(np.sqrt(value))


def recover_band_risks(
    disagreements: np.ndarray,
    *,
    estimator: str,
    ridge: float = 0.0,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Recover a risk vector for each band and report pair-fit residuals."""

    if disagreements.ndim != 3 or disagreements.shape[1] != disagreements.shape[2]:
        raise ValueError("disagreements must have shape [bands, models, models]")
    band_risks = []
    diagnostics = []
    model_count = disagreements.shape[1]
    _, first, second = pair_design(model_count)
    for band, matrix in enumerate(disagreements):
        if estimator == "closed_form":
            risks = closed_form_recovery(matrix)
        elif estimator == "nnls":
            risks = nnls_recovery(matrix, ridge=ridge)
        else:
            raise ValueError(f"unknown risk estimator: {estimator}")
        observed = matrix[first, second]
        fitted = risks[first] + risks[second]
        residual = observed - fitted
        diagnostics.append(
            {
                "band": band,
                "pair_rmse": float(np.sqrt(np.mean(residual**2))),
                "pair_mae": float(np.mean(np.abs(residual))),
                "negative_fraction_before_constraint": float(
                    np.mean(closed_form_recovery_unclipped(matrix) < 0.0)
                ),
            }
        )
        band_risks.append(risks)
    return np.stack(band_risks, axis=1), diagnostics


def closed_form_recovery_unclipped(disagreement: np.ndarray) -> np.ndarray:
    model_count = disagreement.shape[0]
    row_sums = disagreement.sum(axis=1)
    upper_sum = np.triu(disagreement, k=1).sum()
    return (row_sums - upper_sum / (model_count - 1)) / (model_count - 2)
