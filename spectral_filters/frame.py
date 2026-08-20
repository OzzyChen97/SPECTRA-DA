"""Tight graph spectral windows and sparse Chebyshev filtering."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import torch
from torch_geometric.utils import to_undirected


def resolve_centers(num_bands: int, centers: Sequence[float] | None = None) -> np.ndarray:
    if num_bands < 1:
        raise ValueError("num_bands must be positive")
    if centers is None:
        return np.linspace(0.0, 2.0, num_bands, dtype=np.float64)
    values = np.asarray(tuple(centers), dtype=np.float64)
    if values.shape != (num_bands,):
        raise ValueError("centers must contain exactly num_bands entries")
    if not np.isfinite(values).all() or np.any(values < 0.0) or np.any(values > 2.0):
        raise ValueError("spectral centers must be finite and lie in [0, 2]")
    return values


def tight_window_values(
    eigenvalues: np.ndarray,
    *,
    num_bands: int,
    sigma: float,
    centers: Sequence[float] | None = None,
) -> np.ndarray:
    """Evaluate Gaussian windows normalized to a pointwise tight frame."""

    if sigma <= 0 or not math.isfinite(sigma):
        raise ValueError("sigma must be finite and positive")
    lambdas = np.asarray(eigenvalues, dtype=np.float64)
    band_centers = resolve_centers(num_bands, centers)
    unnormalized = np.exp(
        -0.5 * ((lambdas[None, :] - band_centers[:, None]) / sigma) ** 2
    )
    denominator = np.sqrt(np.sum(unnormalized**2, axis=0, keepdims=True))
    if np.any(denominator <= np.finfo(np.float64).tiny):
        raise FloatingPointError("spectral windows underflowed; increase sigma")
    return unnormalized / denominator


def chebyshev_coefficients(
    *,
    num_bands: int,
    sigma: float,
    order: int,
    centers: Sequence[float] | None = None,
    quadrature_points: int | None = None,
) -> np.ndarray:
    """Approximate each tight window on normalized-Laplacian spectrum [0, 2]."""

    if order < 0:
        raise ValueError("Chebyshev order must be non-negative")
    points = quadrature_points or max(256, 8 * (order + 1))
    indices = np.arange(points, dtype=np.float64)
    theta = np.pi * (indices + 0.5) / points
    scaled_nodes = np.cos(theta)
    eigenvalues = scaled_nodes + 1.0
    values = tight_window_values(
        eigenvalues,
        num_bands=num_bands,
        sigma=sigma,
        centers=centers,
    )
    coefficients = np.empty((num_bands, order + 1), dtype=np.float64)
    for degree in range(order + 1):
        coefficients[:, degree] = (
            2.0
            / points
            * np.sum(values * np.cos(degree * theta)[None, :], axis=1)
        )
    coefficients[:, 0] *= 0.5
    return coefficients


def normalized_adjacency(
    edge_index: torch.Tensor,
    *,
    num_nodes: int,
    dtype: torch.dtype,
    device: torch.device,
    edge_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build symmetric D^-1/2 A D^-1/2 as a sparse COO tensor."""

    edge_index = edge_index.to(device=device, dtype=torch.long)
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.shape[1], dtype=dtype, device=device)
    else:
        edge_weight = edge_weight.to(device=device, dtype=dtype)
    edge_index, edge_weight = to_undirected(
        edge_index,
        edge_attr=edge_weight,
        num_nodes=num_nodes,
        reduce="sum",
    )
    degree = torch.zeros(num_nodes, dtype=dtype, device=device)
    degree.scatter_add_(0, edge_index[0], edge_weight)
    inverse_sqrt = degree.clamp_min(torch.finfo(dtype).tiny).pow(-0.5)
    inverse_sqrt = torch.where(degree > 0, inverse_sqrt, torch.zeros_like(inverse_sqrt))
    normalized_weight = (
        inverse_sqrt[edge_index[0]] * edge_weight * inverse_sqrt[edge_index[1]]
    )
    return torch.sparse_coo_tensor(
        edge_index,
        normalized_weight,
        size=(num_nodes, num_nodes),
        dtype=dtype,
        device=device,
    ).coalesce()


def apply_tight_frame(
    edge_index: torch.Tensor,
    signals: torch.Tensor,
    *,
    num_bands: int,
    sigma: float,
    order: int,
    centers: Sequence[float] | None = None,
    edge_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, np.ndarray]:
    """Apply all spectral windows without eigendecomposition.

    ``signals`` has shape ``[num_nodes, channels]``. The result has shape
    ``[num_bands, num_nodes, channels]``.
    """

    if signals.ndim != 2:
        raise ValueError("signals must have shape [num_nodes, channels]")
    coefficients = chebyshev_coefficients(
        num_bands=num_bands,
        sigma=sigma,
        order=order,
        centers=centers,
    )
    coefficient_tensor = torch.as_tensor(
        coefficients,
        dtype=signals.dtype,
        device=signals.device,
    )
    adjacency = normalized_adjacency(
        edge_index,
        num_nodes=signals.shape[0],
        dtype=signals.dtype,
        device=signals.device,
        edge_weight=edge_weight,
    )

    # For normalized L, the scaled operator mapping [0, 2] to [-1, 1] is
    # L - I = -D^-1/2 A D^-1/2.
    def scaled_laplacian_matmul(value: torch.Tensor) -> torch.Tensor:
        return -torch.sparse.mm(adjacency, value)

    t_previous = signals
    outputs = coefficient_tensor[:, 0, None, None] * t_previous[None, :, :]
    if order == 0:
        return outputs, coefficients
    t_current = scaled_laplacian_matmul(signals)
    outputs = outputs + coefficient_tensor[:, 1, None, None] * t_current[None, :, :]
    for degree in range(2, order + 1):
        t_next = 2.0 * scaled_laplacian_matmul(t_current) - t_previous
        outputs = outputs + coefficient_tensor[:, degree, None, None] * t_next[None, :, :]
        t_previous, t_current = t_current, t_next
    return outputs, coefficients


def frame_approximation_diagnostics(
    coefficients: np.ndarray,
    *,
    num_bands: int,
    sigma: float,
    centers: Sequence[float] | None = None,
    grid_size: int = 4097,
) -> dict[str, float]:
    """Measure scalar-window and approximate tight-frame errors on a dense grid."""

    eigenvalues = np.linspace(0.0, 2.0, grid_size, dtype=np.float64)
    scaled = eigenvalues - 1.0
    exact = tight_window_values(
        eigenvalues,
        num_bands=num_bands,
        sigma=sigma,
        centers=centers,
    )
    approximate = np.stack(
        [np.polynomial.chebyshev.chebval(scaled, band) for band in coefficients],
        axis=0,
    )
    return {
        "max_window_error": float(np.max(np.abs(approximate - exact))),
        "mean_window_error": float(np.mean(np.abs(approximate - exact))),
        "max_frame_error": float(np.max(np.abs(np.sum(approximate**2, axis=0) - 1.0))),
        "mean_frame_error": float(np.mean(np.abs(np.sum(approximate**2, axis=0) - 1.0))),
    }
