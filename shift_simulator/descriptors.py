"""Unlabeled graph descriptors for source-to-target shift matching."""

from __future__ import annotations

import numpy as np
import torch

from spectral_filters.frame import normalized_adjacency

DESCRIPTOR_NAMES = (
    "log_nodes",
    "log_edges_per_node",
    "log_degree_mean",
    "log_degree_std",
    "log_degree_q10",
    "log_degree_q50",
    "log_degree_q90",
    "feature_global_mean",
    "feature_global_std",
    "feature_node_norm_mean",
    "feature_node_norm_std",
    "feature_channel_mean_q10",
    "feature_channel_mean_q50",
    "feature_channel_mean_q90",
    "feature_smoothness",
    "laplacian_moment_1",
    "laplacian_moment_2",
    "laplacian_moment_3",
    "laplacian_moment_4",
)


def graph_descriptor(graph, *, probe_count: int = 8, seed: int = 911) -> np.ndarray:
    """Compute a compact descriptor using no labels or target truth."""

    device = graph.x.device
    dtype = torch.float32
    node_count = graph.num_nodes
    edge_count = graph.edge_index.shape[1]
    row = graph.edge_index[0]
    degree = torch.bincount(row, minlength=node_count).to(dtype=dtype, device=device)
    log_degree = torch.log1p(degree)
    degree_quantiles = torch.quantile(log_degree, torch.tensor([0.1, 0.5, 0.9], device=device))

    x = graph.x.to(dtype)
    node_norm = torch.linalg.vector_norm(x, dim=1)
    channel_mean = x.mean(dim=0)
    channel_quantiles = torch.quantile(
        channel_mean,
        torch.tensor([0.1, 0.5, 0.9], device=device),
    )
    selected_count = min(32, x.shape[1])
    selected_indices = torch.linspace(
        0,
        x.shape[1] - 1,
        steps=selected_count,
        device=device,
    ).round().to(torch.long)
    selected_features = x[:, selected_indices]

    adjacency = normalized_adjacency(
        graph.edge_index,
        num_nodes=node_count,
        dtype=dtype,
        device=device,
        edge_weight=getattr(graph, "edge_weight", None),
    )
    propagated = torch.sparse.mm(adjacency, selected_features)
    smoothness = (
        torch.sum(selected_features * (selected_features - propagated))
        / torch.sum(selected_features * selected_features).clamp_min(1e-12)
    )

    generator = torch.Generator(device=device).manual_seed(seed)
    probes = torch.randint(
        0,
        2,
        (node_count, probe_count),
        generator=generator,
        device=device,
        dtype=torch.int64,
    ).to(dtype)
    probes = probes.mul_(2.0).sub_(1.0)
    current = probes
    moments = []
    for _ in range(4):
        current = current - torch.sparse.mm(adjacency, current)
        moments.append(torch.sum(probes * current) / (node_count * probe_count))

    values = torch.stack(
        [
            torch.log(torch.tensor(float(node_count), device=device)),
            torch.log1p(torch.tensor(edge_count / max(1, node_count), device=device)),
            log_degree.mean(),
            log_degree.std(unbiased=False),
            degree_quantiles[0],
            degree_quantiles[1],
            degree_quantiles[2],
            x.mean(),
            x.std(unbiased=False),
            node_norm.mean(),
            node_norm.std(unbiased=False),
            channel_quantiles[0],
            channel_quantiles[1],
            channel_quantiles[2],
            smoothness,
            *moments,
        ]
    )
    descriptor = values.detach().cpu().numpy().astype(np.float64, copy=False)
    if descriptor.shape != (len(DESCRIPTOR_NAMES),) or not np.isfinite(descriptor).all():
        raise FloatingPointError("graph descriptor contains invalid values")
    return descriptor


def shift_delta(source_descriptor: np.ndarray, other_descriptor: np.ndarray) -> np.ndarray:
    source = np.asarray(source_descriptor, dtype=np.float64)
    other = np.asarray(other_descriptor, dtype=np.float64)
    if source.shape != other.shape:
        raise ValueError("descriptor shapes do not match")
    scale = np.maximum(np.abs(source), 1e-3)
    return (other - source) / scale
