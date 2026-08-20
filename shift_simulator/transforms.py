"""Deterministic source-simulated feature, structure, homophily, and label shifts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch_geometric.data import Data
from torch_geometric.utils import subgraph


@dataclass(frozen=True)
class ShiftSpec:
    name: str
    family: str
    parameters: dict[str, Any]
    seed: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def default_shift_specs(base_seed: int = 7400) -> tuple[ShiftSpec, ...]:
    definitions = (
        ("feature_mask_015", "feature", {"kind": "feature_mask", "rate": 0.15}),
        ("feature_mask_035", "feature", {"kind": "feature_mask", "rate": 0.35}),
        ("feature_noise_010", "feature", {"kind": "feature_noise", "scale": 0.10}),
        ("feature_noise_030", "feature", {"kind": "feature_noise", "scale": 0.30}),
        ("edge_dropout_015", "structure", {"kind": "edge_dropout", "rate": 0.15}),
        ("edge_dropout_035", "structure", {"kind": "edge_dropout", "rate": 0.35}),
        (
            "homophily_lower",
            "homophily",
            {"kind": "homophily_dropout", "same_keep": 0.50, "different_keep": 1.0},
        ),
        (
            "homophily_higher",
            "homophily",
            {"kind": "homophily_dropout", "same_keep": 1.0, "different_keep": 0.50},
        ),
        (
            "label_prior_majority",
            "label_prior",
            {"kind": "label_prior", "class_policy": "majority", "class_keep": 0.40},
        ),
        (
            "label_prior_minority",
            "label_prior",
            {"kind": "label_prior", "class_policy": "minority", "class_keep": 0.40},
        ),
        (
            "conditional_structure",
            "conditional_structure",
            {"kind": "conditional_dropout", "pair_policy": "most_common", "pair_keep": 0.35},
        ),
    )
    return tuple(
        ShiftSpec(name=name, family=family, parameters=parameters, seed=base_seed + index)
        for index, (name, family, parameters) in enumerate(definitions)
    )


def _copy_graph(graph: Data, *, x: torch.Tensor, edge_index: torch.Tensor) -> Data:
    shifted = Data(x=x, edge_index=edge_index, num_nodes=x.shape[0])
    return shifted


def _generator(seed: int, device: torch.device) -> torch.Generator:
    return torch.Generator(device=device).manual_seed(seed)


def _feature_mask(graph: Data, rate: float, seed: int) -> tuple[Data, torch.Tensor | None]:
    if not 0.0 < rate < 1.0:
        raise ValueError("feature mask rate must lie in (0, 1)")
    x = graph.x.clone()
    generator = _generator(seed, x.device)
    keep = torch.rand(x.shape[1], generator=generator, device=x.device) >= rate
    if not bool(keep.any()):
        keep[seed % x.shape[1]] = True
    x[:, ~keep] = 0
    return _copy_graph(graph, x=x, edge_index=graph.edge_index.clone()), None


def _feature_noise(graph: Data, scale: float, seed: int) -> tuple[Data, torch.Tensor | None]:
    if scale <= 0:
        raise ValueError("feature noise scale must be positive")
    x = graph.x
    generator = _generator(seed, x.device)
    global_scale = x.float().std().clamp_min(1e-6)
    noise = torch.randn(x.shape, generator=generator, dtype=x.dtype, device=x.device)
    shifted_x = x + scale * global_scale.to(x.dtype) * noise
    return _copy_graph(graph, x=shifted_x, edge_index=graph.edge_index.clone()), None


def _edge_subset(graph: Data, keep: torch.Tensor) -> Data:
    if keep.ndim != 1 or keep.shape[0] != graph.edge_index.shape[1]:
        raise ValueError("edge keep mask shape mismatch")
    if not bool(keep.any()):
        keep[0] = True
    shifted = _copy_graph(graph, x=graph.x.clone(), edge_index=graph.edge_index[:, keep].clone())
    return shifted


def _edge_dropout(graph: Data, rate: float, seed: int) -> tuple[Data, torch.Tensor | None]:
    generator = _generator(seed, graph.edge_index.device)
    keep = torch.rand(graph.edge_index.shape[1], generator=generator, device=graph.edge_index.device) >= rate
    return _edge_subset(graph, keep), None


def _homophily_dropout(
    graph: Data,
    labels: torch.Tensor,
    same_keep: float,
    different_keep: float,
    seed: int,
) -> tuple[Data, torch.Tensor | None]:
    row, col = graph.edge_index
    same = labels[row] == labels[col]
    probability = torch.where(
        same,
        torch.full_like(same, same_keep, dtype=torch.float32),
        torch.full_like(same, different_keep, dtype=torch.float32),
    )
    generator = _generator(seed, graph.edge_index.device)
    keep = torch.rand(probability.shape, generator=generator, device=probability.device) < probability
    return _edge_subset(graph, keep), None


def _label_prior(
    graph: Data,
    labels: torch.Tensor,
    class_policy: str,
    class_keep: float,
    seed: int,
) -> tuple[Data, torch.Tensor]:
    class_count = int(labels.max().item()) + 1
    counts = torch.bincount(labels, minlength=class_count)
    selected_class = int(counts.argmax().item()) if class_policy == "majority" else int(counts.argmin().item())
    probability = torch.ones(labels.shape[0], dtype=torch.float32, device=labels.device)
    probability[labels == selected_class] = class_keep
    generator = _generator(seed, labels.device)
    keep = torch.rand(probability.shape, generator=generator, device=labels.device) < probability
    for class_index in range(class_count):
        if not bool((keep & (labels == class_index)).any()):
            first = torch.nonzero(labels == class_index, as_tuple=False).flatten()[0]
            keep[first] = True
    node_indices = torch.nonzero(keep, as_tuple=False).flatten()
    shifted_edges, _ = subgraph(
        node_indices,
        graph.edge_index,
        relabel_nodes=True,
        num_nodes=graph.num_nodes,
    )
    shifted = _copy_graph(graph, x=graph.x[node_indices].clone(), edge_index=shifted_edges)
    return shifted, labels[node_indices].clone()


def _conditional_dropout(
    graph: Data,
    labels: torch.Tensor,
    pair_policy: str,
    pair_keep: float,
    seed: int,
) -> tuple[Data, torch.Tensor | None]:
    class_count = int(labels.max().item()) + 1
    row, col = graph.edge_index
    pair_index = labels[row] * class_count + labels[col]
    pair_counts = torch.bincount(pair_index, minlength=class_count * class_count)
    if pair_policy != "most_common":
        raise ValueError(f"unknown conditional pair policy: {pair_policy}")
    selected_pair = int(pair_counts.argmax().item())
    probability = torch.ones(graph.edge_index.shape[1], dtype=torch.float32, device=labels.device)
    probability[pair_index == selected_pair] = pair_keep
    generator = _generator(seed, labels.device)
    keep = torch.rand(probability.shape, generator=generator, device=labels.device) < probability
    return _edge_subset(graph, keep), None


def apply_shift(graph: Data, labels: torch.Tensor, spec: ShiftSpec) -> tuple[Data, torch.Tensor]:
    """Apply one shift without mutating the source graph or labels."""

    kind = spec.parameters["kind"]
    if kind == "feature_mask":
        shifted, shifted_labels = _feature_mask(graph, spec.parameters["rate"], spec.seed)
    elif kind == "feature_noise":
        shifted, shifted_labels = _feature_noise(graph, spec.parameters["scale"], spec.seed)
    elif kind == "edge_dropout":
        shifted, shifted_labels = _edge_dropout(graph, spec.parameters["rate"], spec.seed)
    elif kind == "homophily_dropout":
        shifted, shifted_labels = _homophily_dropout(
            graph,
            labels,
            spec.parameters["same_keep"],
            spec.parameters["different_keep"],
            spec.seed,
        )
    elif kind == "label_prior":
        shifted, shifted_labels = _label_prior(
            graph,
            labels,
            spec.parameters["class_policy"],
            spec.parameters["class_keep"],
            spec.seed,
        )
    elif kind == "conditional_dropout":
        shifted, shifted_labels = _conditional_dropout(
            graph,
            labels,
            spec.parameters["pair_policy"],
            spec.parameters["pair_keep"],
            spec.seed,
        )
    else:
        raise ValueError(f"unknown source-simulated shift: {kind}")
    return shifted, labels.clone() if shifted_labels is None else shifted_labels
