"""Dataset construction without exposing an oracle training entrypoint."""

from __future__ import annotations

import os.path as osp

import torch
from pygda.datasets import (
    AirportDataset,
    BlogDataset,
    CitationDataset,
    MAGDataset,
    TwitchDataset,
)
from torch_geometric.data import Data
from torch_geometric.utils import degree

from configs import resolve_domain_name


def build_dataset(domain: str, root: str):
    """Load one supported public graph-domain dataset."""

    domain = resolve_domain_name(domain)
    if domain in {"DBLPv7", "ACMv9", "Citationv1"}:
        return CitationDataset(osp.join(root, "data", "Citation", domain), domain)
    if domain in {"BRAZIL", "EUROPE", "USA"}:
        return AirportDataset(osp.join(root, "data", "Airport", domain), domain)
    if domain in {"Blog1", "Blog2"}:
        return BlogDataset(osp.join(root, "data", "Blog", domain), domain)
    if domain in {"MAG_CN", "MAG_DE", "MAG_FR", "MAG_JP", "MAG_RU", "MAG_US"}:
        return MAGDataset(osp.join(root, "data", "MAG", domain), domain)
    if domain in {"DE", "EN", "ES", "FR", "PT", "RU"}:
        return TwitchDataset(osp.join(root, "data", "Twitch", domain), domain)
    raise ValueError(f"unsupported domain: {domain}")


def ensure_features(data: Data, device: str, default_num_features: int = 241) -> Data:
    """Construct one-hot degree features when a dataset has no node features."""

    if not hasattr(data, "x") or data.x is None:
        node_degrees = degree(data.edge_index[0], num_nodes=data.num_nodes).long()
        width = max(default_num_features, int(node_degrees.max().item()) + 1)
        data.x = torch.nn.functional.one_hot(node_degrees, num_classes=width).float()
    return data.to(device)
