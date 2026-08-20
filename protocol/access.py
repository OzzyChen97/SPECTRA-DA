"""Task-scoped access API used by trusted trajectory exporters."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

from .tasks import DOMAINS, task_id

REPO = Path(__file__).resolve().parents[1]
WORKSPACE = REPO.parents[1]
PUBLIC_ROOT = Path(
    os.environ.get("SPECTRA_PUBLIC_ROOT", REPO / "trajectory_bank" / "public")
).resolve()
SEALED_ROOT = WORKSPACE / ".sealed" / "spectra_da"


def _append_audit(event: dict[str, object]) -> None:
    audit_path = SEALED_ROOT / "access_audit.jsonl"
    audit_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    event = {"timestamp": datetime.now(timezone.utc).isoformat(), **event}
    line = json.dumps(event, sort_keys=True) + "\n"
    descriptor = os.open(audit_path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        os.write(descriptor, line.encode("utf-8"))
    finally:
        os.close(descriptor)


def load_public_graph(domain: str) -> Data:
    if domain not in DOMAINS:
        raise KeyError(f"unknown domain: {domain}")
    path = PUBLIC_ROOT / "graphs" / f"{domain}.pt"
    graph = torch.load(path, weights_only=False)
    if "y" in graph:
        raise RuntimeError(f"protocol violation: public graph {domain} contains y")
    return graph


def _load_source_labels(source: str, target: str) -> torch.Tensor:
    if source == target:
        raise ValueError("source and target must differ")
    path = SEALED_ROOT / "labels" / f"{source}.pt"
    labels = torch.load(path, weights_only=True)
    _append_audit(
        {
            "event": "label_read",
            "role": "source",
            "task": task_id(source, target),
            "domain": source,
            "status": "allowed",
        }
    )
    return labels


def load_training_task(source: str, target: str, device: str = "cpu") -> tuple[Data, Data]:
    """Return a labeled source graph and a target graph with no label field."""

    source_graph = load_public_graph(source)
    target_graph = load_public_graph(target)
    source_graph.y = _load_source_labels(source, target)

    split = np.load(PUBLIC_ROOT / "splits" / f"{source}.npz")
    source_graph.train_mask = torch.zeros(source_graph.num_nodes, dtype=torch.bool)
    source_graph.val_mask = torch.zeros(source_graph.num_nodes, dtype=torch.bool)
    source_graph.train_mask[torch.from_numpy(split["train_idx"])] = True
    source_graph.val_mask[torch.from_numpy(split["val_idx"])] = True

    if "y" in target_graph:
        _append_audit(
            {
                "event": "protocol_violation",
                "role": "target",
                "task": task_id(source, target),
                "domain": target,
                "status": "blocked",
            }
        )
        raise RuntimeError("target graph contains labels")

    return source_graph.to(device), target_graph.to(device)
