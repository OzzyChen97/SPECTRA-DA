#!/usr/bin/env python3
"""Materialize label-free public graphs and externally sealed labels.

This is a trusted setup command.  Candidate exporters and selectors consume
only ``trajectory_bank/public``.  Labels are written outside the optimization
repository under ``<workspace>/.sealed/spectra_da``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data

REPO = Path(__file__).resolve().parents[1]
WORKSPACE = REPO.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from data.loaders import build_dataset, ensure_features  # noqa: E402
from protocol.tasks import DOMAINS, domain_family  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_torch_save(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def atomic_json(value: object, path: Path, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.chmod(temporary, mode)
    temporary.replace(path)


def stratified_split(labels: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(labels.shape[0])
    train, val = train_test_split(
        indices,
        test_size=0.2,
        random_state=seed,
        shuffle=True,
        stratify=labels,
    )
    return np.sort(train), np.sort(val)


def sanitize_graph(data: Data) -> Data:
    public = Data(
        x=data.x.detach().cpu().contiguous(),
        edge_index=data.edge_index.detach().cpu().contiguous(),
        num_nodes=data.num_nodes,
    )
    if hasattr(data, "edge_weight") and data.edge_weight is not None:
        public.edge_weight = data.edge_weight.detach().cpu().contiguous()
    if "y" in public:
        raise AssertionError("public graph unexpectedly contains labels")
    return public


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-root", type=Path, default=REPO / "trajectory_bank" / "public")
    parser.add_argument("--sealed-root", type=Path, default=WORKSPACE / ".sealed" / "spectra_da")
    parser.add_argument("--split-seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    public_root = args.public_root.resolve()
    sealed_root = args.sealed_root.resolve()
    sealed_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(sealed_root, 0o700)

    public_manifest: dict[str, object] = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "split_seed": args.split_seed,
        "domains": {},
    }
    sealed_manifest: dict[str, object] = {
        "schema_version": 1,
        "created_at": public_manifest["created_at"],
        "labels": {},
    }

    for domain in DOMAINS:
        dataset = build_dataset(domain, str(REPO))
        data = ensure_features(dataset[0], "cpu")
        labels = data.y.detach().cpu().to(torch.long).contiguous()
        public = sanitize_graph(data)

        graph_path = public_root / "graphs" / f"{domain}.pt"
        split_path = public_root / "splits" / f"{domain}.npz"
        label_path = sealed_root / "labels" / f"{domain}.pt"
        atomic_torch_save(public, graph_path)

        train_idx, val_idx = stratified_split(labels.numpy(), args.split_seed)
        split_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(split_path, train_idx=train_idx, val_idx=val_idx)
        atomic_torch_save(labels, label_path)
        os.chmod(label_path, 0o600)

        public_manifest["domains"][domain] = {
            "family": domain_family(domain),
            "num_nodes": public.num_nodes,
            "num_edges": public.num_edges,
            "num_features": public.num_features,
            "graph": str(graph_path.relative_to(REPO)),
            "graph_sha256": sha256_file(graph_path),
            "split": str(split_path.relative_to(REPO)),
            "split_sha256": sha256_file(split_path),
        }
        sealed_manifest["labels"][domain] = {
            "path": str(label_path),
            "sha256": sha256_file(label_path),
            "num_labels": int(labels.numel()),
        }
        print(
            f"sealed {domain:10s} family={domain_family(domain):8s} "
            f"nodes={public.num_nodes:5d} edges={public.num_edges:6d}"
        )

    atomic_json(public_manifest, public_root / "manifest.json")
    atomic_json(sealed_manifest, sealed_root / "manifest.json", mode=0o600)
    atomic_json(
        {
            "event": "materialize",
            "created_at": public_manifest["created_at"],
            "domains": list(DOMAINS),
            "label_access_role": "trusted_setup",
        },
        sealed_root / "materialization_audit.json",
        mode=0o600,
    )
    print(f"public graphs: {public_root}")
    print(f"sealed labels: {sealed_root}")


if __name__ == "__main__":
    main()
