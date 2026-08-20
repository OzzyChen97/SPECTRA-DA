#!/usr/bin/env python3
"""Build source-simulated band-risk and covariance calibration artifacts."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from protocol.access import load_training_task  # noqa: E402
from protocol.tasks import TASKS  # noqa: E402
from scripts.trajectory_export.common import enforce_gpu7  # noqa: E402
from scripts.trajectory_export.schema import (  # noqa: E402
    atomic_json,
    atomic_npz,
    candidate_bank_hash,
    canonical_hash,
    discover_candidate_records,
    sha256_file,
)
from shift_simulator.descriptors import (  # noqa: E402
    DESCRIPTOR_NAMES,
    graph_descriptor,
    shift_delta,
)
from shift_simulator.model_inference import prepare_candidate  # noqa: E402
from shift_simulator.transforms import apply_shift, default_shift_specs  # noqa: E402
from spectral_filters import (  # noqa: E402
    apply_tight_frame,
    frame_approximation_diagnostics,
)

TASK_BY_ID = {task.id: task for task in TASKS}


def parse_centers(value: str | None, num_bands: int) -> tuple[float, ...] | None:
    if value is None:
        return None
    centers = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if len(centers) != num_bands:
        raise ValueError("--centers must contain exactly --num-bands values")
    return centers


def spectral_error_statistics(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    graph,
    *,
    num_classes: int,
    num_bands: int,
    sigma: float,
    cheb_order: int,
    centers: tuple[float, ...] | None,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model_count, node_count = predictions.shape
    prediction_onehot = F.one_hot(
        predictions.to(device=device, dtype=torch.long),
        num_classes=num_classes,
    ).to(torch.float32)
    label_onehot = F.one_hot(
        labels.to(device=device, dtype=torch.long),
        num_classes=num_classes,
    ).to(torch.float32)
    errors = prediction_onehot - label_onehot[None, :, :]
    joint = errors.permute(1, 0, 2).reshape(node_count, model_count * num_classes)
    filtered_joint, coefficients = apply_tight_frame(
        graph.edge_index.to(device),
        joint,
        num_bands=num_bands,
        sigma=sigma,
        order=cheb_order,
        centers=centers,
        edge_weight=getattr(graph, "edge_weight", None),
    )
    filtered = (
        filtered_joint.reshape(num_bands, node_count, model_count, num_classes)
        .permute(2, 0, 1, 3)
        .contiguous()
    )
    risks = torch.empty((model_count, num_bands), dtype=torch.float32, device=device)
    covariances = torch.empty(
        (num_bands, model_count, model_count),
        dtype=torch.float32,
        device=device,
    )
    for band in range(num_bands):
        flattened = filtered[:, band].reshape(model_count, -1)
        covariance = flattened @ flattened.T / node_count
        covariances[band] = covariance
        risks[:, band] = torch.diagonal(covariance)
    return (
        risks.cpu().numpy(),
        covariances.cpu().numpy(),
        coefficients,
    )


def build(args: argparse.Namespace) -> dict[str, Any]:
    if args.task not in TASK_BY_ID:
        raise KeyError(f"unknown GDA-Select task: {args.task}")
    if args.device.startswith("cuda"):
        enforce_gpu7(args.device)
    device = torch.device(args.device)
    task = TASK_BY_ID[args.task]
    records = discover_candidate_records(args.candidate_root.resolve(), args.task)
    bank_hash = candidate_bank_hash(records)
    candidate_ids = [record["metadata"]["candidate_id"] for record in records]
    centers = parse_centers(args.centers, args.num_bands)
    spectral_config = {
        "num_bands": args.num_bands,
        "sigma": args.sigma,
        "chebyshev_order": args.cheb_order,
        "centers": list(centers) if centers is not None else None,
    }
    calibration_id = canonical_hash(
        {
            "candidate_bank_sha256": bank_hash,
            "spectral_config": spectral_config,
            "shift_seed": args.shift_seed,
        }
    )
    final_directory = args.output_root.resolve() / args.task / calibration_id
    if final_directory.exists():
        raise FileExistsError(f"immutable calibration artifact already exists: {final_directory}")
    final_directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = final_directory.with_name(f".{final_directory.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"temporary calibration directory exists: {temporary}")
    temporary.mkdir()

    started = time.perf_counter()
    source_graph, target_graph = load_training_task(task.source, task.target, device=args.device)
    source_labels = source_graph.y.detach().clone()
    source_unlabeled = source_graph.clone()
    del source_unlabeled.y
    for field in ("train_mask", "val_mask"):
        if field in source_unlabeled:
            del source_unlabeled[field]
    if "y" in target_graph:
        raise RuntimeError("protocol violation: target graph contains labels")
    num_classes = int(source_labels.max().item()) + 1
    prepared_candidates = [
        prepare_candidate(
            record,
            in_dim=int(source_unlabeled.x.shape[1]),
            num_classes=num_classes,
            device=device,
        )
        for record in records
    ]

    source_descriptor = graph_descriptor(source_unlabeled)
    target_descriptor = graph_descriptor(target_graph)
    target_delta = shift_delta(source_descriptor, target_descriptor)
    specs = default_shift_specs(args.shift_seed)
    shift_descriptors = []
    shift_deltas = []
    all_risks = []
    all_covariances = []
    coefficients = None

    for shift_index, spec in enumerate(specs):
        shifted_graph, shifted_labels = apply_shift(source_unlabeled, source_labels, spec)
        descriptor = graph_descriptor(shifted_graph)
        predictions = []
        for candidate_index, prepared in enumerate(prepared_candidates):
            inference_seed = spec.seed * 1_000_003 + candidate_index
            logits, _ = prepared.infer(
                shifted_graph,
                inference_seed=inference_seed,
            )
            predictions.append(logits.argmax(dim=1))
        prediction_matrix = torch.stack(predictions, dim=0)
        risks, covariances, coefficients = spectral_error_statistics(
            prediction_matrix,
            shifted_labels,
            shifted_graph,
            num_classes=num_classes,
            num_bands=args.num_bands,
            sigma=args.sigma,
            cheb_order=args.cheb_order,
            centers=centers,
            device=device,
        )
        shift_descriptors.append(descriptor)
        shift_deltas.append(shift_delta(source_descriptor, descriptor))
        all_risks.append(risks)
        all_covariances.append(covariances)
        print(
            f"shift={shift_index + 1:02d}/{len(specs):02d} name={spec.name} "
            f"nodes={shifted_graph.num_nodes} edges={shifted_graph.num_edges}"
        )

    arrays_path = temporary / "calibration.npz"
    atomic_npz(
        arrays_path,
        source_descriptor=source_descriptor,
        target_descriptor=target_descriptor,
        target_delta=target_delta,
        shift_descriptors=np.stack(shift_descriptors),
        shift_deltas=np.stack(shift_deltas),
        band_risks=np.stack(all_risks).astype(np.float32, copy=False),
        band_covariances=np.stack(all_covariances).astype(np.float32, copy=False),
        chebyshev_coefficients=np.asarray(coefficients),
    )
    frame_diagnostics = frame_approximation_diagnostics(
        np.asarray(coefficients),
        num_bands=args.num_bands,
        sigma=args.sigma,
        centers=centers,
    )
    metadata = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task": args.task,
        "source": task.source,
        "target": task.target,
        "calibration_id": calibration_id,
        "candidate_bank_sha256": bank_hash,
        "candidate_ids": candidate_ids,
        "candidate_count": len(candidate_ids),
        "shift_count": len(specs),
        "shift_specs": [spec.to_dict() for spec in specs],
        "descriptor_names": list(DESCRIPTOR_NAMES),
        "spectral_config": spectral_config,
        "frame_diagnostics": frame_diagnostics,
        "target_label_access_count": 0,
        "source_label_access_count": 1,
        "logical_device": args.device,
        "physical_gpu": 7 if args.device.startswith("cuda") else None,
        "runtime_seconds": time.perf_counter() - started,
        "artifact_sha256": {arrays_path.name: sha256_file(arrays_path)},
    }
    atomic_json(metadata, temporary / "metadata.json")
    temporary.replace(final_directory)
    return {"path": str(final_directory), "metadata": metadata}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO / "trajectory_bank" / "calibration",
    )
    parser.add_argument("--num-bands", type=int, default=3)
    parser.add_argument("--centers")
    parser.add_argument("--sigma", type=float, default=0.55)
    parser.add_argument("--cheb-order", type=int, default=8)
    parser.add_argument("--shift-seed", type=int, default=7400)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


if __name__ == "__main__":
    result = build(parse_args())
    print(json.dumps(result["metadata"], indent=2))
