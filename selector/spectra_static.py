#!/usr/bin/env python3
"""SPECTRA-Static: target-label-free spectral agreement risk recovery."""

from __future__ import annotations

import argparse
import json
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

from protocol.access import load_public_graph  # noqa: E402
from protocol.tasks import TASKS  # noqa: E402
from scripts.trajectory_export.common import enforce_gpu7  # noqa: E402
from scripts.trajectory_export.schema import (  # noqa: E402
    atomic_json,
    candidate_bank_hash,
    discover_candidate_records,
)
from selector.risk_recovery import (  # noqa: E402
    pairwise_band_disagreement,
    recover_band_risks,
)
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


def load_predictions(
    records: list[dict[str, Any]],
    *,
    prediction_kind: str,
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    matrices = []
    class_count = None
    node_count = None
    for record in records:
        with np.load(record["path"] / "target_public.npz", allow_pickle=False) as artifact:
            probabilities = artifact["probabilities"].astype(np.float32, copy=False)
            if class_count is None:
                node_count, class_count = probabilities.shape
            elif probabilities.shape != (node_count, class_count):
                raise ValueError("candidate target probability shapes do not match")
            if prediction_kind == "soft":
                matrix = probabilities
            else:
                hard = artifact["hard_predictions"].astype(np.int64, copy=False)
                matrix = F.one_hot(
                    torch.from_numpy(hard),
                    num_classes=class_count,
                ).numpy().astype(np.float32, copy=False)
            matrices.append(torch.from_numpy(matrix))
    return torch.stack(matrices, dim=0).to(device), int(class_count)


def select(
    *,
    candidate_root: Path,
    task: str,
    num_bands: int,
    sigma: float,
    cheb_order: int,
    centers: tuple[float, ...] | None,
    prediction_kind: str,
    estimator: str,
    ridge: float,
    device_name: str,
) -> dict[str, Any]:
    if task not in TASK_BY_ID:
        raise KeyError(f"unknown GDA-Select task: {task}")
    if device_name.startswith("cuda"):
        enforce_gpu7(device_name)
    device = torch.device(device_name)
    started = time.perf_counter()
    records = discover_candidate_records(candidate_root, task)
    identifiers = [record["metadata"]["candidate_id"] for record in records]
    predictions, class_count = load_predictions(
        records,
        prediction_kind=prediction_kind,
        device=device,
    )
    model_count, node_count, _ = predictions.shape
    target_domain = TASK_BY_ID[task].target
    graph = load_public_graph(target_domain).to(device)
    if graph.num_nodes != node_count:
        raise ValueError("target graph and candidate prediction node counts differ")

    # Apply every polynomial recurrence once to a joint [nodes, models*classes]
    # signal matrix, then restore model and class axes.
    joint_signals = predictions.permute(1, 0, 2).reshape(node_count, model_count * class_count)
    filtered_joint, coefficients = apply_tight_frame(
        graph.edge_index,
        joint_signals,
        num_bands=num_bands,
        sigma=sigma,
        order=cheb_order,
        centers=centers,
        edge_weight=getattr(graph, "edge_weight", None),
    )
    filtered = (
        filtered_joint.reshape(num_bands, node_count, model_count, class_count)
        .permute(2, 0, 1, 3)
        .contiguous()
    )
    disagreement = pairwise_band_disagreement(filtered, node_count).cpu().numpy()
    band_risks, recovery_diagnostics = recover_band_risks(
        disagreement,
        estimator=estimator,
        ridge=ridge,
    )
    estimated_risks = 0.5 * band_risks.sum(axis=1)
    scores = {
        identifier: float(estimated_risks[index])
        for index, identifier in enumerate(identifiers)
    }
    optimum = float(estimated_risks.min())
    selected = min(identifier for identifier, score in scores.items() if score == optimum)
    frame_diagnostics = frame_approximation_diagnostics(
        coefficients,
        num_bands=num_bands,
        sigma=sigma,
        centers=centers,
    )
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task": task,
        "selector": f"spectra_static_{prediction_kind}_{estimator}",
        "candidate_bank_sha256": candidate_bank_hash(records),
        "candidate_count": model_count,
        "candidate_scores": scores,
        "score_direction": "minimize",
        "score_semantics": "estimated_error" if prediction_kind == "hard" else "ranking",
        "selected_candidate_id": selected,
        "label_access_count": 0,
        "protocol_violation_count": 0,
        "spectral_config": {
            "num_bands": num_bands,
            "centers": list(centers) if centers is not None else None,
            "sigma": sigma,
            "chebyshev_order": cheb_order,
            "prediction_kind": prediction_kind,
            "risk_estimator": estimator,
            "ridge": ridge,
            "device": device_name,
        },
        "frame_diagnostics": frame_diagnostics,
        "recovery_diagnostics": recovery_diagnostics,
        "candidate_band_risks": {
            identifier: [float(value) for value in band_risks[index]]
            for index, identifier in enumerate(identifiers)
        },
        "selector_runtime_seconds": time.perf_counter() - started,
        "target_num_nodes": node_count,
        "num_classes": class_count,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-bands", type=int, default=3)
    parser.add_argument("--centers")
    parser.add_argument("--sigma", type=float, default=0.55)
    parser.add_argument("--cheb-order", type=int, default=8)
    parser.add_argument("--prediction-kind", choices=("hard", "soft"), default="hard")
    parser.add_argument("--estimator", choices=("closed_form", "nnls"), default="nnls")
    parser.add_argument("--ridge", type=float, default=1e-6)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    centers = parse_centers(args.centers, args.num_bands)
    result = select(
        candidate_root=args.candidate_root.resolve(),
        task=args.task,
        num_bands=args.num_bands,
        sigma=args.sigma,
        cheb_order=args.cheb_order,
        centers=centers,
        prediction_kind=args.prediction_kind,
        estimator=args.estimator,
        ridge=args.ridge,
        device_name=args.device,
    )
    atomic_json(result, args.output.resolve())
    print(
        json.dumps(
            {key: value for key, value in result.items() if key not in {"candidate_scores", "candidate_band_risks"}},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
