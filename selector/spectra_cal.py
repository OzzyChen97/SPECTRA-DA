#!/usr/bin/env python3
"""SPECTRA-Cal/Robust with shift-conditioned covariance transport."""

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

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from covariance_transport import (  # noqa: E402
    corrected_band_risk_recovery,
    match_shift_convex_combination,
    transport_statistics,
)
from protocol.access import load_public_graph  # noqa: E402
from protocol.tasks import TASKS  # noqa: E402
from scripts.trajectory_export.common import enforce_gpu7  # noqa: E402
from scripts.trajectory_export.schema import (  # noqa: E402
    atomic_json,
    candidate_bank_hash,
    discover_candidate_records,
    sha256_file,
)
from selector.risk_recovery import pairwise_band_disagreement  # noqa: E402
from selector.spectra_static import load_predictions  # noqa: E402
from spectral_filters import apply_tight_frame, frame_approximation_diagnostics  # noqa: E402

TASK_BY_ID = {task.id: task for task in TASKS}
PRIOR_DESCRIPTOR_RMSE_THRESHOLD = 2.0


def curvature_prior_strength(
    disagreements: np.ndarray,
    descriptor_rmse: float,
    *,
    threshold: float = PRIOR_DESCRIPTOR_RMSE_THRESHOLD,
) -> float:
    """Return a label-free, support-aware transported-risk prior strength."""

    values = np.asarray(disagreements)
    if values.ndim != 3 or values.shape[1] != values.shape[2]:
        raise ValueError("disagreements must have shape [bands, models, models]")
    if not np.isfinite(descriptor_rmse) or descriptor_rmse < 0:
        raise ValueError("descriptor RMSE must be finite and non-negative")
    if threshold <= 0:
        raise ValueError("descriptor RMSE threshold must be positive")
    model_count = values.shape[1]
    return float(max(1, model_count - 2)) if descriptor_rmse <= threshold else 0.0


def load_calibration(directory: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    metadata_path = directory / "metadata.json"
    arrays_path = directory / "calibration.npz"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_hash = metadata["artifact_sha256"][arrays_path.name]
    if sha256_file(arrays_path) != expected_hash:
        raise ValueError("calibration artifact hash mismatch")
    with np.load(arrays_path, allow_pickle=False) as artifact:
        arrays = {key: artifact[key] for key in artifact.files}
    return metadata, arrays


def target_disagreement(
    records: list[dict[str, Any]],
    task: str,
    spectral_config: dict[str, Any],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    predictions, class_count = load_predictions(
        records,
        prediction_kind="hard",
        device=device,
    )
    model_count, node_count, _ = predictions.shape
    graph = load_public_graph(TASK_BY_ID[task].target).to(device)
    centers_value = spectral_config.get("centers")
    centers = tuple(centers_value) if centers_value is not None else None
    joint = predictions.permute(1, 0, 2).reshape(node_count, model_count * class_count)
    filtered_joint, coefficients = apply_tight_frame(
        graph.edge_index,
        joint,
        num_bands=int(spectral_config["num_bands"]),
        sigma=float(spectral_config["sigma"]),
        order=int(spectral_config["chebyshev_order"]),
        centers=centers,
        edge_weight=getattr(graph, "edge_weight", None),
    )
    band_count = int(spectral_config["num_bands"])
    filtered = (
        filtered_joint.reshape(band_count, node_count, model_count, class_count)
        .permute(2, 0, 1, 3)
        .contiguous()
    )
    disagreement = pairwise_band_disagreement(filtered, node_count).cpu().numpy()
    frame_diagnostics = frame_approximation_diagnostics(
        coefficients,
        num_bands=band_count,
        sigma=float(spectral_config["sigma"]),
        centers=centers,
    )
    return disagreement, coefficients, frame_diagnostics


def recover_with_transport(
    *,
    shift_deltas: np.ndarray,
    target_delta: np.ndarray,
    shift_band_risks: np.ndarray,
    shift_band_covariances: np.ndarray,
    disagreements: np.ndarray,
    transport_regularization: float,
    descriptor_floor: float,
    risk_ridge: float,
    pair_weight_power: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    alpha, match_diagnostics = match_shift_convex_combination(
        shift_deltas,
        target_delta,
        regularization=transport_regularization,
        descriptor_floor=descriptor_floor,
    )
    transported_risks, transported_covariances = transport_statistics(
        alpha,
        shift_band_risks,
        shift_band_covariances,
    )
    recovered, recovery_diagnostics = corrected_band_risk_recovery(
        disagreements,
        transported_risks,
        transported_covariances,
        ridge=risk_ridge,
        pair_weight_power=pair_weight_power,
        # Normalize by the pair design's weakest identifiable curvature.  A
        # two-standard-deviation descriptor gate disables the prior when the
        # target lies outside the source-simulation support; this is fully
        # label-free at deployment time.
        prior_strength=curvature_prior_strength(
            disagreements,
            float(match_diagnostics["descriptor_rmse"]),
        ),
    )
    return recovered, alpha, {
        "matching": match_diagnostics,
        "recovery": recovery_diagnostics,
    }


def collapse_to_global(
    arrays: dict[str, np.ndarray],
    disagreements: np.ndarray,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Collapse a tight-frame calibration to one global energy channel.

    This is the attribution-control baseline for SPECTRA-Cal.  It receives the
    same simulated shifts, descriptors, transported risks, covariances, and
    target predictions as the banded method, but removes band conditioning by
    summing all tight-frame energies before recovery.
    """

    band_risks = np.asarray(arrays["band_risks"])
    band_covariances = np.asarray(arrays["band_covariances"])
    observed = np.asarray(disagreements)
    if band_risks.ndim != 3:
        raise ValueError("band risks must have shape [shifts, models, bands]")
    shift_count, model_count, band_count = band_risks.shape
    if band_covariances.shape != (
        shift_count,
        band_count,
        model_count,
        model_count,
    ):
        raise ValueError("band covariance shape mismatch")
    if observed.shape != (band_count, model_count, model_count):
        raise ValueError("target disagreement shape mismatch")

    collapsed = dict(arrays)
    collapsed["band_risks"] = band_risks.sum(axis=2, keepdims=True)
    collapsed["band_covariances"] = band_covariances.sum(axis=1, keepdims=True)
    global_disagreements = observed.sum(axis=0, keepdims=True)
    return collapsed, global_disagreements


def bootstrap_uncertainty(
    *,
    arrays: dict[str, np.ndarray],
    disagreements: np.ndarray,
    samples: int,
    seed: int,
    transport_regularization: float,
    descriptor_floor: float,
    risk_ridge: float,
    pair_weight_power: float,
) -> np.ndarray:
    if samples <= 1:
        return np.zeros(disagreements.shape[1], dtype=np.float64)
    generator = np.random.default_rng(seed)
    shift_count = arrays["shift_deltas"].shape[0]
    estimates = []
    for _ in range(samples):
        indices = generator.integers(0, shift_count, size=shift_count)
        recovered, _, _ = recover_with_transport(
            shift_deltas=arrays["shift_deltas"][indices],
            target_delta=arrays["target_delta"],
            shift_band_risks=arrays["band_risks"][indices],
            shift_band_covariances=arrays["band_covariances"][indices],
            disagreements=disagreements,
            transport_regularization=transport_regularization,
            descriptor_floor=descriptor_floor,
            risk_ridge=risk_ridge,
            pair_weight_power=pair_weight_power,
        )
        estimates.append(0.5 * recovered.sum(axis=1))
    return np.std(np.stack(estimates), axis=0, ddof=1)


def select(args: argparse.Namespace) -> dict[str, Any]:
    if args.task not in TASK_BY_ID:
        raise KeyError(f"unknown GDA-Select task: {args.task}")
    if args.device.startswith("cuda"):
        enforce_gpu7(args.device)
    device = torch.device(args.device)
    started = time.perf_counter()
    records = discover_candidate_records(args.candidate_root.resolve(), args.task)
    bank_hash = candidate_bank_hash(records)
    identifiers = [record["metadata"]["candidate_id"] for record in records]
    calibration_metadata, arrays = load_calibration(args.calibration_dir.resolve())
    if calibration_metadata["task"] != args.task:
        raise ValueError("calibration task mismatch")
    if calibration_metadata["candidate_bank_sha256"] != bank_hash:
        raise ValueError("calibration candidate-bank hash mismatch")
    if calibration_metadata["candidate_ids"] != identifiers:
        raise ValueError("calibration candidate ordering mismatch")

    spectral_config = calibration_metadata["spectral_config"]
    disagreements, _, frame_diagnostics = target_disagreement(
        records,
        args.task,
        spectral_config,
        device,
    )
    spectral_mode = getattr(args, "spectral_mode", "banded")
    if spectral_mode == "global":
        arrays, disagreements = collapse_to_global(arrays, disagreements)
    elif spectral_mode != "banded":
        raise ValueError(f"unknown spectral mode: {spectral_mode}")
    recovered, alpha, diagnostics = recover_with_transport(
        shift_deltas=arrays["shift_deltas"],
        target_delta=arrays["target_delta"],
        shift_band_risks=arrays["band_risks"],
        shift_band_covariances=arrays["band_covariances"],
        disagreements=disagreements,
        transport_regularization=args.transport_regularization,
        descriptor_floor=args.descriptor_floor,
        risk_ridge=args.risk_ridge,
        pair_weight_power=args.pair_weight_power,
    )
    point_estimate = 0.5 * recovered.sum(axis=1)
    uncertainty = bootstrap_uncertainty(
        arrays=arrays,
        disagreements=disagreements,
        samples=args.bootstrap_samples,
        seed=args.bootstrap_seed,
        transport_regularization=args.transport_regularization,
        descriptor_floor=args.descriptor_floor,
        risk_ridge=args.risk_ridge,
        pair_weight_power=args.pair_weight_power,
    )
    robust_score = point_estimate + args.uncertainty_beta * uncertainty
    scores = {
        identifier: float(robust_score[index])
        for index, identifier in enumerate(identifiers)
    }
    optimum = float(robust_score.min())
    selected = min(identifier for identifier, score in scores.items() if score == optimum)
    shift_names = [spec["name"] for spec in calibration_metadata["shift_specs"]]
    robust = args.bootstrap_samples > 1 and args.uncertainty_beta > 0
    if spectral_mode == "global":
        selector_name = "spectra_global_robust" if robust else "spectra_global_cal"
    else:
        selector_name = "spectra_robust" if robust else "spectra_cal"
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task": args.task,
        "selector": selector_name,
        "candidate_bank_sha256": bank_hash,
        "candidate_count": len(records),
        "candidate_scores": scores,
        "score_direction": "minimize",
        "score_semantics": "estimated_error",
        "selected_candidate_id": selected,
        "label_access_count": 0,
        "protocol_violation_count": 0,
        "spectral_config": spectral_config,
        "spectral_mode": spectral_mode,
        "transport_config": {
            "transport_regularization": args.transport_regularization,
            "descriptor_floor": args.descriptor_floor,
            "risk_ridge": args.risk_ridge,
            "pair_weight_power": args.pair_weight_power,
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_seed": args.bootstrap_seed,
            "uncertainty_beta": args.uncertainty_beta,
            "device": args.device,
        },
        "frame_diagnostics": frame_diagnostics,
        "transport_diagnostics": diagnostics,
        "shift_weights": {
            name: float(alpha[index]) for index, name in enumerate(shift_names)
        },
        "candidate_point_estimates": {
            identifier: float(point_estimate[index])
            for index, identifier in enumerate(identifiers)
        },
        "candidate_uncertainty": {
            identifier: float(uncertainty[index])
            for index, identifier in enumerate(identifiers)
        },
        "candidate_band_risks": {
            identifier: [float(value) for value in recovered[index]]
            for index, identifier in enumerate(identifiers)
        },
        "selector_runtime_seconds": time.perf_counter() - started,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--transport-regularization", type=float, default=0.1)
    parser.add_argument("--descriptor-floor", type=float, default=0.05)
    parser.add_argument("--risk-ridge", type=float, default=1e-6)
    parser.add_argument("--pair-weight-power", type=float, default=1.0)
    parser.add_argument("--bootstrap-samples", type=int, default=0)
    parser.add_argument("--bootstrap-seed", type=int, default=8801)
    parser.add_argument("--uncertainty-beta", type=float, default=0.0)
    parser.add_argument(
        "--spectral-mode",
        choices=("banded", "global"),
        default="banded",
        help="band-conditioned recovery or an equal-information global collapse",
    )
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = select(args)
    atomic_json(result, args.output.resolve())
    print(
        json.dumps(
            {
                key: value
                for key, value in result.items()
                if key
                not in {
                    "candidate_scores",
                    "candidate_point_estimates",
                    "candidate_uncertainty",
                    "candidate_band_risks",
                }
            },
            indent=2,
        )
    )
