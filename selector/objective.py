#!/usr/bin/env python3
"""Target-label-free AutoSOTA objective on held-out source-simulated shifts."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from metrics import normalized_regret, rank_correlations, top_fraction_hit  # noqa: E402
from scripts.trajectory_export.schema import atomic_json  # noqa: E402
from selector.spectra_cal import load_calibration, recover_with_transport  # noqa: E402

DEVELOPMENT_TASKS = (
    "ACMv9_to_Citationv1",
    "Citationv1_to_ACMv9",
    "USA_to_BRAZIL",
    "BRAZIL_to_USA",
)
MAX_FRAME_ERROR = 0.05
ALLOWED_PARAMETERS: dict[str, tuple[int | float, ...]] = {
    "transport_regularization": (0.0, 0.01, 0.1, 1.0, 10.0),
    "descriptor_floor": (0.01, 0.05, 0.1, 0.25),
    "risk_ridge": (0.0, 1.0e-8, 1.0e-6, 1.0e-4, 1.0e-2),
    "pair_weight_power": (0.0, 0.5, 1.0, 2.0),
    "bootstrap_samples": (0, 8, 16, 32),
    "uncertainty_beta": (0.0, 0.25, 0.5, 1.0, 2.0),
}


def _finite_mean(values: list[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and np.isfinite(value)]
    return float(np.mean(finite)) if finite else None


def _finite_median(values: list[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and np.isfinite(value)]
    return float(statistics.median(finite)) if finite else None


def _heldout_disagreement(
    band_risks: np.ndarray,
    band_covariances: np.ndarray,
) -> np.ndarray:
    risks = np.asarray(band_risks, dtype=np.float64)
    covariances = np.asarray(band_covariances, dtype=np.float64)
    if risks.ndim != 2:
        raise ValueError("held-out band risks must have shape [models, bands]")
    model_count, band_count = risks.shape
    if covariances.shape != (band_count, model_count, model_count):
        raise ValueError("held-out band covariance shape mismatch")
    disagreements = np.empty_like(covariances)
    for band in range(band_count):
        band_risk = risks[:, band]
        matrix = band_risk[:, None] + band_risk[None, :] - 2.0 * covariances[band]
        disagreements[band] = np.maximum(matrix, 0.0)
        np.fill_diagonal(disagreements[band], 0.0)
    return disagreements


def _bootstrap_fold_scores(
    *,
    shift_deltas: np.ndarray,
    band_risks: np.ndarray,
    band_covariances: np.ndarray,
    target_delta: np.ndarray,
    disagreements: np.ndarray,
    samples: int,
    seed: int,
    parameters: dict[str, Any],
) -> tuple[np.ndarray, float | None]:
    model_count = disagreements.shape[1]
    if samples <= 1:
        return np.zeros(model_count, dtype=np.float64), None
    generator = np.random.default_rng(seed)
    shift_count = shift_deltas.shape[0]
    estimates = []
    selected_indices = []
    for _ in range(samples):
        sampled = generator.integers(0, shift_count, size=shift_count)
        recovered, _, _ = recover_with_transport(
            shift_deltas=shift_deltas[sampled],
            target_delta=target_delta,
            shift_band_risks=band_risks[sampled],
            shift_band_covariances=band_covariances[sampled],
            disagreements=disagreements,
            transport_regularization=float(parameters["transport_regularization"]),
            descriptor_floor=float(parameters["descriptor_floor"]),
            risk_ridge=float(parameters["risk_ridge"]),
            pair_weight_power=float(parameters["pair_weight_power"]),
        )
        estimate = 0.5 * recovered.sum(axis=1)
        estimates.append(estimate)
        selected_indices.append(int(np.argmin(estimate)))
    stacked = np.stack(estimates)
    counts = np.bincount(selected_indices, minlength=model_count)
    stability = float(counts.max() / samples)
    return np.std(stacked, axis=0, ddof=1), stability


def evaluate_calibration(
    directory: Path,
    parameters: dict[str, Any],
    *,
    bootstrap_seed: int,
) -> dict[str, Any]:
    metadata, arrays = load_calibration(directory)
    if metadata.get("target_label_access_count") != 0:
        raise RuntimeError("calibration artifact reports target-label access")
    shift_deltas = np.asarray(arrays["shift_deltas"], dtype=np.float64)
    band_risks = np.asarray(arrays["band_risks"], dtype=np.float64)
    band_covariances = np.asarray(arrays["band_covariances"], dtype=np.float64)
    shift_count, model_count, band_count = band_risks.shape
    if shift_count < 3 or model_count < 3:
        raise ValueError("objective requires at least three shifts and three candidates")
    if shift_deltas.shape[0] != shift_count:
        raise ValueError("calibration shift descriptors do not align")
    if band_covariances.shape != (shift_count, band_count, model_count, model_count):
        raise ValueError("calibration covariance arrays do not align")

    fold_reports = []
    bootstrap_samples = int(parameters["bootstrap_samples"])
    uncertainty_beta = float(parameters["uncertainty_beta"])
    all_indices = np.arange(shift_count)
    for heldout in range(shift_count):
        development = all_indices[all_indices != heldout]
        disagreements = _heldout_disagreement(
            band_risks[heldout],
            band_covariances[heldout],
        )
        recovered, alpha, diagnostics = recover_with_transport(
            shift_deltas=shift_deltas[development],
            target_delta=shift_deltas[heldout],
            shift_band_risks=band_risks[development],
            shift_band_covariances=band_covariances[development],
            disagreements=disagreements,
            transport_regularization=float(parameters["transport_regularization"]),
            descriptor_floor=float(parameters["descriptor_floor"]),
            risk_ridge=float(parameters["risk_ridge"]),
            pair_weight_power=float(parameters["pair_weight_power"]),
        )
        point_estimate = 0.5 * recovered.sum(axis=1)
        uncertainty, stability = _bootstrap_fold_scores(
            shift_deltas=shift_deltas[development],
            band_risks=band_risks[development],
            band_covariances=band_covariances[development],
            target_delta=shift_deltas[heldout],
            disagreements=disagreements,
            samples=bootstrap_samples,
            seed=bootstrap_seed + heldout,
            parameters=parameters,
        )
        selection_score = point_estimate + uncertainty_beta * uncertainty
        true_risk = 0.5 * band_risks[heldout].sum(axis=1)
        selected = int(np.argmin(selection_score))
        correlations = rank_correlations(selection_score, true_risk)
        fold_reports.append(
            {
                "heldout_shift_index": heldout,
                "heldout_shift_name": metadata["shift_specs"][heldout]["name"],
                "selected_candidate_index": selected,
                "oracle_candidate_index": int(np.argmin(true_risk)),
                "normalized_regret": normalized_regret(float(true_risk[selected]), true_risk),
                "kendall_tau": correlations["kendall_tau"],
                "spearman_rho": correlations["spearman_rho"],
                "risk_estimation_mae": float(np.mean(np.abs(point_estimate - true_risk))),
                "oracle_f1_gap": float(true_risk[selected] - true_risk.min()),
                "top_5pct_hit": top_fraction_hit(selected, true_risk, fraction=0.05),
                "selection_stability": stability,
                "effective_shift_count": float(1.0 / np.sum(alpha**2)),
                "transport_diagnostics": diagnostics,
            }
        )

    return {
        "task": metadata["task"],
        "calibration_id": metadata["calibration_id"],
        "candidate_bank_sha256": metadata["candidate_bank_sha256"],
        "candidate_count": model_count,
        "shift_count": shift_count,
        "max_frame_error": float(metadata["frame_diagnostics"]["max_frame_error"]),
        "mean_normalized_regret": _finite_mean(
            [fold["normalized_regret"] for fold in fold_reports]
        ),
        "median_kendall_tau": _finite_median([fold["kendall_tau"] for fold in fold_reports]),
        "mean_spearman_rho": _finite_mean([fold["spearman_rho"] for fold in fold_reports]),
        "risk_estimation_mae": _finite_mean(
            [fold["risk_estimation_mae"] for fold in fold_reports]
        ),
        "mean_oracle_f1_gap": _finite_mean([fold["oracle_f1_gap"] for fold in fold_reports]),
        "top_5pct_hit_rate": _finite_mean(
            [float(fold["top_5pct_hit"]) for fold in fold_reports]
        ),
        "selection_stability": _finite_mean(
            [fold["selection_stability"] for fold in fold_reports]
        ),
        "folds": fold_reports,
    }


def load_active_parameters(path: Path) -> dict[str, Any]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    parameters = dict(document.get("active_parameters", {}))
    required = set(ALLOWED_PARAMETERS)
    missing = sorted(required - set(parameters))
    if missing:
        raise ValueError(f"missing active selector parameters: {missing}")
    extra = sorted(set(parameters) - required)
    if extra:
        raise ValueError(f"unregistered active selector parameters: {extra}")
    for name, allowed in ALLOWED_PARAMETERS.items():
        if parameters[name] not in allowed:
            raise ValueError(
                f"active parameter {name}={parameters[name]!r} is outside the frozen search space {allowed}"
            )
    return parameters


def discover_calibrations(root: Path, tasks: list[str]) -> list[Path]:
    directories = []
    for task in tasks:
        task_root = root / task
        candidates = sorted(
            path.parent
            for path in task_root.glob("*/metadata.json")
            if (path.parent / "calibration.npz").exists()
        )
        if not candidates:
            raise FileNotFoundError(f"no calibration artifact found for task {task}: {task_root}")
        if len(candidates) != 1:
            raise ValueError(
                f"task {task} has {len(candidates)} calibration artifacts; "
                "pass a root with one frozen artifact per task"
            )
        directories.append(candidates[0])
    return directories


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "7":
        raise RuntimeError("AutoSOTA objective requires CUDA_VISIBLE_DEVICES=7")
    parameters = load_active_parameters(args.config.resolve())
    tasks = args.tasks or list(DEVELOPMENT_TASKS)
    directories = discover_calibrations(args.calibration_root.resolve(), tasks)
    reports = [
        evaluate_calibration(directory, parameters, bootstrap_seed=args.bootstrap_seed + index * 1009)
        for index, directory in enumerate(directories)
    ]
    max_frame_error = max(report["max_frame_error"] for report in reports)
    protocol_violations = int(max_frame_error > MAX_FRAME_ERROR)
    result = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "objective_source": "leave-one-simulated-shift-out; no real target labels",
        "task_count": len(reports),
        "candidate_count_total": sum(report["candidate_count"] for report in reports),
        "simulated_shift_fold_count": sum(report["shift_count"] for report in reports),
        "mean_normalized_regret_dev": _finite_mean(
            [report["mean_normalized_regret"] for report in reports]
        ),
        "median_kendall_tau_dev": _finite_median(
            [report["median_kendall_tau"] for report in reports]
        ),
        "mean_spearman_rho_dev": _finite_mean(
            [report["mean_spearman_rho"] for report in reports]
        ),
        "risk_estimation_mae_dev": _finite_mean(
            [report["risk_estimation_mae"] for report in reports]
        ),
        "mean_oracle_f1_gap_dev": _finite_mean(
            [report["mean_oracle_f1_gap"] for report in reports]
        ),
        "top_5pct_hit_rate_dev": _finite_mean(
            [report["top_5pct_hit_rate"] for report in reports]
        ),
        "selection_stability_dev": _finite_mean(
            [report["selection_stability"] for report in reports]
        ),
        "selector_runtime_seconds": time.perf_counter() - started,
        "max_frame_error": max_frame_error,
        "label_access_count": 0,
        "protocol_violation_count": protocol_violations,
        "physical_gpu": 7,
        "visible_cuda_devices": torch.cuda.device_count(),
        "active_parameters": parameters,
        "tasks": reports,
    }
    if protocol_violations:
        raise RuntimeError(
            f"frame approximation guardrail failed: {max_frame_error} > {MAX_FRAME_ERROR}"
        )
    if args.output is not None:
        atomic_json(result, args.output.resolve())
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--calibration-root",
        type=Path,
        default=REPO / "trajectory_bank" / "calibration" / "formal",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO / "configs" / "search_space.yaml",
    )
    parser.add_argument("--tasks", nargs="*")
    parser.add_argument("--bootstrap-seed", type=int, default=19461)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    result = run(parse_args())
    metric_names = [
        "mean_normalized_regret_dev",
        "median_kendall_tau_dev",
        "mean_spearman_rho_dev",
        "risk_estimation_mae_dev",
        "mean_oracle_f1_gap_dev",
        "top_5pct_hit_rate_dev",
        "selection_stability_dev",
        "selector_runtime_seconds",
        "max_frame_error",
        "label_access_count",
        "protocol_violation_count",
        "physical_gpu",
        "visible_cuda_devices",
    ]
    for name in metric_names:
        print(f"{name} = {result[name]}")
    print(json.dumps({name: result[name] for name in metric_names}, sort_keys=True))


if __name__ == "__main__":
    main()
