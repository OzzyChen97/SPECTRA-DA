#!/usr/bin/env python3
"""Leave-one-shift-family-out diagnostics for source-simulated calibration.

This diagnostic never reads real target labels.  It groups the frozen
source-simulated shifts by mechanism, removes an entire mechanism family from
calibration, and evaluates each held-out simulated shift using only the
remaining families.  The original covariance recovery and the
curvature-normalized transported-risk prior are evaluated side by side.
"""

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

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from covariance_transport import (  # noqa: E402
    corrected_band_risk_recovery,
    match_shift_convex_combination,
    transport_statistics,
)
from metrics import normalized_regret, rank_correlations, top_fraction_hit  # noqa: E402
from scripts.trajectory_export.schema import atomic_json  # noqa: E402
from selector.objective import (  # noqa: E402
    DEVELOPMENT_TASKS,
    MAX_FRAME_ERROR,
    _heldout_disagreement,
    discover_calibrations,
    load_active_parameters,
)
from selector.spectra_cal import curvature_prior_strength, load_calibration  # noqa: E402

METHODS = (
    "spectra_cal_base",
    "spectra_curvature_prior",
    "spectra_match_adaptive_prior",
    "spectra_two_sigma_prior",
)

SHIFT_FAMILIES = (
    "feature",
    "structure",
    "homophily",
    "label_prior",
    "conditional_structure",
)


def shift_family(name: str) -> str:
    """Map a registered shift name to its mechanism family."""

    if name.startswith(("feature_mask", "feature_noise")):
        return "feature"
    if name.startswith("edge_dropout"):
        return "structure"
    if name.startswith("homophily"):
        return "homophily"
    if name.startswith("label_prior"):
        return "label_prior"
    if name == "conditional_structure":
        return name
    raise ValueError(f"unregistered shift mechanism: {name}")


def _finite_mean(values: list[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and np.isfinite(value)]
    return float(np.mean(finite)) if finite else None


def _finite_median(values: list[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and np.isfinite(value)]
    return float(statistics.median(finite)) if finite else None


def _recover(
    *,
    method: str,
    shift_deltas: np.ndarray,
    target_delta: np.ndarray,
    band_risks: np.ndarray,
    band_covariances: np.ndarray,
    disagreements: np.ndarray,
    parameters: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    alpha, match_diagnostics = match_shift_convex_combination(
        shift_deltas,
        target_delta,
        regularization=float(parameters["transport_regularization"]),
        descriptor_floor=float(parameters["descriptor_floor"]),
    )
    transported_risks, transported_covariances = transport_statistics(
        alpha,
        band_risks,
        band_covariances,
    )
    model_count = disagreements.shape[1]
    curvature = float(max(1, model_count - 2))
    if method == "spectra_cal_base":
        prior_strength = 0.0
    elif method == "spectra_curvature_prior":
        prior_strength = curvature
    elif method == "spectra_match_adaptive_prior":
        descriptor_rmse = float(match_diagnostics["descriptor_rmse"])
        prior_strength = curvature / (1.0 + descriptor_rmse**2)
    elif method == "spectra_two_sigma_prior":
        descriptor_rmse = float(match_diagnostics["descriptor_rmse"])
        prior_strength = curvature_prior_strength(disagreements, descriptor_rmse)
    else:
        raise KeyError(f"unknown diagnostic method: {method}")
    recovered, _ = corrected_band_risk_recovery(
        disagreements,
        transported_risks,
        transported_covariances,
        ridge=float(parameters["risk_ridge"]),
        pair_weight_power=float(parameters["pair_weight_power"]),
        prior_strength=prior_strength,
    )
    return recovered, alpha, match_diagnostics


def _bootstrap_uncertainty(
    *,
    method: str,
    shift_deltas: np.ndarray,
    target_delta: np.ndarray,
    band_risks: np.ndarray,
    band_covariances: np.ndarray,
    disagreements: np.ndarray,
    samples: int,
    seed: int,
    parameters: dict[str, Any],
) -> tuple[np.ndarray, float | None]:
    model_count = disagreements.shape[1]
    if samples <= 1:
        return np.zeros(model_count, dtype=np.float64), None
    generator = np.random.default_rng(seed)
    estimates = []
    selected_indices = []
    for _ in range(samples):
        sampled = generator.integers(0, shift_deltas.shape[0], size=shift_deltas.shape[0])
        recovered, _, _ = _recover(
            method=method,
            shift_deltas=shift_deltas[sampled],
            target_delta=target_delta,
            band_risks=band_risks[sampled],
            band_covariances=band_covariances[sampled],
            disagreements=disagreements,
            parameters=parameters,
        )
        estimate = 0.5 * recovered.sum(axis=1)
        estimates.append(estimate)
        selected_indices.append(int(np.argmin(estimate)))
    stacked = np.stack(estimates)
    counts = np.bincount(selected_indices, minlength=model_count)
    return np.std(stacked, axis=0, ddof=1), float(counts.max() / samples)


def evaluate_method(
    directory: Path,
    method: str,
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
    shift_specs = metadata["shift_specs"]
    shift_names = [spec["name"] for spec in shift_specs]
    families = []
    for spec in shift_specs:
        inferred = shift_family(spec["name"])
        declared = spec.get("family")
        if declared != inferred:
            raise ValueError(
                f"shift family mismatch for {spec['name']}: "
                f"declared={declared!r}, inferred={inferred!r}"
            )
        families.append(inferred)
    observed_families = set(families)
    if observed_families != set(SHIFT_FAMILIES):
        raise ValueError(
            "calibration must contain the five frozen shift families: "
            f"expected={list(SHIFT_FAMILIES)}, observed={sorted(observed_families)}"
        )
    family_order = list(SHIFT_FAMILIES)
    all_indices = np.arange(len(shift_names))
    fold_reports = []
    bootstrap_samples = int(parameters["bootstrap_samples"])
    uncertainty_beta = float(parameters["uncertainty_beta"])

    for family_index, family in enumerate(family_order):
        heldout_indices = np.asarray(
            [index for index, value in enumerate(families) if value == family],
            dtype=np.int64,
        )
        development = np.asarray(
            [index for index in all_indices if families[index] != family],
            dtype=np.int64,
        )
        if development.size < 3:
            raise ValueError(f"too few development shifts after holding out {family}")
        for heldout in heldout_indices:
            disagreements = _heldout_disagreement(
                band_risks[heldout],
                band_covariances[heldout],
            )
            recovered, alpha, match_diagnostics = _recover(
                method=method,
                shift_deltas=shift_deltas[development],
                target_delta=shift_deltas[heldout],
                band_risks=band_risks[development],
                band_covariances=band_covariances[development],
                disagreements=disagreements,
                parameters=parameters,
            )
            point_estimate = 0.5 * recovered.sum(axis=1)
            uncertainty, stability = _bootstrap_uncertainty(
                method=method,
                shift_deltas=shift_deltas[development],
                target_delta=shift_deltas[heldout],
                band_risks=band_risks[development],
                band_covariances=band_covariances[development],
                disagreements=disagreements,
                samples=bootstrap_samples,
                seed=bootstrap_seed + family_index * 1009 + int(heldout),
                parameters=parameters,
            )
            selection_score = point_estimate + uncertainty_beta * uncertainty
            true_risk = 0.5 * band_risks[heldout].sum(axis=1)
            selected = int(np.argmin(selection_score))
            correlations = rank_correlations(selection_score, true_risk)
            fold_reports.append(
                {
                    "heldout_family": family,
                    "heldout_shift_index": int(heldout),
                    "heldout_shift_name": shift_names[heldout],
                    "development_shift_count": int(development.size),
                    "selected_candidate_index": selected,
                    "oracle_candidate_index": int(np.argmin(true_risk)),
                    "normalized_regret": normalized_regret(
                        float(true_risk[selected]),
                        true_risk,
                    ),
                    "kendall_tau": correlations["kendall_tau"],
                    "spearman_rho": correlations["spearman_rho"],
                    "risk_estimation_mae": float(np.mean(np.abs(point_estimate - true_risk))),
                    "oracle_f1_gap": float(true_risk[selected] - true_risk.min()),
                    "top_5pct_hit": top_fraction_hit(selected, true_risk, fraction=0.05),
                    "selection_stability": stability,
                    "effective_shift_count": float(1.0 / np.sum(alpha**2)),
                    "descriptor_rmse": float(match_diagnostics["descriptor_rmse"]),
                    "descriptor_mae": float(match_diagnostics["descriptor_mae"]),
                }
            )

    family_reports = []
    for family in family_order:
        folds = [fold for fold in fold_reports if fold["heldout_family"] == family]
        family_reports.append(
            {
                "family": family,
                "fold_count": len(folds),
                "mean_normalized_regret": _finite_mean(
                    [fold["normalized_regret"] for fold in folds]
                ),
                "median_kendall_tau": _finite_median([fold["kendall_tau"] for fold in folds]),
                "mean_spearman_rho": _finite_mean([fold["spearman_rho"] for fold in folds]),
                "risk_estimation_mae": _finite_mean(
                    [fold["risk_estimation_mae"] for fold in folds]
                ),
                "mean_oracle_f1_gap": _finite_mean([fold["oracle_f1_gap"] for fold in folds]),
                "top_5pct_hit_rate": _finite_mean(
                    [float(fold["top_5pct_hit"]) for fold in folds]
                ),
            }
        )

    return {
        "task": metadata["task"],
        "method": method,
        "candidate_count": int(band_risks.shape[1]),
        "shift_count": int(band_risks.shape[0]),
        "family_count": len(family_order),
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
        "families": family_reports,
        "folds": fold_reports,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "7":
        raise RuntimeError("diagnostic requires CUDA_VISIBLE_DEVICES=7")
    calibration_root = args.calibration_root.resolve()
    if ".sealed" in calibration_root.parts:
        raise RuntimeError("diagnostic must not read sealed artifacts")
    parameters = load_active_parameters(args.config.resolve())
    tasks = args.tasks or list(DEVELOPMENT_TASKS)
    directories = discover_calibrations(calibration_root, tasks)
    reports = []
    for method_index, method in enumerate(METHODS):
        task_reports = [
            evaluate_method(
                directory,
                method,
                parameters,
                bootstrap_seed=args.bootstrap_seed + method_index * 100_003 + index * 10_007,
            )
            for index, directory in enumerate(directories)
        ]
        reports.append(
            {
                "method": method,
                "mean_normalized_regret": _finite_mean(
                    [report["mean_normalized_regret"] for report in task_reports]
                ),
                "median_kendall_tau": _finite_median(
                    [report["median_kendall_tau"] for report in task_reports]
                ),
                "mean_spearman_rho": _finite_mean(
                    [report["mean_spearman_rho"] for report in task_reports]
                ),
                "risk_estimation_mae": _finite_mean(
                    [report["risk_estimation_mae"] for report in task_reports]
                ),
                "mean_oracle_f1_gap": _finite_mean(
                    [report["mean_oracle_f1_gap"] for report in task_reports]
                ),
                "top_5pct_hit_rate": _finite_mean(
                    [report["top_5pct_hit_rate"] for report in task_reports]
                ),
                "selection_stability": _finite_mean(
                    [report["selection_stability"] for report in task_reports]
                ),
                "tasks": task_reports,
            }
        )
    max_frame_error = max(
        task["max_frame_error"]
        for method in reports
        for task in method["tasks"]
    )
    protocol_violations = int(max_frame_error > MAX_FRAME_ERROR)
    result = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "diagnostic": "leave-one-shift-family-out on source-simulated shifts",
        "objective_source": "source-simulated calibration only; no real target labels",
        "tasks": tasks,
        "active_parameters": parameters,
        "methods": reports,
        "max_frame_error": max_frame_error,
        "label_access_count": 0,
        "protocol_violation_count": protocol_violations,
        "physical_gpu": 7,
        "visible_cuda_devices": torch.cuda.device_count(),
        "runtime_seconds": time.perf_counter() - started,
    }
    if protocol_violations:
        raise RuntimeError(
            f"frame approximation guardrail failed: {max_frame_error} > {MAX_FRAME_ERROR}"
        )
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
    parser.add_argument("--bootstrap-seed", type=int, default=27183)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO / "results" / "gda_select" / "diagnostics" / "shift_type_holdout.json",
    )
    return parser.parse_args()


def main() -> None:
    result = run(parse_args())
    summary = {
        "diagnostic": result["diagnostic"],
        "methods": [
            {
                key: report[key]
                for key in (
                    "method",
                    "mean_normalized_regret",
                    "median_kendall_tau",
                    "mean_spearman_rho",
                    "risk_estimation_mae",
                    "mean_oracle_f1_gap",
                    "top_5pct_hit_rate",
                    "selection_stability",
                )
            }
            for report in result["methods"]
        ],
        "label_access_count": result["label_access_count"],
        "protocol_violation_count": result["protocol_violation_count"],
        "physical_gpu": result["physical_gpu"],
        "visible_cuda_devices": result["visible_cuda_devices"],
        "runtime_seconds": result["runtime_seconds"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
