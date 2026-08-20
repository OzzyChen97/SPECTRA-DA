#!/usr/bin/env python3
"""Controlled SPECTRA-DA refinement objective with frozen calibration sidecars.

The original ``selector/objective.py`` and formal calibration artifacts remain
unchanged.  This entrypoint evaluates the same 44 leave-one-simulated-shift-out
folds while adding a pre-frozen, source-labelled sidecar bank to each fold's
development set.  Real target labels and the sealed evaluator are never read.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from metrics import normalized_regret, rank_correlations, top_fraction_hit  # noqa: E402
from scripts.trajectory_export.schema import atomic_json, sha256_file  # noqa: E402
from selector.objective import (  # noqa: E402
    DEVELOPMENT_TASKS,
    MAX_FRAME_ERROR,
    _heldout_disagreement,
    discover_calibrations,
    load_active_parameters,
)
from selector.spectra_cal import load_calibration, recover_with_transport  # noqa: E402

SIDECAR_ARRAY_NAMES = (
    "shift_deltas",
    "band_risks",
    "band_covariances",
)


def _finite_mean(values: list[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and np.isfinite(value)]
    return float(np.mean(finite)) if finite else None


def _finite_median(values: list[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and np.isfinite(value)]
    return float(statistics.median(finite)) if finite else None


def _top_weighted_kendall(predicted: np.ndarray, truth: np.ndarray) -> float:
    model_count = truth.size
    first, second = np.triu_indices(model_count, k=1)
    true_order = np.argsort(np.argsort(truth, kind="stable"), kind="stable")
    temperature = max(1.0, 0.1 * model_count)
    weights = np.abs(truth[first] - truth[second]) * np.exp(
        -np.minimum(true_order[first], true_order[second]) / temperature
    )
    valid = weights > 0.0
    discordant = (
        (predicted[first] - predicted[second])
        * (truth[first] - truth[second])
        < 0.0
    )
    total = float(np.sum(weights[valid]))
    return float(
        1.0
        - 2.0
        * float(np.sum(weights[valid & discordant]))
        / max(total, 1.0e-12)
    )


def _is_feature_mask_grid(metadata: dict[str, Any]) -> bool:
    selected = metadata.get("selected_shifts") or []
    return bool(selected) and all(
        shift.get("calibration_family") == "feature_mask_grid"
        for shift in selected
    )


def _source_node_count(metadata: dict[str, Any]) -> int | None:
    selected = metadata.get("selected_shifts") or []
    if not selected:
        return None
    counts = {int(shift["node_count"]) for shift in selected if "node_count" in shift}
    return next(iter(counts)) if len(counts) == 1 else None


def covariance_rank_gate_allows(metadata: dict[str, Any]) -> bool:
    """Reject only feature-mask grids with rank-deficient source covariance."""

    if not _is_feature_mask_grid(metadata):
        return True
    source_nodes = _source_node_count(metadata)
    if source_nodes is None:
        raise ValueError("feature-mask sidecar has inconsistent source node counts")
    return source_nodes >= int(metadata["candidate_count"])


def load_sidecar_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported refinement sidecar manifest schema")
    if manifest.get("target_label_access_count") != 0:
        raise RuntimeError("sidecar manifest reports target-label access")
    if manifest.get("protocol_violation_count") != 0:
        raise RuntimeError("sidecar manifest reports a protocol violation")
    if manifest.get("rank_gate") != "source_nodes_gte_candidate_count_for_feature_mask_grid":
        raise ValueError("unexpected covariance rank gate")
    tasks = manifest.get("tasks")
    if not isinstance(tasks, dict) or set(tasks) != set(DEVELOPMENT_TASKS):
        raise ValueError("sidecar manifest must freeze all development tasks")
    return manifest


def _load_task_sidecars(
    *,
    manifest: dict[str, Any],
    task: str,
    base_metadata: dict[str, Any],
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], list[dict[str, Any]]]:
    parts: list[dict[str, np.ndarray]] = []
    accepted: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for entry in manifest["tasks"][task]:
        directory = Path(entry["path"]).resolve()
        metadata_path = directory / "metadata.json"
        arrays_path = directory / "calibration_sidecar.npz"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected_hash = str(entry["artifact_sha256"])
        if metadata.get("artifact_sha256", {}).get(arrays_path.name) != expected_hash:
            raise ValueError(f"sidecar metadata hash mismatch: {directory}")
        if sha256_file(arrays_path) != expected_hash:
            raise ValueError(f"sidecar artifact hash mismatch: {directory}")
        if metadata.get("target_label_access_count") != 0:
            raise RuntimeError(f"sidecar reports target-label access: {directory}")
        if metadata.get("protocol_violation_count") != 0:
            raise RuntimeError(f"sidecar reports a protocol violation: {directory}")
        if metadata.get("task") != task:
            raise ValueError(f"sidecar task mismatch: {directory}")
        if metadata.get("candidate_bank_sha256") != base_metadata.get(
            "candidate_bank_sha256"
        ):
            raise ValueError(f"sidecar candidate-bank mismatch: {directory}")
        if metadata.get("candidate_ids") != base_metadata.get("candidate_ids"):
            raise ValueError(f"sidecar candidate ordering mismatch: {directory}")
        if metadata.get("spectral_config") != base_metadata.get("spectral_config"):
            raise ValueError(f"sidecar spectral configuration mismatch: {directory}")
        parent_hash = base_metadata.get("artifact_sha256", {}).get("calibration.npz")
        if metadata.get("parent_calibration_sha256") != parent_hash:
            raise ValueError(f"sidecar parent calibration mismatch: {directory}")
        if not covariance_rank_gate_allows(metadata):
            skipped.append(
                {
                    "task": task,
                    "path": str(directory),
                    "reason": "source_nodes_below_candidate_count",
                    "source_nodes": _source_node_count(metadata),
                    "candidate_count": int(metadata["candidate_count"]),
                }
            )
            continue
        with np.load(arrays_path, allow_pickle=False) as artifact:
            arrays = {name: np.asarray(artifact[name]) for name in SIDECAR_ARRAY_NAMES}
        parts.append(arrays)
        accepted.append(
            {
                "task": task,
                "path": str(directory),
                "artifact_sha256": expected_hash,
                "shift_count": int(arrays["shift_deltas"].shape[0]),
                "feature_mask_grid": _is_feature_mask_grid(metadata),
                "source_nodes": _source_node_count(metadata),
            }
        )
    if not parts:
        raise ValueError(f"no sidecar survived the frozen gate for task {task}")
    combined = {
        name: np.concatenate([part[name] for part in parts], axis=0)
        for name in SIDECAR_ARRAY_NAMES
    }
    return combined, accepted, skipped


def _bootstrap_fold_scores_threaded(
    *,
    shift_deltas: np.ndarray,
    band_risks: np.ndarray,
    band_covariances: np.ndarray,
    target_delta: np.ndarray,
    disagreements: np.ndarray,
    samples: int,
    seed: int,
    parameters: dict[str, Any],
    workers: int,
) -> tuple[np.ndarray, float | None]:
    model_count = disagreements.shape[1]
    if samples <= 1:
        return np.zeros(model_count, dtype=np.float64), None
    generator = np.random.default_rng(seed)
    shift_count = shift_deltas.shape[0]
    sampled_sets = [
        generator.integers(0, shift_count, size=shift_count)
        for _ in range(samples)
    ]

    def recover(sampled: np.ndarray) -> np.ndarray:
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
        return 0.5 * recovered.sum(axis=1)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        estimates = list(executor.map(recover, sampled_sets))
    stacked = np.stack(estimates)
    selected = np.argmin(stacked, axis=1)
    counts = np.bincount(selected, minlength=model_count)
    return np.std(stacked, axis=0, ddof=1), float(counts.max() / samples)


def _task_report(
    *,
    directory: Path,
    manifest: dict[str, Any],
    parameters: dict[str, Any],
    bootstrap_seed: int,
    bootstrap_workers: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metadata, parent = load_calibration(directory)
    task = str(metadata["task"])
    sidecar, accepted, skipped = _load_task_sidecars(
        manifest=manifest,
        task=task,
        base_metadata=metadata,
    )
    shift_deltas = np.asarray(parent["shift_deltas"], dtype=np.float64)
    band_risks = np.asarray(parent["band_risks"], dtype=np.float64)
    band_covariances = np.asarray(parent["band_covariances"], dtype=np.float64)
    true_risks = 0.5 * band_risks.sum(axis=2)
    all_indices = np.arange(shift_deltas.shape[0])
    folds: list[dict[str, Any]] = []
    for heldout in all_indices:
        development = all_indices[all_indices != heldout]
        combined_deltas = np.concatenate(
            [shift_deltas[development], sidecar["shift_deltas"]], axis=0
        )
        combined_risks = np.concatenate(
            [band_risks[development], sidecar["band_risks"]], axis=0
        )
        combined_covariances = np.concatenate(
            [band_covariances[development], sidecar["band_covariances"]], axis=0
        )
        disagreements = _heldout_disagreement(
            band_risks[heldout],
            band_covariances[heldout],
        )
        recovered, alpha, diagnostics = recover_with_transport(
            shift_deltas=combined_deltas,
            target_delta=shift_deltas[heldout],
            shift_band_risks=combined_risks,
            shift_band_covariances=combined_covariances,
            disagreements=disagreements,
            transport_regularization=float(parameters["transport_regularization"]),
            descriptor_floor=float(parameters["descriptor_floor"]),
            risk_ridge=float(parameters["risk_ridge"]),
            pair_weight_power=float(parameters["pair_weight_power"]),
        )
        point_estimate = 0.5 * recovered.sum(axis=1)
        uncertainty, stability = _bootstrap_fold_scores_threaded(
            shift_deltas=combined_deltas,
            band_risks=combined_risks,
            band_covariances=combined_covariances,
            target_delta=shift_deltas[heldout],
            disagreements=disagreements,
            samples=int(parameters["bootstrap_samples"]),
            seed=bootstrap_seed + int(heldout),
            parameters=parameters,
            workers=bootstrap_workers,
        )
        selection_score = point_estimate + float(parameters["uncertainty_beta"]) * uncertainty
        truth = true_risks[heldout]
        selected = int(np.argmin(selection_score))
        correlations = rank_correlations(selection_score, truth)
        folds.append(
            {
                "heldout_shift_index": int(heldout),
                "heldout_shift_name": metadata["shift_specs"][heldout]["name"],
                "selected_candidate_index": selected,
                "oracle_candidate_index": int(np.argmin(truth)),
                "normalized_regret": normalized_regret(float(truth[selected]), truth),
                "kendall_tau": correlations["kendall_tau"],
                "spearman_rho": correlations["spearman_rho"],
                "top_weighted_kendall": _top_weighted_kendall(selection_score, truth),
                "risk_estimation_mae": float(np.mean(np.abs(selection_score - truth))),
                "oracle_f1_gap": float(truth[selected] - truth.min()),
                "top_5pct_hit": top_fraction_hit(selected, truth, fraction=0.05),
                "selection_stability": stability,
                "effective_shift_count": float(1.0 / np.sum(alpha**2)),
                "sidecar_weight_mass": float(np.sum(alpha[len(development) :])),
                "descriptor_rmse": float(diagnostics["matching"]["descriptor_rmse"]),
            }
        )
    regrets = np.asarray([fold["normalized_regret"] for fold in folds], dtype=np.float64)
    tail_count = max(1, math.ceil(0.2 * len(regrets)))
    report = {
        "task": task,
        "calibration_id": metadata["calibration_id"],
        "candidate_bank_sha256": metadata["candidate_bank_sha256"],
        "candidate_count": int(band_risks.shape[1]),
        "base_shift_count": int(band_risks.shape[0]),
        "sidecar_shift_count": int(sidecar["shift_deltas"].shape[0]),
        "accepted_sidecars": accepted,
        "skipped_sidecars": skipped,
        "max_frame_error": float(metadata["frame_diagnostics"]["max_frame_error"]),
        "mean_normalized_regret": float(np.mean(regrets)),
        "cvar20": float(np.mean(np.sort(regrets)[-tail_count:])),
        "worst_fold_regret": float(np.max(regrets)),
        "median_kendall_tau": _finite_median([fold["kendall_tau"] for fold in folds]),
        "mean_spearman_rho": _finite_mean([fold["spearman_rho"] for fold in folds]),
        "mean_top_weighted_kendall": _finite_mean(
            [fold["top_weighted_kendall"] for fold in folds]
        ),
        "risk_estimation_mae": _finite_mean(
            [fold["risk_estimation_mae"] for fold in folds]
        ),
        "mean_oracle_f1_gap": _finite_mean([fold["oracle_f1_gap"] for fold in folds]),
        "top_5pct_hit_rate": _finite_mean(
            [float(fold["top_5pct_hit"]) for fold in folds]
        ),
        "selection_stability": _finite_mean(
            [fold["selection_stability"] for fold in folds]
        ),
        "folds": folds,
    }
    return report, skipped


def _reference_regrets(path: Path) -> dict[tuple[str, str], float]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("label_access_count") != 0:
        raise RuntimeError("reference development result reports target-label access")
    return {
        (str(task_report["task"]), str(fold["heldout_shift_name"])): float(
            fold["normalized_regret"]
        )
        for task_report in document["tasks"]
        for fold in task_report["folds"]
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "7":
        raise RuntimeError("refinement objective requires CUDA_VISIBLE_DEVICES=7")
    if args.bootstrap_workers < 1:
        raise ValueError("bootstrap workers must be positive")
    parameters = load_active_parameters(args.config.resolve())
    manifest_path = args.sidecar_manifest.resolve()
    manifest = load_sidecar_manifest(manifest_path)
    tasks = args.tasks or list(DEVELOPMENT_TASKS)
    if tasks != list(DEVELOPMENT_TASKS):
        raise ValueError("refinement objective requires all frozen development tasks")
    directories = discover_calibrations(args.calibration_root.resolve(), tasks)
    reports: list[dict[str, Any]] = []
    skipped_sidecars: list[dict[str, Any]] = []
    for index, directory in enumerate(directories):
        report, skipped = _task_report(
            directory=directory,
            manifest=manifest,
            parameters=parameters,
            bootstrap_seed=args.bootstrap_seed + index * 1009,
            bootstrap_workers=args.bootstrap_workers,
        )
        reports.append(report)
        skipped_sidecars.extend(skipped)

    folds = [
        {"task": report["task"], **fold}
        for report in reports
        for fold in report["folds"]
    ]
    regrets = np.asarray([fold["normalized_regret"] for fold in folds], dtype=np.float64)
    tail_count = max(1, math.ceil(0.2 * len(regrets)))
    reference = _reference_regrets(args.reference_results.resolve())
    reference_values = np.asarray(
        [reference[(fold["task"], fold["heldout_shift_name"])] for fold in folds],
        dtype=np.float64,
    )
    gains = np.maximum(reference_values - regrets, 0.0)
    max_frame_error = max(float(report["max_frame_error"]) for report in reports)
    protocol_violations = int(max_frame_error > MAX_FRAME_ERROR)
    result = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "objective_source": (
            "same frozen source-simulated folds plus pre-frozen source-labelled "
            "calibration sidecars; no real target labels"
        ),
        "task_count": len(reports),
        "candidate_count_total": sum(int(report["candidate_count"]) for report in reports),
        "simulated_shift_fold_count": len(folds),
        "sidecar_shift_count_total": sum(
            int(report["sidecar_shift_count"]) for report in reports
        ),
        "skipped_sidecar_count": len(skipped_sidecars),
        "mean_normalized_regret_dev": float(np.mean(regrets)),
        "cvar20_dev": float(np.mean(np.sort(regrets)[-tail_count:])),
        "worst_fold_regret_dev": float(np.max(regrets)),
        "median_kendall_tau_dev": _finite_median(
            [report["median_kendall_tau"] for report in reports]
        ),
        "mean_spearman_rho_dev": _finite_mean(
            [report["mean_spearman_rho"] for report in reports]
        ),
        "mean_top_weighted_kendall_dev": _finite_mean(
            [report["mean_top_weighted_kendall"] for report in reports]
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
        "localized_gain_share_dev": float(
            np.max(gains) / max(float(np.sum(gains)), 1.0e-12)
        ),
        "positive_gain_fold_count_dev": int(np.sum(gains > 0.0)),
        "selector_runtime_seconds": time.perf_counter() - started,
        "max_frame_error": max_frame_error,
        "label_access_count": 0,
        "protocol_violation_count": protocol_violations,
        "physical_gpu": 7,
        "visible_cuda_devices": torch.cuda.device_count(),
        "active_parameters": parameters,
        "bootstrap_workers": args.bootstrap_workers,
        "sidecar_manifest": str(manifest_path),
        "sidecar_manifest_sha256": sha256_file(manifest_path),
        "reference_results": str(args.reference_results.resolve()),
        "reference_results_sha256": sha256_file(args.reference_results.resolve()),
        "skipped_sidecars": skipped_sidecars,
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
        required=True,
        help="directory containing the frozen source-simulated calibration tasks",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO / "configs" / "search_space.yaml",
    )
    parser.add_argument(
        "--sidecar-manifest",
        type=Path,
        required=True,
        help="local manifest built from configs/refinement_sidecars.example.json",
    )
    parser.add_argument(
        "--reference-results",
        type=Path,
        required=True,
        help="frozen development-fold reference result used for gain diagnostics",
    )
    parser.add_argument("--tasks", nargs="*")
    parser.add_argument("--bootstrap-seed", type=int, default=19461)
    parser.add_argument("--bootstrap-workers", type=int, default=4)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    result = run(parse_args())
    metric_names = [
        "mean_normalized_regret_dev",
        "cvar20_dev",
        "worst_fold_regret_dev",
        "median_kendall_tau_dev",
        "mean_spearman_rho_dev",
        "mean_top_weighted_kendall_dev",
        "risk_estimation_mae_dev",
        "mean_oracle_f1_gap_dev",
        "top_5pct_hit_rate_dev",
        "selection_stability_dev",
        "localized_gain_share_dev",
        "positive_gain_fold_count_dev",
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
