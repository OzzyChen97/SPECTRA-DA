#!/usr/bin/env python3
"""Core SPECTRA-DA ablations on frozen source-simulated development folds."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from covariance_transport import (  # noqa: E402
    corrected_band_risk_recovery,
    match_shift_convex_combination,
    transport_statistics,
)
from metrics import normalized_regret, rank_correlations, top_fraction_hit  # noqa: E402
from scripts.trajectory_export.schema import atomic_json, sha256_file  # noqa: E402
from selector.objective import (  # noqa: E402
    DEVELOPMENT_TASKS,
    _heldout_disagreement,
    discover_calibrations,
    load_active_parameters,
)
from selector.refinement_objective import (  # noqa: E402
    _finite_mean,
    _finite_median,
    _load_task_sidecars,
    _top_weighted_kendall,
    load_sidecar_manifest,
)
from selector.spectra_cal import (  # noqa: E402
    curvature_prior_strength,
    load_calibration,
)


VARIANTS = (
    "global_linear",
    "spectral_linear",
    "spectral_no_covariance",
    "spectral_no_covariance_uncertainty",
    "global_covariance",
    "global_covariance_uncertainty",
    "spectral_covariance_no_robust",
    "spectral_covariance_no_robust_uncertainty",
    "spectral_covariance",
    "spectral_covariance_uncertainty",
)


def _risk_score(recovered: np.ndarray) -> np.ndarray:
    return 0.5 * np.asarray(recovered, dtype=np.float64).sum(axis=1)


def _recover(
    disagreements: np.ndarray,
    transported_risks: np.ndarray,
    transported_covariances: np.ndarray,
    *,
    parameters: dict[str, Any],
    prior_strength: float,
    robust: bool,
    pair_weight_power: float | None = None,
) -> np.ndarray:
    recovered, _ = corrected_band_risk_recovery(
        disagreements,
        transported_risks,
        transported_covariances,
        ridge=float(parameters["risk_ridge"]),
        pair_weight_power=(
            float(parameters["pair_weight_power"])
            if pair_weight_power is None
            else pair_weight_power
        ),
        prior_strength=prior_strength,
        robust=robust,
    )
    return _risk_score(recovered)


def _fold_metrics(score: np.ndarray, truth: np.ndarray) -> dict[str, Any]:
    selected = int(np.argmin(score))
    correlations = rank_correlations(score, truth)
    return {
        "selected_candidate_index": selected,
        "oracle_candidate_index": int(np.argmin(truth)),
        "normalized_regret": normalized_regret(float(truth[selected]), truth),
        "kendall_tau": correlations["kendall_tau"],
        "spearman_rho": correlations["spearman_rho"],
        "top_weighted_kendall": _top_weighted_kendall(score, truth),
        "risk_estimation_mae": float(np.mean(np.abs(score - truth))),
        "oracle_f1_gap": float(truth[selected] - truth.min()),
        "top_5pct_hit": top_fraction_hit(selected, truth, fraction=0.05),
    }


def _bootstrap_variant_uncertainty(
    *,
    variant: str,
    sampled_sets: list[np.ndarray],
    shift_deltas: np.ndarray,
    band_risks: np.ndarray,
    band_covariances: np.ndarray,
    target_delta: np.ndarray,
    disagreements: np.ndarray,
    parameters: dict[str, Any],
    workers: int,
) -> np.ndarray:
    """Bootstrap one point-estimator variant with every other choice fixed."""

    model_count = disagreements.shape[1]
    if len(sampled_sets) <= 1:
        return np.zeros(model_count, dtype=np.float64)
    if variant not in {
        "spectral_no_covariance",
        "global_covariance",
        "spectral_covariance_no_robust",
        "spectral_covariance",
    }:
        raise KeyError(f"unsupported bootstrap ablation variant: {variant}")

    def recover(sampled: np.ndarray) -> np.ndarray:
        sampled_risks = band_risks[sampled]
        sampled_covariances = band_covariances[sampled]
        observed = disagreements
        if variant == "global_covariance":
            sampled_risks = sampled_risks.sum(axis=2, keepdims=True)
            sampled_covariances = sampled_covariances.sum(axis=1, keepdims=True)
            observed = disagreements.sum(axis=0, keepdims=True)
        alpha, matching = match_shift_convex_combination(
            shift_deltas[sampled],
            target_delta,
            regularization=float(parameters["transport_regularization"]),
            descriptor_floor=float(parameters["descriptor_floor"]),
        )
        transported_risks, transported_covariances = transport_statistics(
            alpha,
            sampled_risks,
            sampled_covariances,
        )
        robust = variant != "spectral_covariance_no_robust"
        pair_weight_power: float | None = None
        if variant == "spectral_no_covariance":
            transported_covariances = np.zeros_like(transported_covariances)
            pair_weight_power = 0.0
        return _recover(
            observed,
            transported_risks,
            transported_covariances,
            parameters=parameters,
            prior_strength=curvature_prior_strength(
                observed,
                float(matching["descriptor_rmse"]),
            ),
            robust=robust,
            pair_weight_power=pair_weight_power,
        )

    with ThreadPoolExecutor(max_workers=workers) as executor:
        estimates = list(executor.map(recover, sampled_sets))
    return np.std(np.stack(estimates), axis=0, ddof=1)


def _aggregate(folds: list[dict[str, Any]]) -> dict[str, Any]:
    regrets = np.asarray([fold["normalized_regret"] for fold in folds])
    tail_count = max(1, math.ceil(0.2 * len(regrets)))
    return {
        "fold_count": len(folds),
        "mean_normalized_regret": float(np.mean(regrets)),
        "cvar20": float(np.mean(np.sort(regrets)[-tail_count:])),
        "worst_fold_regret": float(np.max(regrets)),
        "median_kendall_tau": _finite_median(
            [fold["kendall_tau"] for fold in folds]
        ),
        "mean_spearman_rho": _finite_mean(
            [fold["spearman_rho"] for fold in folds]
        ),
        "mean_top_weighted_kendall": _finite_mean(
            [fold["top_weighted_kendall"] for fold in folds]
        ),
        "risk_estimation_mae": _finite_mean(
            [fold["risk_estimation_mae"] for fold in folds]
        ),
        "mean_oracle_f1_gap": _finite_mean(
            [fold["oracle_f1_gap"] for fold in folds]
        ),
        "top_5pct_hit_rate": _finite_mean(
            [float(fold["top_5pct_hit"]) for fold in folds]
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    parameters = load_active_parameters(args.config.resolve())
    manifest_path = args.sidecar_manifest.resolve()
    manifest = load_sidecar_manifest(manifest_path)
    tasks = args.tasks or list(DEVELOPMENT_TASKS)
    if tasks != list(DEVELOPMENT_TASKS):
        raise ValueError("core ablation requires the four frozen development tasks")
    directories = discover_calibrations(args.calibration_root.resolve(), tasks)
    variant_folds: dict[str, list[dict[str, Any]]] = {
        variant: [] for variant in VARIANTS
    }

    for task_index, directory in enumerate(directories):
        metadata, parent = load_calibration(directory)
        task = str(metadata["task"])
        sidecar, _, _ = _load_task_sidecars(
            manifest=manifest,
            task=task,
            base_metadata=metadata,
        )
        shift_deltas = np.asarray(parent["shift_deltas"], dtype=np.float64)
        band_risks = np.asarray(parent["band_risks"], dtype=np.float64)
        band_covariances = np.asarray(
            parent["band_covariances"], dtype=np.float64
        )
        all_indices = np.arange(shift_deltas.shape[0])
        for heldout in all_indices:
            development = all_indices[all_indices != heldout]
            combined_deltas = np.concatenate(
                [shift_deltas[development], sidecar["shift_deltas"]], axis=0
            )
            combined_risks = np.concatenate(
                [band_risks[development], sidecar["band_risks"]], axis=0
            )
            combined_covariances = np.concatenate(
                [
                    band_covariances[development],
                    sidecar["band_covariances"],
                ],
                axis=0,
            )
            target_delta = shift_deltas[heldout]
            disagreements = _heldout_disagreement(
                band_risks[heldout], band_covariances[heldout]
            )
            truth = 0.5 * band_risks[heldout].sum(axis=1)
            alpha, matching = match_shift_convex_combination(
                combined_deltas,
                target_delta,
                regularization=float(parameters["transport_regularization"]),
                descriptor_floor=float(parameters["descriptor_floor"]),
            )
            transported_risks, transported_covariances = transport_statistics(
                alpha,
                combined_risks,
                combined_covariances,
            )
            zero_covariances = np.zeros_like(transported_covariances)
            zero_prior = np.zeros_like(transported_risks)
            descriptor_rmse = float(matching["descriptor_rmse"])
            spectral_prior = curvature_prior_strength(
                disagreements, descriptor_rmse
            )

            global_disagreements = disagreements.sum(axis=0, keepdims=True)
            global_risks = transported_risks.sum(axis=1, keepdims=True)
            global_covariances = transported_covariances.sum(
                axis=0, keepdims=True
            )
            global_zero_prior = np.zeros_like(global_risks)
            global_prior = curvature_prior_strength(
                global_disagreements, descriptor_rmse
            )

            scores = {
                "global_linear": _recover(
                    global_disagreements,
                    global_zero_prior,
                    np.zeros_like(global_covariances),
                    parameters=parameters,
                    prior_strength=0.0,
                    robust=False,
                    pair_weight_power=0.0,
                ),
                "spectral_linear": _recover(
                    disagreements,
                    zero_prior,
                    zero_covariances,
                    parameters=parameters,
                    prior_strength=0.0,
                    robust=False,
                    pair_weight_power=0.0,
                ),
                "spectral_no_covariance": _recover(
                    disagreements,
                    transported_risks,
                    zero_covariances,
                    parameters=parameters,
                    prior_strength=spectral_prior,
                    robust=True,
                    pair_weight_power=0.0,
                ),
                "global_covariance": _recover(
                    global_disagreements,
                    global_risks,
                    global_covariances,
                    parameters=parameters,
                    prior_strength=global_prior,
                    robust=True,
                ),
                "spectral_covariance_no_robust": _recover(
                    disagreements,
                    transported_risks,
                    transported_covariances,
                    parameters=parameters,
                    prior_strength=spectral_prior,
                    robust=False,
                ),
                "spectral_covariance": _recover(
                    disagreements,
                    transported_risks,
                    transported_covariances,
                    parameters=parameters,
                    prior_strength=spectral_prior,
                    robust=True,
                ),
            }
            generator = np.random.default_rng(
                args.bootstrap_seed + task_index * 1009 + int(heldout)
            )
            sampled_sets = [
                generator.integers(
                    0,
                    combined_deltas.shape[0],
                    size=combined_deltas.shape[0],
                )
                for _ in range(int(parameters["bootstrap_samples"]))
            ]
            uncertainty_beta = float(parameters["uncertainty_beta"])
            for point_variant in (
                "spectral_no_covariance",
                "global_covariance",
                "spectral_covariance_no_robust",
                "spectral_covariance",
            ):
                uncertainty = _bootstrap_variant_uncertainty(
                    variant=point_variant,
                    sampled_sets=sampled_sets,
                    shift_deltas=combined_deltas,
                    band_risks=combined_risks,
                    band_covariances=combined_covariances,
                    target_delta=target_delta,
                    disagreements=disagreements,
                    parameters=parameters,
                    workers=args.bootstrap_workers,
                )
                scores[f"{point_variant}_uncertainty"] = (
                    scores[point_variant] + uncertainty_beta * uncertainty
                )

            for variant, score in scores.items():
                variant_folds[variant].append(
                    {
                        "task": task,
                        "heldout_shift_name": metadata["shift_specs"][heldout][
                            "name"
                        ],
                        **_fold_metrics(score, truth),
                    }
                )
            print(
                f"task={task} fold={heldout + 1:02d}/{len(all_indices):02d}",
                flush=True,
            )

    result = {
        "schema_version": 1,
        "objective_source": (
            "frozen source-simulated folds and source-labelled sidecars; "
            "no real target labels"
        ),
        "variants": {
            variant: {
                **_aggregate(folds),
                "folds": folds,
            }
            for variant, folds in variant_folds.items()
        },
        "active_parameters": parameters,
        "runtime_seconds": time.perf_counter() - started,
        "sidecar_manifest": str(manifest_path),
        "sidecar_manifest_sha256": sha256_file(manifest_path),
        "label_access_count": 0,
        "protocol_violation_count": 0,
    }
    atomic_json(result, args.output.resolve())
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-root", type=Path, required=True)
    parser.add_argument("--sidecar-manifest", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "configs/search_space.yaml",
    )
    parser.add_argument("--tasks", nargs="*")
    parser.add_argument("--bootstrap-seed", type=int, default=19461)
    parser.add_argument("--bootstrap-workers", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    result = run(parse_args())
    print(
        json.dumps(
            {
                variant: {
                    key: result["variants"][variant][key]
                    for key in (
                        "mean_normalized_regret",
                        "cvar20",
                        "worst_fold_regret",
                        "median_kendall_tau",
                        "top_5pct_hit_rate",
                    )
                }
                for variant in VARIANTS
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
