#!/usr/bin/env python3
"""Descriptor-to-correction diagnostic and diagonal metric learning.

This tool checks whether a shift descriptor predicts the covariance correction
that actually changes recovered model risks.  It uses only source-simulated
calibration artifacts: band risks, band covariances, and shift descriptors.
It does not read real target labels or final sealed evaluation outputs.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import lsq_linear
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from covariance_transport import corrected_band_risk_recovery  # noqa: E402
from scripts.trajectory_export.schema import atomic_json, sha256_file  # noqa: E402


TRANSFORM_CHOICES = ("none", "signed_log1p")


def load_calibration(directory: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    metadata_path = directory / "metadata.json"
    arrays_path = directory / "calibration.npz"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_hash = metadata.get("artifact_sha256", {}).get(arrays_path.name)
    if expected_hash is not None and sha256_file(arrays_path) != expected_hash:
        raise ValueError(f"calibration hash mismatch: {arrays_path}")
    if int(metadata.get("target_label_access_count", 0)) != 0:
        raise RuntimeError("calibration artifact reports target-label access")
    if int(metadata.get("protocol_violation_count", 0)) != 0:
        raise RuntimeError("calibration artifact reports protocol violations")
    with np.load(arrays_path, allow_pickle=False) as artifact:
        arrays = {name: np.asarray(artifact[name]) for name in artifact.files}
    return metadata, arrays


def transform_descriptor_values(values: np.ndarray, *, transform: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if transform == "none":
        return matrix.copy()
    if transform == "signed_log1p":
        return np.sign(matrix) * np.log1p(np.abs(matrix))
    raise ValueError(f"unknown descriptor transform: {transform}")


def robust_scale_descriptors(
    descriptors: np.ndarray,
    *,
    transform: str = "none",
    robust_scale: bool = True,
    clip: float = 8.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    if clip <= 0:
        raise ValueError("clip must be positive")
    transformed = transform_descriptor_values(descriptors, transform=transform)
    if not robust_scale:
        return transformed, {
            "transform": transform,
            "robust_scale": False,
            "clip": None,
        }
    finite = np.isfinite(transformed)
    nan_marked = np.where(finite, transformed, np.nan)
    center = np.nanmedian(nan_marked, axis=0)
    center = np.where(np.isfinite(center), center, 0.0)
    filled = np.where(finite, transformed, center)
    mad = np.nanmedian(np.abs(np.where(finite, transformed - center, np.nan)), axis=0)
    mad = np.where(np.isfinite(mad), mad, 0.0)
    scale = 1.4826 * mad
    safe_scale = np.where(scale > 1.0e-12, scale, 1.0)
    scaled = (filled - center) / safe_scale
    clipped = np.clip(scaled, -clip, clip)
    return clipped, {
        "transform": transform,
        "robust_scale": True,
        "clip": float(clip),
        "zero_mad_dimension_count": int(np.count_nonzero(scale <= 1.0e-12)),
        "filled_nonfinite_count": int(np.count_nonzero(~finite)),
    }


def descriptor_dimension_audit(
    raw_descriptors: np.ndarray,
    processed_descriptors: np.ndarray,
    *,
    descriptor_names: list[str],
    target_descriptor: np.ndarray | None = None,
    transform: str = "none",
    robust_scale: bool = True,
    clip: float = 8.0,
) -> list[dict[str, Any]]:
    raw = np.asarray(raw_descriptors, dtype=np.float64)
    processed = np.asarray(processed_descriptors, dtype=np.float64)
    if raw.ndim != 2 or processed.shape != raw.shape:
        raise ValueError("descriptor audit expects aligned raw and processed matrices")
    finite = np.isfinite(raw)
    raw_filled = np.where(finite, raw, np.nan)
    processed_pair_shares = np.zeros(raw.shape[1], dtype=np.float64)
    if raw.shape[0] >= 2:
        pairwise_squared, _ = pairwise_squared_rows(processed)
        pairwise_total = np.sum(pairwise_squared, axis=1)
        valid = pairwise_total > 1.0e-12
        if np.any(valid):
            processed_pair_shares = np.mean(pairwise_squared[valid] / pairwise_total[valid, None], axis=0)

    target_shares = np.full(raw.shape[1], np.nan, dtype=np.float64)
    if target_descriptor is not None:
        target = np.asarray(target_descriptor, dtype=np.float64)
        if target.shape != (raw.shape[1],):
            raise ValueError("target descriptor dimension mismatch")
        transformed_bank = transform_descriptor_values(raw, transform=transform)
        transformed_target = transform_descriptor_values(target[None, :], transform=transform)[0]
        if robust_scale:
            finite = np.isfinite(transformed_bank)
            nan_marked = np.where(finite, transformed_bank, np.nan)
            center = np.nanmedian(nan_marked, axis=0)
            center = np.where(np.isfinite(center), center, 0.0)
            filled_bank = np.where(finite, transformed_bank, center)
            mad = np.nanmedian(
                np.abs(np.where(finite, transformed_bank - center, np.nan)),
                axis=0,
            )
            mad = np.where(np.isfinite(mad), mad, 0.0)
            scale = 1.4826 * mad
            safe_scale = np.where(scale > 1.0e-12, scale, 1.0)
            bank_processed = np.clip((filled_bank - center) / safe_scale, -clip, clip)
            target_processed = np.clip((transformed_target - center) / safe_scale, -clip, clip)
        else:
            bank_processed = transformed_bank
            target_processed = transformed_target
        squared = (bank_processed - target_processed[None, :]) ** 2
        nearest_index = int(np.argmin(np.sum(squared, axis=1)))
        nearest_squared = squared[nearest_index]
        total = float(np.sum(nearest_squared))
        if total > 1.0e-12:
            target_shares = nearest_squared / total

    rows = []
    for index, name in enumerate(descriptor_names):
        column = raw_filled[:, index]
        finite_column = column[np.isfinite(column)]
        if finite_column.size == 0:
            stats = {
                "min": None,
                "median": None,
                "mad": None,
                "p95": None,
                "max": None,
            }
        else:
            median = float(np.median(finite_column))
            stats = {
                "min": float(np.min(finite_column)),
                "median": median,
                "mad": float(np.median(np.abs(finite_column - median))),
                "p95": float(np.quantile(finite_column, 0.95)),
                "max": float(np.max(finite_column)),
            }
        target_share = target_shares[index]
        rows.append(
            {
                "name": name,
                **stats,
                "missing_or_nonfinite_count": int(np.count_nonzero(~finite[:, index])),
                "pairwise_distance_share_mean": float(processed_pair_shares[index]),
                "target_nearest_distance_share": None
                if not np.isfinite(target_share)
                else float(target_share),
            }
        )
    return rows


def heldout_disagreement(
    band_risks: np.ndarray,
    band_covariances: np.ndarray,
) -> np.ndarray:
    risks = np.asarray(band_risks, dtype=np.float64)
    covariances = np.asarray(band_covariances, dtype=np.float64)
    if risks.ndim != 2:
        raise ValueError("band risks must have shape [models, bands]")
    model_count, band_count = risks.shape
    if covariances.shape != (band_count, model_count, model_count):
        raise ValueError("band covariance shape mismatch")
    disagreements = np.empty_like(covariances)
    for band in range(band_count):
        matrix = risks[:, band, None] + risks[None, :, band] - 2.0 * covariances[band]
        disagreements[band] = np.maximum(matrix, 0.0)
        np.fill_diagonal(disagreements[band], 0.0)
    return disagreements


def zero_covariance_recovery(
    band_risks: np.ndarray,
    band_covariances: np.ndarray,
) -> np.ndarray:
    disagreements = heldout_disagreement(band_risks, band_covariances)
    recovered, _ = corrected_band_risk_recovery(
        disagreements,
        np.asarray(band_risks, dtype=np.float64),
        np.zeros_like(disagreements),
        ridge=0.0,
        pair_weight_power=0.0,
        prior_strength=0.0,
        robust=False,
    )
    return 0.5 * recovered.sum(axis=1)


def correction_vectors(
    band_risks: np.ndarray,
    band_covariances: np.ndarray,
) -> np.ndarray:
    risks = np.asarray(band_risks, dtype=np.float64)
    covariances = np.asarray(band_covariances, dtype=np.float64)
    if risks.ndim != 3:
        raise ValueError("band risks must have shape [shifts, models, bands]")
    shift_count, model_count, band_count = risks.shape
    if covariances.shape != (shift_count, band_count, model_count, model_count):
        raise ValueError("band covariance arrays do not align")
    corrections = []
    for shift_index in range(shift_count):
        true_risk = 0.5 * risks[shift_index].sum(axis=1)
        zero_risk = zero_covariance_recovery(
            risks[shift_index],
            covariances[shift_index],
        )
        corrections.append(true_risk - zero_risk)
    return np.stack(corrections, axis=0)


def pairwise_squared_rows(values: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]]]:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 2:
        raise ValueError("expected at least two descriptor/correction rows")
    rows = []
    pairs = []
    for i in range(matrix.shape[0]):
        for j in range(i + 1, matrix.shape[0]):
            rows.append((matrix[i] - matrix[j]) ** 2)
            pairs.append((i, j))
    return np.stack(rows, axis=0), pairs


def pairwise_l2_distances(values: np.ndarray) -> np.ndarray:
    squared, _ = pairwise_squared_rows(values)
    return np.sqrt(np.sum(squared, axis=1))


def finite_spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.size < 2 or right.size < 2:
        return None
    statistic = spearmanr(left, right, nan_policy="omit").statistic
    value = float(statistic)
    return value if np.isfinite(value) else None


def fit_diagonal_metric(
    descriptors: np.ndarray,
    corrections: np.ndarray,
    *,
    ridge: float = 1.0e-6,
) -> tuple[np.ndarray, np.ndarray]:
    descriptor_differences, _ = pairwise_squared_rows(descriptors)
    correction_targets = np.sum(pairwise_squared_rows(corrections)[0], axis=1)
    if ridge < 0:
        raise ValueError("ridge must be non-negative")
    design = descriptor_differences
    target = correction_targets
    descriptor_dimension = design.shape[1]
    if ridge > 0:
        design = np.vstack(
            [
                design,
                np.sqrt(ridge) * np.eye(descriptor_dimension, dtype=np.float64),
            ]
        )
        target = np.concatenate([target, np.zeros(descriptor_dimension, dtype=np.float64)])
    result = lsq_linear(design, target, bounds=(0.0, np.inf), lsmr_tol="auto")
    if not result.success:
        raise RuntimeError(f"diagonal metric fit failed: {result.message}")
    weights = np.asarray(result.x, dtype=np.float64)
    learned_distances = np.sqrt(np.maximum(pairwise_squared_rows(descriptors)[0] @ weights, 0.0))
    return weights, learned_distances


def shift_families(metadata: dict[str, Any], shift_count: int) -> list[str]:
    specs = metadata.get("shift_specs") or []
    families = []
    for index in range(shift_count):
        name = str(specs[index].get("name", f"shift_{index}")) if index < len(specs) else f"shift_{index}"
        families.append(shift_family_from_name(name))
    return families


def shift_family_from_name(name: str) -> str:
    normalized = name.lower().replace("-", "_")
    explicit_prefixes = (
        "feature_mask",
        "feature_noise",
        "feature_scaling",
        "edge_dropout",
        "edge_drop",
        "edge_rewire",
        "degree_preserving",
        "homophily",
        "label_prior",
        "conditional_structure",
        "conditional_edge",
        "community_mask",
    )
    for prefix in explicit_prefixes:
        if normalized.startswith(prefix) or prefix in normalized:
            return prefix
    parts = [part for part in normalized.split("_") if part]
    return parts[0] if parts else "unknown"


def leave_one_family_reports(
    descriptors: np.ndarray,
    corrections: np.ndarray,
    families: list[str],
    *,
    ridge: float,
) -> list[dict[str, Any]]:
    unique_families = sorted(set(families))
    reports = []
    for family in unique_families:
        train = np.asarray([value != family for value in families], dtype=bool)
        test = ~train
        if train.sum() < 3 or test.sum() < 2:
            reports.append(
                {
                    "family": family,
                    "status": "skipped",
                    "train_count": int(train.sum()),
                    "test_count": int(test.sum()),
                }
            )
            continue
        weights, _ = fit_diagonal_metric(
            descriptors[train],
            corrections[train],
            ridge=ridge,
        )
        test_descriptor_diffs, _ = pairwise_squared_rows(descriptors[test])
        test_correction_distances = pairwise_l2_distances(corrections[test])
        learned_distances = np.sqrt(np.maximum(test_descriptor_diffs @ weights, 0.0))
        reports.append(
            {
                "family": family,
                "status": "evaluated",
                "train_count": int(train.sum()),
                "test_count": int(test.sum()),
                "spearman_descriptor_vs_correction": finite_spearman(
                    learned_distances,
                    test_correction_distances,
                ),
                "nonzero_weight_count": int(np.count_nonzero(weights > 1.0e-12)),
            }
        )
    return reports


def evaluate_descriptor_metric(
    descriptors: np.ndarray,
    corrections: np.ndarray,
    *,
    descriptor_names: list[str],
    families: list[str],
    ridge: float = 1.0e-6,
) -> dict[str, Any]:
    descriptor_values = np.asarray(descriptors, dtype=np.float64)
    correction_values = np.asarray(corrections, dtype=np.float64)
    if descriptor_values.ndim != 2 or correction_values.ndim != 2:
        raise ValueError("descriptors and corrections must be matrices")
    if descriptor_values.shape[0] != correction_values.shape[0]:
        raise ValueError("descriptor and correction rows must align")
    if descriptor_values.shape[1] != len(descriptor_names):
        raise ValueError("descriptor_names length does not match descriptor dimension")
    if len(families) != descriptor_values.shape[0]:
        raise ValueError("family count must match descriptor rows")
    if not np.isfinite(descriptor_values).all():
        raise ValueError("descriptors contain non-finite values")
    if not np.isfinite(correction_values).all():
        raise ValueError("corrections contain non-finite values")
    raw_descriptor_distances = pairwise_l2_distances(descriptor_values)
    correction_distances = pairwise_l2_distances(correction_values)
    weights, learned_distances = fit_diagonal_metric(
        descriptor_values,
        correction_values,
        ridge=ridge,
    )
    top_indices = np.argsort(weights)[::-1][: min(10, weights.size)]
    return {
        "shift_count": int(descriptor_values.shape[0]),
        "descriptor_dimension": int(descriptor_values.shape[1]),
        "correction_dimension": int(correction_values.shape[1]),
        "raw_descriptor_correction_spearman": finite_spearman(
            raw_descriptor_distances,
            correction_distances,
        ),
        "learned_metric_correction_spearman": finite_spearman(
            learned_distances,
            correction_distances,
        ),
        "diagonal_metric_ridge": float(ridge),
        "diagonal_metric_weights": [float(value) for value in weights],
        "top_weighted_descriptors": [
            {
                "name": descriptor_names[index],
                "weight": float(weights[index]),
            }
            for index in top_indices
        ],
        "leave_one_family": leave_one_family_reports(
            descriptor_values,
            correction_values,
            families,
            ridge=ridge,
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    metadata, arrays = load_calibration(args.calibration_dir.resolve())
    descriptor_key = args.descriptor_key
    if descriptor_key not in arrays:
        raise KeyError(f"descriptor key not found in calibration artifact: {descriptor_key}")
    raw_descriptors = np.asarray(arrays[descriptor_key], dtype=np.float64)
    if descriptor_key == "shift_deltas":
        descriptor_names = [
            f"delta:{name}"
            for name in metadata.get("descriptor_names", [])
        ]
    else:
        descriptor_names = list(metadata.get("descriptor_names", []))
    if len(descriptor_names) != raw_descriptors.shape[1]:
        descriptor_names = [f"{descriptor_key}_{index}" for index in range(raw_descriptors.shape[1])]
    descriptors, preprocessing = robust_scale_descriptors(
        raw_descriptors,
        transform=args.descriptor_transform,
        robust_scale=not args.no_robust_scale,
        clip=args.robust_clip,
    )
    target_descriptor = None
    if args.target_descriptor_key and args.target_descriptor_key in arrays:
        target_descriptor = np.asarray(arrays[args.target_descriptor_key], dtype=np.float64)
    audit = descriptor_dimension_audit(
        raw_descriptors,
        descriptors,
        descriptor_names=descriptor_names,
        target_descriptor=target_descriptor,
        transform=args.descriptor_transform,
        robust_scale=not args.no_robust_scale,
        clip=args.robust_clip,
    )
    corrections = correction_vectors(
        np.asarray(arrays["band_risks"], dtype=np.float64),
        np.asarray(arrays["band_covariances"], dtype=np.float64),
    )
    families = shift_families(metadata, shift_count=descriptors.shape[0])
    report = evaluate_descriptor_metric(
        descriptors,
        corrections,
        descriptor_names=descriptor_names,
        families=families,
        ridge=args.ridge,
    )
    result = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task": metadata.get("task"),
        "calibration_id": metadata.get("calibration_id"),
        "candidate_bank_sha256": metadata.get("candidate_bank_sha256"),
        "candidate_count": int(metadata.get("candidate_count", corrections.shape[1])),
        "descriptor_key": descriptor_key,
        "target_descriptor_key": args.target_descriptor_key
        if target_descriptor is not None
        else None,
        "descriptor_preprocessing": preprocessing,
        "descriptor_dimension_audit": audit,
        "objective": "descriptor_distance_vs_recovery_relevant_correction_distance",
        "label_access_count": 0,
        "protocol_violation_count": 0,
        **report,
    }
    if args.output is not None:
        atomic_json(result, args.output.resolve())
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--descriptor-key", default="shift_deltas")
    parser.add_argument("--target-descriptor-key", default="target_delta")
    parser.add_argument("--descriptor-transform", choices=TRANSFORM_CHOICES, default="none")
    parser.add_argument("--no-robust-scale", action="store_true")
    parser.add_argument("--robust-clip", type=float, default=8.0)
    parser.add_argument("--ridge", type=float, default=1.0e-6)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    result = run(parse_args())
    print(
        json.dumps(
            {
                "task": result["task"],
                "descriptor_key": result["descriptor_key"],
                "descriptor_preprocessing": result["descriptor_preprocessing"],
                "raw_spearman": result["raw_descriptor_correction_spearman"],
                "learned_spearman": result["learned_metric_correction_spearman"],
                "label_access_count": result["label_access_count"],
                "protocol_violation_count": result["protocol_violation_count"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
