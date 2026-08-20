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
DEFAULT_COVARIANCE_SHRINKAGE_MODE = "none"
DEFAULT_FIXED_COVARIANCE_GAMMA = 1.0
DEFAULT_COVARIANCE_CONSISTENCY_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
COVARIANCE_SHRINKAGE_MODES = ("none", "support_gate", "fixed", "pair_consistency")
SIDECAR_ARRAY_NAMES = (
    "shift_deltas",
    "band_risks",
    "band_covariances",
)


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


def covariance_support_confidence(
    descriptor_rmse: float,
    *,
    threshold: float = PRIOR_DESCRIPTOR_RMSE_THRESHOLD,
) -> float:
    """Return the hard label-free support confidence used for covariance gating."""

    if not np.isfinite(descriptor_rmse) or descriptor_rmse < 0:
        raise ValueError("descriptor RMSE must be finite and non-negative")
    if threshold <= 0:
        raise ValueError("descriptor RMSE threshold must be positive")
    return 1.0 if descriptor_rmse <= threshold else 0.0


def covariance_shrinkage_gamma(
    descriptor_rmse: float,
    *,
    mode: str = DEFAULT_COVARIANCE_SHRINKAGE_MODE,
    fixed_gamma: float = DEFAULT_FIXED_COVARIANCE_GAMMA,
    threshold: float = PRIOR_DESCRIPTOR_RMSE_THRESHOLD,
) -> float:
    """Choose how strongly transported covariance enters target recovery.

    ``none`` preserves the frozen v2 behavior. ``support_gate`` applies the
    same descriptor-support test used by the transported-risk prior to the
    covariance correction itself. ``fixed`` is the controlled gamma-sweep mode
    needed for support-shrinkage ablations.
    """

    if mode not in COVARIANCE_SHRINKAGE_MODES:
        raise ValueError(f"unknown covariance shrinkage mode: {mode}")
    if mode == "none":
        return 1.0
    if mode == "support_gate":
        return covariance_support_confidence(descriptor_rmse, threshold=threshold)
    if mode == "pair_consistency":
        raise ValueError("pair_consistency gamma requires target disagreements")
    if not np.isfinite(fixed_gamma) or not 0.0 <= fixed_gamma <= 1.0:
        raise ValueError("fixed covariance gamma must lie in [0, 1]")
    return float(fixed_gamma)


def pair_sum_consistency_gamma(
    disagreements: np.ndarray,
    transported_covariances: np.ndarray,
    *,
    grid: tuple[float, ...] = DEFAULT_COVARIANCE_CONSISTENCY_GRID,
) -> tuple[float, list[dict[str, float]]]:
    """Select gamma by projecting corrected pairs onto the pair-sum subspace."""

    disagreement = np.asarray(disagreements, dtype=np.float64)
    covariance = np.asarray(transported_covariances, dtype=np.float64)
    if disagreement.ndim != 3 or disagreement.shape != covariance.shape:
        raise ValueError("disagreements and covariances must share [bands, models, models]")
    band_count, model_count, second_count = disagreement.shape
    if second_count != model_count:
        raise ValueError("pair consistency expects square model-pair matrices")
    if model_count < 3:
        raise ValueError("at least three models are required for pair consistency")
    if not grid:
        raise ValueError("covariance consistency grid must be non-empty")
    if any((not np.isfinite(gamma)) or gamma < 0.0 or gamma > 1.0 for gamma in grid):
        raise ValueError("covariance consistency grid values must lie in [0, 1]")

    first, second = np.triu_indices(model_count, k=1)
    design = np.zeros((first.size, model_count), dtype=np.float64)
    design[np.arange(first.size), first] = 1.0
    design[np.arange(first.size), second] = 1.0
    reports: list[dict[str, float]] = []
    for gamma in grid:
        residual_energy = 0.0
        observation_energy = 0.0
        for band in range(band_count):
            observations = (
                disagreement[band, first, second]
                + 2.0 * float(gamma) * covariance[band, first, second]
            )
            fitted, *_ = np.linalg.lstsq(design, observations, rcond=None)
            residual = observations - design @ fitted
            residual_energy += float(np.sum(residual**2))
            observation_energy += float(np.sum(observations**2))
        normalized = residual_energy / max(observation_energy, 1.0e-12)
        reports.append(
            {
                "gamma": float(gamma),
                "residual_energy": residual_energy,
                "normalized_residual": float(normalized),
            }
        )
    best = min(reports, key=lambda report: (report["normalized_residual"], report["gamma"]))
    return float(best["gamma"]), reports


def shrink_transported_covariances(
    transported_covariances: np.ndarray,
    descriptor_rmse: float,
    *,
    disagreements: np.ndarray | None = None,
    mode: str = DEFAULT_COVARIANCE_SHRINKAGE_MODE,
    fixed_gamma: float = DEFAULT_FIXED_COVARIANCE_GAMMA,
    threshold: float = PRIOR_DESCRIPTOR_RMSE_THRESHOLD,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Scale transported covariance by a label-free support confidence."""

    covariances = np.asarray(transported_covariances, dtype=np.float64)
    consistency_reports: list[dict[str, float]] | None = None
    if mode == "pair_consistency":
        if disagreements is None:
            raise ValueError("pair_consistency mode requires disagreements")
        gamma, consistency_reports = pair_sum_consistency_gamma(
            disagreements,
            covariances,
        )
    else:
        gamma = covariance_shrinkage_gamma(
            descriptor_rmse,
            mode=mode,
            fixed_gamma=fixed_gamma,
            threshold=threshold,
        )
    diagnostics: dict[str, Any] = {
        "mode": mode,
        "gamma": float(gamma),
        "descriptor_rmse": float(descriptor_rmse),
        "threshold": float(threshold),
    }
    if consistency_reports is not None:
        diagnostics["consistency_grid"] = consistency_reports
    return gamma * covariances, diagnostics


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


def _is_feature_mask_grid(metadata: dict[str, Any]) -> bool:
    selected = metadata.get("selected_shifts") or []
    return bool(selected) and all(
        shift.get("calibration_family") == "feature_mask_grid"
        for shift in selected
    )


def _source_node_count(metadata: dict[str, Any]) -> int | None:
    selected = metadata.get("selected_shifts") or []
    counts = {
        int(shift["node_count"])
        for shift in selected
        if "node_count" in shift
    }
    return next(iter(counts)) if len(counts) == 1 else None


def _sidecar_rank_gate_allows(metadata: dict[str, Any]) -> bool:
    if not _is_feature_mask_grid(metadata):
        return True
    source_nodes = _source_node_count(metadata)
    if source_nodes is None:
        raise ValueError("feature-mask sidecar has inconsistent source node counts")
    return source_nodes >= int(metadata["candidate_count"])


def load_calibration_sidecars(
    manifest_path: Path,
    *,
    task: str,
    base_metadata: dict[str, Any],
) -> tuple[dict[str, np.ndarray], list[str], list[dict[str, Any]], list[dict[str, Any]]]:
    """Load a frozen task's source-only calibration sidecars.

    The manifest is deliberately task-generic so the same audited loader can
    serve both the four development tasks and the final 16-task deployment.
    Every artifact is bound to the base candidate ordering, spectral config,
    and parent calibration hash before any arrays are exposed to the selector.
    """

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported calibration-sidecar manifest schema")
    if manifest.get("target_label_access_count") != 0:
        raise RuntimeError("sidecar manifest reports target-label access")
    if manifest.get("protocol_violation_count") != 0:
        raise RuntimeError("sidecar manifest reports a protocol violation")
    if (
        manifest.get("rank_gate")
        != "source_nodes_gte_candidate_count_for_feature_mask_grid"
    ):
        raise ValueError("unexpected covariance rank gate")
    task_entries = manifest.get("tasks", {}).get(task)
    if not isinstance(task_entries, list) or not task_entries:
        raise ValueError(f"sidecar manifest does not freeze task {task}")

    parts: list[dict[str, np.ndarray]] = []
    shift_names: list[str] = []
    accepted: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for entry_index, entry in enumerate(task_entries):
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
        if not _sidecar_rank_gate_allows(metadata):
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
            if not set(SIDECAR_ARRAY_NAMES).issubset(artifact.files):
                raise ValueError(f"sidecar arrays are incomplete: {directory}")
            arrays = {
                name: np.asarray(artifact[name])
                for name in SIDECAR_ARRAY_NAMES
            }
        shift_count = int(arrays["shift_deltas"].shape[0])
        if any(arrays[name].shape[0] != shift_count for name in SIDECAR_ARRAY_NAMES):
            raise ValueError(f"sidecar shift dimensions do not align: {directory}")
        selected = metadata.get("selected_shifts") or []
        if selected and len(selected) != shift_count:
            raise ValueError(f"sidecar shift metadata count mismatch: {directory}")
        for shift_index in range(shift_count):
            shift = selected[shift_index] if selected else {}
            family = str(shift.get("calibration_family", "source_sidecar"))
            candidate_index = shift.get("candidate_index", shift_index)
            shift_names.append(
                f"sidecar{entry_index}:{family}:{candidate_index}"
            )
        parts.append(arrays)
        accepted.append(
            {
                "task": task,
                "path": str(directory),
                "artifact_sha256": expected_hash,
                "shift_count": shift_count,
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
    return combined, shift_names, accepted, skipped


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
    covariance_shrinkage_mode: str = DEFAULT_COVARIANCE_SHRINKAGE_MODE,
    fixed_covariance_gamma: float = DEFAULT_FIXED_COVARIANCE_GAMMA,
    support_rmse_threshold: float = PRIOR_DESCRIPTOR_RMSE_THRESHOLD,
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
    descriptor_rmse = float(match_diagnostics["descriptor_rmse"])
    shrunk_covariances, shrinkage_diagnostics = shrink_transported_covariances(
        transported_covariances,
        descriptor_rmse,
        disagreements=disagreements,
        mode=covariance_shrinkage_mode,
        fixed_gamma=fixed_covariance_gamma,
        threshold=support_rmse_threshold,
    )
    recovered, recovery_diagnostics = corrected_band_risk_recovery(
        disagreements,
        transported_risks,
        shrunk_covariances,
        ridge=risk_ridge,
        pair_weight_power=pair_weight_power,
        # Normalize by the pair design's weakest identifiable curvature.  A
        # two-standard-deviation descriptor gate disables the prior when the
        # target lies outside the source-simulation support; this is fully
        # label-free at deployment time.
        prior_strength=curvature_prior_strength(
            disagreements,
            descriptor_rmse,
            threshold=support_rmse_threshold,
        ),
    )
    return recovered, alpha, {
        "matching": match_diagnostics,
        "covariance_shrinkage": shrinkage_diagnostics,
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
    covariance_shrinkage_mode: str = DEFAULT_COVARIANCE_SHRINKAGE_MODE,
    fixed_covariance_gamma: float = DEFAULT_FIXED_COVARIANCE_GAMMA,
    support_rmse_threshold: float = PRIOR_DESCRIPTOR_RMSE_THRESHOLD,
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
            covariance_shrinkage_mode=covariance_shrinkage_mode,
            fixed_covariance_gamma=fixed_covariance_gamma,
            support_rmse_threshold=support_rmse_threshold,
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

    shift_names = [spec["name"] for spec in calibration_metadata["shift_specs"]]
    sidecar_manifest = getattr(args, "sidecar_manifest", None)
    accepted_sidecars: list[dict[str, Any]] = []
    skipped_sidecars: list[dict[str, Any]] = []
    sidecar_manifest_hash: str | None = None
    if sidecar_manifest is not None:
        sidecar_manifest = Path(sidecar_manifest).resolve()
        sidecar_arrays, sidecar_shift_names, accepted_sidecars, skipped_sidecars = (
            load_calibration_sidecars(
                sidecar_manifest,
                task=args.task,
                base_metadata=calibration_metadata,
            )
        )
        arrays = dict(arrays)
        for name in SIDECAR_ARRAY_NAMES:
            arrays[name] = np.concatenate(
                [np.asarray(arrays[name]), sidecar_arrays[name]], axis=0
            )
        shift_names.extend(sidecar_shift_names)
        sidecar_manifest_hash = sha256_file(sidecar_manifest)

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
    covariance_shrinkage_mode = getattr(
        args,
        "covariance_shrinkage_mode",
        DEFAULT_COVARIANCE_SHRINKAGE_MODE,
    )
    fixed_covariance_gamma = float(
        getattr(args, "fixed_covariance_gamma", DEFAULT_FIXED_COVARIANCE_GAMMA)
    )
    support_rmse_threshold = float(
        getattr(args, "support_rmse_threshold", PRIOR_DESCRIPTOR_RMSE_THRESHOLD)
    )
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
        covariance_shrinkage_mode=covariance_shrinkage_mode,
        fixed_covariance_gamma=fixed_covariance_gamma,
        support_rmse_threshold=support_rmse_threshold,
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
        covariance_shrinkage_mode=covariance_shrinkage_mode,
        fixed_covariance_gamma=fixed_covariance_gamma,
        support_rmse_threshold=support_rmse_threshold,
    )
    robust_score = point_estimate + args.uncertainty_beta * uncertainty
    scores = {
        identifier: float(robust_score[index])
        for index, identifier in enumerate(identifiers)
    }
    optimum = float(robust_score.min())
    selected = min(identifier for identifier, score in scores.items() if score == optimum)
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
            "covariance_shrinkage_mode": covariance_shrinkage_mode,
            "fixed_covariance_gamma": fixed_covariance_gamma,
            "support_rmse_threshold": support_rmse_threshold,
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_seed": args.bootstrap_seed,
            "uncertainty_beta": args.uncertainty_beta,
            "device": args.device,
        },
        "frame_diagnostics": frame_diagnostics,
        "transport_diagnostics": diagnostics,
        "calibration_sidecars": {
            "manifest": str(sidecar_manifest) if sidecar_manifest is not None else None,
            "manifest_sha256": sidecar_manifest_hash,
            "accepted": accepted_sidecars,
            "skipped": skipped_sidecars,
        },
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
    parser.add_argument(
        "--covariance-shrinkage-mode",
        choices=COVARIANCE_SHRINKAGE_MODES,
        default=DEFAULT_COVARIANCE_SHRINKAGE_MODE,
        help=(
            "how to scale transported covariance before risk recovery; "
            "'none' preserves the frozen v2 selector"
        ),
    )
    parser.add_argument(
        "--fixed-covariance-gamma",
        type=float,
        default=DEFAULT_FIXED_COVARIANCE_GAMMA,
        help="fixed gamma in [0, 1] when --covariance-shrinkage-mode=fixed",
    )
    parser.add_argument(
        "--support-rmse-threshold",
        type=float,
        default=PRIOR_DESCRIPTOR_RMSE_THRESHOLD,
        help="descriptor RMSE threshold shared by the prior and support-gated covariance",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=0)
    parser.add_argument("--bootstrap-seed", type=int, default=8801)
    parser.add_argument("--uncertainty-beta", type=float, default=0.0)
    parser.add_argument(
        "--sidecar-manifest",
        type=Path,
        help="frozen source-only calibration sidecars for this task",
    )
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
