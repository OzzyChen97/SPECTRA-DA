#!/usr/bin/env python3
"""Build or validate one immutable SPECTRA calibration artifact per task."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.trajectory_export.schema import (  # noqa: E402
    candidate_bank_hash,
    discover_candidate_records,
    sha256_file,
)
from shift_simulator.build_calibration import build  # noqa: E402


def discover_tasks(candidate_root: Path) -> list[str]:
    tasks = sorted(
        path.name
        for path in candidate_root.iterdir()
        if path.is_dir() and not path.name.startswith("_")
    )
    if not tasks:
        raise FileNotFoundError(f"no candidate tasks under {candidate_root}")
    return tasks


def validate_calibration(
    directory: Path,
    *,
    task: str,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    metadata_path = directory / "metadata.json"
    arrays_path = directory / "calibration.npz"
    if not metadata_path.is_file() or not arrays_path.is_file():
        raise FileNotFoundError(f"incomplete calibration artifact: {directory}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_ids = [record["metadata"]["candidate_id"] for record in records]
    checks = {
        "task": task,
        "candidate_bank_sha256": candidate_bank_hash(records),
        "candidate_count": len(records),
        "target_label_access_count": 0,
        "source_label_access_count": 1,
        "physical_gpu": 7,
        "logical_device": "cuda:0",
    }
    for name, expected in checks.items():
        if metadata.get(name) != expected:
            raise ValueError(
                f"calibration mismatch in {directory}: "
                f"{name}={metadata.get(name)!r}, expected {expected!r}"
            )
    if metadata.get("candidate_ids") != expected_ids:
        raise ValueError(f"calibration candidate ordering mismatch: {directory}")
    if not isinstance(metadata.get("runtime_seconds"), (int, float)) or not np.isfinite(
        metadata["runtime_seconds"]
    ) or float(metadata["runtime_seconds"]) <= 0.0:
        raise ValueError(f"calibration runtime is invalid: {directory}")
    shift_specs = metadata.get("shift_specs", [])
    if metadata.get("shift_count") != len(shift_specs) or len(shift_specs) < 6:
        raise ValueError(f"calibration shift registry is incomplete: {directory}")
    shift_names = [spec.get("name", "") for spec in shift_specs]
    required_shift_families = {
        "feature": any(name.startswith(("feature_mask", "feature_noise")) for name in shift_names),
        "structure": any(name.startswith("edge_dropout") for name in shift_names),
        "homophily": any(name.startswith("homophily") for name in shift_names),
        "label_prior": any(name.startswith("label_prior") for name in shift_names),
        "conditional_structure": "conditional_structure" in shift_names,
    }
    missing_families = sorted(
        family for family, present in required_shift_families.items() if not present
    )
    if missing_families:
        raise ValueError(
            f"calibration shift families are missing in {directory}: {missing_families}"
        )
    expected_hash = metadata.get("artifact_sha256", {}).get(arrays_path.name)
    if expected_hash != sha256_file(arrays_path):
        raise ValueError(f"calibration hash mismatch: {arrays_path}")
    with np.load(arrays_path, allow_pickle=False) as artifact:
        expected_keys = {
            "source_descriptor",
            "target_descriptor",
            "target_delta",
            "shift_descriptors",
            "shift_deltas",
            "band_risks",
            "band_covariances",
            "chebyshev_coefficients",
        }
        if set(artifact.files) != expected_keys:
            raise ValueError(
                f"calibration array schema mismatch in {arrays_path}: {sorted(artifact.files)}"
            )
        arrays = {name: np.asarray(artifact[name]) for name in artifact.files}
    if any(not np.isfinite(array).all() for array in arrays.values()):
        raise ValueError(f"calibration contains non-finite arrays: {arrays_path}")

    shift_count = int(metadata["shift_count"])
    candidate_count = len(expected_ids)
    spectral_config = metadata.get("spectral_config", {})
    band_count = int(spectral_config.get("num_bands", 0))
    descriptor_count = len(metadata.get("descriptor_names", []))
    if shift_count <= 0 or candidate_count < 3 or band_count <= 0 or descriptor_count <= 0:
        raise ValueError(f"invalid calibration dimensions in metadata: {directory}")
    expected_shapes = {
        "source_descriptor": (descriptor_count,),
        "target_descriptor": (descriptor_count,),
        "target_delta": (descriptor_count,),
        "shift_descriptors": (shift_count, descriptor_count),
        "shift_deltas": (shift_count, descriptor_count),
        "band_risks": (shift_count, candidate_count, band_count),
        "band_covariances": (
            shift_count,
            band_count,
            candidate_count,
            candidate_count,
        ),
    }
    for name, shape in expected_shapes.items():
        if arrays[name].shape != shape:
            raise ValueError(
                f"calibration array shape mismatch for {name}: "
                f"{arrays[name].shape} != {shape}"
            )
    if arrays["chebyshev_coefficients"].shape[0] != band_count:
        raise ValueError(f"Chebyshev coefficient band count mismatch: {arrays_path}")
    risks = arrays["band_risks"]
    covariances = arrays["band_covariances"]
    if float(risks.min()) < -1e-6:
        raise ValueError(f"calibration contains negative spectral risks: {arrays_path}")
    if not np.allclose(covariances, covariances.swapaxes(-1, -2), atol=2e-5, rtol=1e-5):
        raise ValueError(f"calibration covariance is not symmetric: {arrays_path}")
    covariance_diagonal = np.diagonal(covariances, axis1=-2, axis2=-1).transpose(0, 2, 1)
    if not np.allclose(risks, covariance_diagonal, atol=2e-5, rtol=1e-5):
        raise ValueError(f"calibration risk/covariance diagonal mismatch: {arrays_path}")
    if float(metadata.get("frame_diagnostics", {}).get("max_frame_error", 1.0)) > 0.05:
        raise ValueError(f"calibration frame error exceeds guardrail: {directory}")
    return metadata


def existing_calibration(
    output_root: Path,
    *,
    task: str,
    records: list[dict[str, Any]],
) -> Path | None:
    task_root = output_root / task
    directories = sorted(
        path.parent
        for path in task_root.glob("*/metadata.json")
        if (path.parent / "calibration.npz").is_file()
    )
    if not directories:
        return None
    if len(directories) != 1:
        raise ValueError(f"expected at most one calibration for {task}, found {directories}")
    validate_calibration(directories[0], task=task, records=records)
    return directories[0]


def quarantine_temporaries(output_root: Path, task: str) -> list[Path]:
    task_root = output_root / task
    temporaries = sorted(path for path in task_root.glob(".*.tmp-*") if path.is_dir())
    quarantined = []
    for path in temporaries:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        destination = output_root / "_failed" / task / f"{path.name}.{stamp}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        path.replace(destination)
        quarantined.append(destination)
    return quarantined


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--task", action="append", dest="tasks")
    parser.add_argument("--num-bands", type=int, default=3)
    parser.add_argument("--centers")
    parser.add_argument("--sigma", type=float, default=0.55)
    parser.add_argument("--cheb-order", type=int, default=8)
    parser.add_argument("--shift-seed", type=int, default=7400)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "7":
        raise SystemExit("calibration suite requires CUDA_VISIBLE_DEVICES=7")
    candidate_root = args.candidate_root.resolve()
    output_root = args.output_root.resolve()
    tasks = args.tasks or discover_tasks(candidate_root)
    completed = []
    skipped = []
    failed = []
    for task in tasks:
        try:
            records = discover_candidate_records(candidate_root, task)
            existing = existing_calibration(
                output_root,
                task=task,
                records=records,
            )
            if existing is not None:
                print(f"skip validated {task} -> {existing}", flush=True)
                skipped.append(task)
                continue
            for destination in quarantine_temporaries(output_root, task):
                print(f"quarantined incomplete calibration {destination}", flush=True)
            result = build(
                SimpleNamespace(
                    candidate_root=candidate_root,
                    task=task,
                    output_root=output_root,
                    num_bands=args.num_bands,
                    centers=args.centers,
                    sigma=args.sigma,
                    cheb_order=args.cheb_order,
                    shift_seed=args.shift_seed,
                    device=args.device,
                )
            )
            directory = Path(result["path"])
            validate_calibration(directory, task=task, records=records)
            completed.append(task)
            print(f"completed {task} -> {directory}", flush=True)
        except Exception as error:
            failed.append({"task": task, "error": f"{type(error).__name__}: {error}"})
            print(f"failed {task}: {type(error).__name__}: {error}", flush=True)
            if not args.continue_on_error:
                raise
    report = {
        "task_count": len(tasks),
        "completed": completed,
        "skipped": skipped,
        "failed": failed,
        "label_access_count": 0,
        "protocol_violation_count": 0 if not failed else len(failed),
        "physical_gpu": 7,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
