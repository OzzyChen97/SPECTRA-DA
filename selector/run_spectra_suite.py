#!/usr/bin/env python3
"""Generate SPECTRA selector outputs for every task in a frozen candidate bank."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.trajectory_export.schema import atomic_json  # noqa: E402
from selector.spectra_cal import select as select_calibrated  # noqa: E402
from selector.spectra_static import select as select_static  # noqa: E402


def discover_tasks(candidate_root: Path) -> list[str]:
    tasks = sorted(
        path.name
        for path in candidate_root.iterdir()
        if path.is_dir() and not path.name.startswith("_")
    )
    if not tasks:
        raise FileNotFoundError(f"no task directories under {candidate_root}")
    return tasks


def calibration_directory(root: Path, task: str) -> Path:
    directories = sorted(
        metadata.parent
        for metadata in (root / task).glob("*/metadata.json")
        if (metadata.parent / "calibration.npz").is_file()
    )
    if len(directories) != 1:
        raise ValueError(f"expected one calibration directory for {task}, found {len(directories)}")
    return directories[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--calibration-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=REPO / "configs" / "search_space.yaml")
    parser.add_argument("--task", action="append", dest="tasks")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip-static", action="store_true")
    parser.add_argument("--skip-cal", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidate_root = args.candidate_root.resolve()
    calibration_root = args.calibration_root.resolve()
    output_root = args.output_root.resolve()
    config = yaml.safe_load(args.config.resolve().read_text(encoding="utf-8"))
    active = config["active_parameters"]
    tasks = args.tasks or discover_tasks(candidate_root)
    for task in tasks:
        destination = output_root / task
        if not args.skip_static:
            for prediction_kind in ("hard", "soft"):
                selector_name = f"spectra_static_{prediction_kind}_nnls"
                print(f"[spectra] task={task} selector={selector_name}", flush=True)
                result = select_static(
                    candidate_root=candidate_root,
                    task=task,
                    num_bands=3,
                    sigma=0.55,
                    cheb_order=8,
                    centers=None,
                    prediction_kind=prediction_kind,
                    estimator="nnls",
                    ridge=1e-6,
                    device_name=args.device,
                )
                output = destination / f"{selector_name}.json"
                atomic_json(result, output)
                print(json.dumps({"task": task, "selector": result["selector"], "output": str(output)}), flush=True)
        if not args.skip_cal:
            calibration_dir = calibration_directory(calibration_root, task)
            for spectral_mode in ("global", "banded"):
                requested_name = "spectra_global_cal" if spectral_mode == "global" else "spectra_cal"
                print(f"[spectra] task={task} selector={requested_name}", flush=True)
                namespace = argparse.Namespace(
                    task=task,
                    device=args.device,
                    candidate_root=candidate_root,
                    calibration_dir=calibration_dir,
                    transport_regularization=float(active["transport_regularization"]),
                    descriptor_floor=float(active["descriptor_floor"]),
                    risk_ridge=float(active["risk_ridge"]),
                    pair_weight_power=float(active["pair_weight_power"]),
                    bootstrap_samples=int(active["bootstrap_samples"]),
                    bootstrap_seed=8801,
                    uncertainty_beta=float(active["uncertainty_beta"]),
                    spectral_mode=spectral_mode,
                )
                result = select_calibrated(namespace)
                output = destination / f"{result['selector']}.json"
                atomic_json(result, output)
                print(json.dumps({"task": task, "selector": result["selector"], "output": str(output)}), flush=True)


if __name__ == "__main__":
    main()
