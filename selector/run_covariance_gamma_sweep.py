#!/usr/bin/env python3
"""Run controlled covariance-shrinkage SPECTRA selectors over a task bank.

This runner is intentionally thin: it delegates scoring to
``selector.spectra_cal.select`` and only enumerates pre-registered covariance
interventions while writing distinct selector JSON files plus a manifest. It
does not read target labels or evaluator outputs.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.trajectory_export.schema import atomic_json  # noqa: E402
from selector.run_spectra_suite import calibration_directory, discover_tasks  # noqa: E402
from selector.spectra_cal import select as select_calibrated  # noqa: E402


def parse_float_list(text: str) -> list[float]:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one numeric value")
    for value in values:
        if value < 0.0 or value > 1.0:
            raise argparse.ArgumentTypeError("fixed gamma values must be in [0, 1]")
    return values


def format_gamma(value: float) -> str:
    scaled = round(value * 100)
    if abs(value * 100 - scaled) < 1.0e-9:
        return f"{int(scaled):03d}"
    return f"{value:.3g}".replace("-", "m").replace(".", "p")


def fixed_selector_name(prefix: str, gamma: float) -> str:
    return f"{prefix}_gamma{format_gamma(gamma)}"


def pair_consistency_selector_name(prefix: str) -> str:
    return f"{prefix}_pair_consistency"


def support_gate_selector_name(prefix: str) -> str:
    return f"{prefix}_support_gate"


def build_selector_specs(
    *,
    output_prefix: str,
    fixed_gammas: list[float],
    include_pair_consistency: bool,
    include_support_gate: bool,
) -> list[dict[str, Any]]:
    specs = [
        {
            "selector": fixed_selector_name(output_prefix, gamma),
            "covariance_shrinkage_mode": "fixed",
            "fixed_covariance_gamma": float(gamma),
        }
        for gamma in fixed_gammas
    ]
    if include_pair_consistency:
        specs.append(
            {
                "selector": pair_consistency_selector_name(output_prefix),
                "covariance_shrinkage_mode": "pair_consistency",
                "fixed_covariance_gamma": 1.0,
            }
        )
    if include_support_gate:
        specs.append(
            {
                "selector": support_gate_selector_name(output_prefix),
                "covariance_shrinkage_mode": "support_gate",
                "fixed_covariance_gamma": 1.0,
            }
        )
    selector_names = [spec["selector"] for spec in specs]
    if len(selector_names) != len(set(selector_names)):
        raise ValueError(f"duplicate selector names in sweep: {selector_names}")
    return specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--calibration-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=REPO / "configs" / "search_space.yaml")
    parser.add_argument(
        "--sidecar-manifest",
        type=Path,
        help="optional frozen source-only calibration sidecars shared across tasks",
    )
    parser.add_argument("--task", action="append", dest="tasks")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-prefix", default="spectra_cov")
    parser.add_argument("--fixed-gammas", type=parse_float_list, default=parse_float_list("0,0.5,1"))
    parser.add_argument("--include-pair-consistency", action="store_true")
    parser.add_argument("--include-support-gate", action="store_true")
    parser.add_argument(
        "--support-rmse-threshold",
        type=float,
        help="override the support threshold passed to spectra_cal.py",
    )
    return parser.parse_args()


def run_sweep(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    candidate_root = args.candidate_root.resolve()
    calibration_root = args.calibration_root.resolve()
    output_root = args.output_root.resolve()
    config = yaml.safe_load(args.config.resolve().read_text(encoding="utf-8"))
    active = config["active_parameters"]
    tasks = args.tasks or discover_tasks(candidate_root)
    selector_specs = build_selector_specs(
        output_prefix=args.output_prefix,
        fixed_gammas=args.fixed_gammas,
        include_pair_consistency=args.include_pair_consistency,
        include_support_gate=args.include_support_gate,
    )

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "runner": "selector/run_covariance_gamma_sweep.py",
        "candidate_root": str(candidate_root),
        "calibration_root": str(calibration_root),
        "output_root": str(output_root),
        "config": str(args.config.resolve()),
        "tasks": list(tasks),
        "device": args.device,
        "sidecar_manifest": (
            str(args.sidecar_manifest.resolve())
            if args.sidecar_manifest is not None
            else None
        ),
        "selector_specs": selector_specs,
        "outputs": [],
        "label_access_count": 0,
        "protocol_violation_count": 0,
    }

    for task in tasks:
        calibration_dir = calibration_directory(calibration_root, task)
        for spec in selector_specs:
            selector_name = spec["selector"]
            print(f"[cov-sweep] task={task} selector={selector_name}", flush=True)
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
                sidecar_manifest=(
                    args.sidecar_manifest.resolve()
                    if args.sidecar_manifest is not None
                    else None
                ),
                spectral_mode="banded",
                covariance_shrinkage_mode=spec["covariance_shrinkage_mode"],
                fixed_covariance_gamma=float(spec["fixed_covariance_gamma"]),
                output_selector=selector_name,
            )
            if args.support_rmse_threshold is not None:
                namespace.support_rmse_threshold = float(args.support_rmse_threshold)
            result = select_calibrated(namespace)
            output = output_root / task / f"{result['selector']}.json"
            atomic_json(result, output)
            manifest["label_access_count"] += int(result.get("label_access_count", 0))
            manifest["protocol_violation_count"] += int(
                result.get("protocol_violation_count", 0)
            )
            manifest["outputs"].append(
                {
                    "task": task,
                    "selector": result["selector"],
                    "output": str(output),
                    "covariance_shrinkage_mode": spec["covariance_shrinkage_mode"],
                    "fixed_covariance_gamma": float(spec["fixed_covariance_gamma"]),
                    "selector_runtime_seconds": float(
                        result.get("selector_runtime_seconds", 0.0)
                    ),
                    "label_access_count": int(result.get("label_access_count", 0)),
                    "protocol_violation_count": int(
                        result.get("protocol_violation_count", 0)
                    ),
                }
            )
            print(
                json.dumps(
                    {
                        "task": task,
                        "selector": result["selector"],
                        "output": str(output),
                    }
                ),
                flush=True,
            )

    manifest["wall_clock_seconds"] = time.perf_counter() - started
    atomic_json(manifest, output_root / "covariance_gamma_sweep_manifest.json")
    return manifest


def main() -> None:
    run_sweep(parse_args())


if __name__ == "__main__":
    main()
