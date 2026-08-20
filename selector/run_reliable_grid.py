#!/usr/bin/env python3
"""Generate a pre-registered grid of reliability-aware fused selectors."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from selector.reliable_selection import (  # noqa: E402
    FUSION_MODES,
    atomic_json,
    load_selection,
    reliable_rank_fusion,
)
from selector.run_reliable_suite import discover_paired_tasks  # noqa: E402


def parse_float_list(text: str) -> list[float]:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one numeric value")
    return values


def format_weight(value: float) -> str:
    scaled = round(value * 100)
    if abs(value * 100 - scaled) < 1.0e-9:
        return f"{int(scaled):03d}"
    return f"{value:.3g}".replace("-", "m").replace(".", "p")


def selector_name(
    prefix: str,
    uncertainty: float,
    transfer: float,
    shrinkage: float,
    temperature: float,
    fusion_mode: str,
    shortlist_fraction: float,
) -> str:
    base = (
        f"{prefix}_uw{format_weight(uncertainty)}"
        f"_tw{format_weight(transfer)}"
        f"_cs{format_weight(shrinkage)}"
        f"_ct{format_weight(temperature)}"
    )
    if fusion_mode == "rank_fusion":
        return base
    mode_code = {
        "transfer_shortlist_spectra_rerank": "tsr",
        "spectra_shortlist_transfer_rerank": "str",
        "support_adaptive": "sa",
    }[fusion_mode]
    return f"{base}_{mode_code}_sf{format_weight(shortlist_fraction)}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-root", type=Path, required=True)
    parser.add_argument(
        "--spectra-root",
        type=Path,
        help="optional root for SPECTRA JSON files; defaults to --selection-root",
    )
    parser.add_argument(
        "--transfer-root",
        type=Path,
        help="optional root for Transfer Score JSON files; defaults to --selection-root",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--spectra-selector", default="spectra_robust")
    parser.add_argument("--transfer-selector", default="transfer_score")
    parser.add_argument("--output-prefix", default="spectra_reliable")
    parser.add_argument("--task", action="append", dest="tasks")
    parser.add_argument("--uncertainty-weights", type=parse_float_list, default=parse_float_list("0,0.1,0.5,1"))
    parser.add_argument("--transfer-score-weights", type=parse_float_list, default=parse_float_list("0,0.25,0.5,0.75,1"))
    parser.add_argument("--covariance-shrinkages", type=parse_float_list, default=parse_float_list("0,0.25,0.5,0.75"))
    parser.add_argument("--calibration-temperatures", type=parse_float_list, default=parse_float_list("1"))
    parser.add_argument(
        "--fusion-modes",
        default="rank_fusion",
        help="comma-separated fusion modes",
    )
    parser.add_argument("--shortlist-fractions", type=parse_float_list, default=parse_float_list("0.2"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selection_root = args.selection_root.resolve()
    spectra_root = (args.spectra_root or selection_root).resolve()
    transfer_root = (args.transfer_root or selection_root).resolve()
    output_root = args.output_root.resolve()
    tasks = args.tasks or discover_paired_tasks(
        spectra_root,
        transfer_root,
        args.spectra_selector,
        args.transfer_selector,
    )
    grid = list(
        itertools.product(
            args.uncertainty_weights,
            args.transfer_score_weights,
            args.covariance_shrinkages,
            args.calibration_temperatures,
            [item.strip() for item in args.fusion_modes.split(",") if item.strip()],
            args.shortlist_fractions,
        )
    )
    for *_, fusion_mode, _ in grid:
        if fusion_mode not in FUSION_MODES:
            raise ValueError(f"unknown fusion mode: {fusion_mode}")
    selector_names = [
        selector_name(
            args.output_prefix,
            uncertainty,
            transfer,
            shrinkage,
            temperature,
            fusion_mode,
            shortlist_fraction,
        )
        for uncertainty, transfer, shrinkage, temperature, fusion_mode, shortlist_fraction in grid
    ]
    if len(selector_names) != len(set(selector_names)):
        raise ValueError("grid contains duplicate selector names after numeric formatting")
    manifest = {
        "schema_version": 1,
        "selection_root": str(selection_root),
        "spectra_root": str(spectra_root),
        "transfer_root": str(transfer_root),
        "spectra_selector": args.spectra_selector,
        "transfer_selector": args.transfer_selector,
        "tasks": tasks,
        "selector_count": len(grid),
        "selectors": [],
        "protocol": {
            "label_free_post_selector": True,
            "allowed_knobs": [
                "uncertainty_weight",
                "transfer_score_weight",
                "covariance_shrinkage",
                "calibration_temperature",
                "fusion_mode",
                "shortlist_fraction",
            ],
            "forbidden": [
                "target labels",
                "new shift types",
                "new spectral filters",
                "new recovery solvers",
            ],
        },
        "label_access_count": 0,
        "protocol_violation_count": 0,
        "contains_target_labels": False,
    }
    for (
        uncertainty,
        transfer,
        shrinkage,
        temperature,
        fusion_mode,
        shortlist_fraction,
    ), name in zip(grid, selector_names):
        manifest["selectors"].append(
            {
                "selector": name,
                "uncertainty_weight": float(uncertainty),
                "transfer_score_weight": float(transfer),
                "covariance_shrinkage": float(shrinkage),
                "calibration_temperature": float(temperature),
                "fusion_mode": fusion_mode,
                "shortlist_fraction": float(shortlist_fraction),
            }
        )
        for task in tasks:
            spectra_path = spectra_root / task / f"{args.spectra_selector}.json"
            transfer_path = transfer_root / task / f"{args.transfer_selector}.json"
            result = reliable_rank_fusion(
                load_selection(spectra_path),
                load_selection(transfer_path),
                uncertainty_weight=float(uncertainty),
                transfer_score_weight=float(transfer),
                covariance_shrinkage=float(shrinkage),
                calibration_temperature=float(temperature),
                selector_name=name,
                fusion_mode=fusion_mode,
                shortlist_fraction=float(shortlist_fraction),
            )
            atomic_json(result, output_root / task / f"{name}.json")
    atomic_json(manifest, output_root / "reliable_grid_manifest.json")
    print(
        json.dumps(
            {
                "task_count": len(tasks),
                "selector_count": len(grid),
                "output_root": str(output_root),
                "manifest": str(output_root / "reliable_grid_manifest.json"),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
