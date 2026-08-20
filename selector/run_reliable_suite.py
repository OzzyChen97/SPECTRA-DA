#!/usr/bin/env python3
"""Generate reliability-aware fused selections for a directory of tasks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from selector.reliable_selection import (  # noqa: E402
    atomic_json,
    load_selection,
    reliable_rank_fusion,
)


def discover_tasks(selection_root: Path, spectra_selector: str, transfer_selector: str) -> list[str]:
    return discover_paired_tasks(selection_root, selection_root, spectra_selector, transfer_selector)


def discover_paired_tasks(
    spectra_root: Path,
    transfer_root: Path,
    spectra_selector: str,
    transfer_selector: str,
) -> list[str]:
    tasks = sorted(
        path.name
        for path in spectra_root.iterdir()
        if path.is_dir()
        and (path / f"{spectra_selector}.json").is_file()
        and (transfer_root / path.name / f"{transfer_selector}.json").is_file()
    )
    if not tasks:
        raise FileNotFoundError(
            f"no tasks with {spectra_selector}.json under {spectra_root} "
            f"and {transfer_selector}.json under {transfer_root}"
        )
    return tasks


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
    parser.add_argument("--output-selector", default="spectra_reliable_rank_fusion")
    parser.add_argument("--task", action="append", dest="tasks")
    parser.add_argument("--uncertainty-weight", type=float, default=0.25)
    parser.add_argument("--transfer-score-weight", type=float, default=0.50)
    parser.add_argument("--covariance-shrinkage", type=float, default=0.25)
    parser.add_argument("--calibration-temperature", type=float, default=1.0)
    parser.add_argument(
        "--fusion-mode",
        choices=(
            "rank_fusion",
            "transfer_shortlist_spectra_rerank",
            "spectra_shortlist_transfer_rerank",
            "support_adaptive",
        ),
        default="rank_fusion",
    )
    parser.add_argument("--shortlist-fraction", type=float, default=0.20)
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
    for task in tasks:
        spectra_path = spectra_root / task / f"{args.spectra_selector}.json"
        transfer_path = transfer_root / task / f"{args.transfer_selector}.json"
        result = reliable_rank_fusion(
            load_selection(spectra_path),
            load_selection(transfer_path),
            uncertainty_weight=args.uncertainty_weight,
            transfer_score_weight=args.transfer_score_weight,
            covariance_shrinkage=args.covariance_shrinkage,
            calibration_temperature=args.calibration_temperature,
            selector_name=args.output_selector,
            fusion_mode=args.fusion_mode,
            shortlist_fraction=args.shortlist_fraction,
        )
        output = output_root / task / f"{args.output_selector}.json"
        atomic_json(result, output)
        print(
            json.dumps(
                {
                    "task": task,
                    "selector": result["selector"],
                    "selected_candidate_id": result["selected_candidate_id"],
                    "output": str(output),
                },
                sort_keys=True,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
