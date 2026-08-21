#!/usr/bin/env python3
"""Exactly decompose shortlist coverage and reranking error on open development.

For a shortlist S and final selected candidate i-hat, this tool reports

    R(i-hat)-R* = [R*_{G(S)}-R*]
                    + [R*_S-R*_{G(S)}]
                    + [R(i-hat)-R*_S],

where G(S) contains every checkpoint from a trajectory represented in S.  It
also emits the analogous method-coverage decomposition.  Inputs are restricted
to selector JSON files and the already exported open-development truth report.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from selector.objective_v2 import (  # noqa: E402
    OPEN_DEVELOPMENT_TASKS,
    discover_selection_paths,
    load_open_dev_truth,
    load_selection,
)
from selector.selection_error_decomposition import (  # noqa: E402
    parse_candidate_id,
    reject_forbidden_path,
    trajectory_key,
)


def atomic_json(document: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(document, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def infer_shortlist(selection: dict[str, Any]) -> set[str]:
    fusion = selection.get("fusion_config")
    if not isinstance(fusion, dict):
        raise ValueError("shortlist decomposition requires fusion_config")
    exclusion = fusion.get("shortlist_exclusion_score")
    if exclusion is None:
        raise ValueError("fusion_config lacks shortlist_exclusion_score")
    threshold = float(exclusion)
    shortlist = {
        candidate
        for candidate, score in selection["candidate_scores"].items()
        if float(score) < threshold
    }
    if not shortlist:
        raise ValueError("inferred shortlist is empty")
    declared_sizes = [
        fusion.get("shortlist_size"),
        fusion.get("shortlist_candidate_count"),
    ]
    declared = next((int(value) for value in declared_sizes if value is not None), None)
    if declared is not None and declared != len(shortlist):
        raise ValueError(
            f"declared shortlist size {declared} does not match inferred {len(shortlist)}"
        )
    return shortlist


def _best_candidate(candidates: set[str], risks: dict[str, float]) -> str:
    if not candidates:
        raise ValueError("candidate pool is empty")
    return min(candidates, key=lambda candidate: (float(risks[candidate]), candidate))


def decompose_shortlist(
    selection: dict[str, Any],
    truth: dict[str, Any],
) -> dict[str, Any]:
    task = str(selection["task"])
    if task != truth["task"]:
        raise ValueError("selection task does not match truth task")
    risks = {str(candidate): float(value) for candidate, value in truth["risks"].items()}
    if set(selection["candidate_scores"]) != set(risks):
        raise ValueError(f"candidate coverage mismatch for {task}/{selection['selector']}")
    shortlist = infer_shortlist(selection)
    metadata = {candidate: parse_candidate_id(candidate) for candidate in risks}
    represented_trajectories = {trajectory_key(metadata[candidate]) for candidate in shortlist}
    represented_methods = {str(metadata[candidate]["method"]) for candidate in shortlist}
    trajectory_pool = {
        candidate
        for candidate, candidate_metadata in metadata.items()
        if trajectory_key(candidate_metadata) in represented_trajectories
    }
    method_pool = {
        candidate
        for candidate, candidate_metadata in metadata.items()
        if str(candidate_metadata["method"]) in represented_methods
    }

    all_candidates = set(risks)
    oracle = _best_candidate(all_candidates, risks)
    trajectory_oracle = _best_candidate(trajectory_pool, risks)
    method_oracle = _best_candidate(method_pool, risks)
    shortlist_oracle = _best_candidate(shortlist, risks)
    selected = str(selection["selected_candidate_id"])
    if selected not in shortlist:
        raise ValueError("selected candidate is outside the inferred shortlist")

    oracle_risk = risks[oracle]
    selected_risk = risks[selected]
    trajectory_oracle_risk = risks[trajectory_oracle]
    method_oracle_risk = risks[method_oracle]
    shortlist_oracle_risk = risks[shortlist_oracle]
    risk_range = max(risks.values()) - min(risks.values())
    denominator = max(float(risk_range), 1.0e-12)

    trajectory_coverage_gap = trajectory_oracle_risk - oracle_risk
    checkpoint_coverage_gap = shortlist_oracle_risk - trajectory_oracle_risk
    method_coverage_gap = method_oracle_risk - oracle_risk
    within_method_coverage_gap = shortlist_oracle_risk - method_oracle_risk
    reranking_gap = selected_risk - shortlist_oracle_risk
    total_gap = selected_risk - oracle_risk
    if not np.isclose(
        total_gap,
        trajectory_coverage_gap + checkpoint_coverage_gap + reranking_gap,
        atol=1.0e-12,
    ):
        raise AssertionError("trajectory shortlist decomposition is not exact")
    if not np.isclose(
        total_gap,
        method_coverage_gap + within_method_coverage_gap + reranking_gap,
        atol=1.0e-12,
    ):
        raise AssertionError("method shortlist decomposition is not exact")

    return {
        "task": task,
        "selector": str(selection["selector"]),
        "shortlist_size": len(shortlist),
        "represented_trajectory_count": len(represented_trajectories),
        "represented_method_count": len(represented_methods),
        "selected_candidate_id": selected,
        "shortlist_oracle_candidate_id": shortlist_oracle,
        "trajectory_pool_oracle_candidate_id": trajectory_oracle,
        "method_pool_oracle_candidate_id": method_oracle,
        "global_oracle_candidate_id": oracle,
        "selected_target_error": selected_risk,
        "shortlist_oracle_target_error": shortlist_oracle_risk,
        "trajectory_pool_oracle_target_error": trajectory_oracle_risk,
        "method_pool_oracle_target_error": method_oracle_risk,
        "global_oracle_target_error": oracle_risk,
        "risk_range": risk_range,
        "total_gap": total_gap,
        "trajectory_coverage_gap": trajectory_coverage_gap,
        "checkpoint_coverage_gap": checkpoint_coverage_gap,
        "method_coverage_gap": method_coverage_gap,
        "within_method_coverage_gap": within_method_coverage_gap,
        "reranking_gap": reranking_gap,
        "normalized_total_gap": total_gap / denominator,
        "normalized_trajectory_coverage_gap": trajectory_coverage_gap / denominator,
        "normalized_checkpoint_coverage_gap": checkpoint_coverage_gap / denominator,
        "normalized_method_coverage_gap": method_coverage_gap / denominator,
        "normalized_within_method_coverage_gap": within_method_coverage_gap / denominator,
        "normalized_reranking_gap": reranking_gap / denominator,
    }


def aggregate(reports: list[dict[str, Any]]) -> dict[str, Any]:
    keys = (
        "normalized_total_gap",
        "normalized_trajectory_coverage_gap",
        "normalized_checkpoint_coverage_gap",
        "normalized_method_coverage_gap",
        "normalized_within_method_coverage_gap",
        "normalized_reranking_gap",
    )
    means = {
        f"mean_{key}": float(np.mean([float(report[key]) for report in reports]))
        for key in keys
    }
    total = sum(float(report["normalized_total_gap"]) for report in reports)
    component_shares = {}
    for key in keys[1:]:
        component_shares[key.removeprefix("normalized_") + "_share"] = (
            sum(float(report[key]) for report in reports) / total if total > 1.0e-12 else 0.0
        )
    return {
        "selector": reports[0]["selector"],
        "task_count": len(reports),
        **means,
        **component_shares,
        "mean_shortlist_size": float(np.mean([report["shortlist_size"] for report in reports])),
        "mean_represented_trajectory_count": float(
            np.mean([report["represented_trajectory_count"] for report in reports])
        ),
        "tasks": reports,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    tasks = args.tasks or list(OPEN_DEVELOPMENT_TASKS)
    reject_forbidden_path(args.dev_truth_report)
    for root in args.selection_root:
        reject_forbidden_path(root)
    truth = load_open_dev_truth(args.dev_truth_report.resolve())
    paths = discover_selection_paths(
        args.selection_root, selectors=args.selector, tasks=tasks
    )
    by_selector: dict[str, list[dict[str, Any]]] = {}
    for path in paths:
        selection = load_selection(path)
        task = str(selection["task"])
        if task in tasks:
            report = decompose_shortlist(selection, truth[task])
            by_selector.setdefault(report["selector"], []).append(report)
    expected = set(tasks)
    selectors = {}
    for selector, reports in sorted(by_selector.items()):
        if {report["task"] for report in reports} != expected or len(reports) != len(tasks):
            raise ValueError(f"selector {selector} does not cover tasks exactly once")
        selectors[selector] = aggregate(reports)
    result = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "open_development_tasks": tasks,
        "selectors": selectors,
        "label_access_count": 0,
        "protocol_violation_count": 0,
    }
    if args.output is not None:
        atomic_json(result, args.output.resolve())
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-truth-report", type=Path, required=True)
    parser.add_argument("--selection-root", type=Path, action="append", required=True)
    parser.add_argument("--selector", action="append", required=True)
    parser.add_argument("--tasks", nargs="*")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    result = run(parse_args())
    print(
        json.dumps(
            {
                "selector_count": len(result["selectors"]),
                "label_access_count": 0,
                "protocol_violation_count": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
