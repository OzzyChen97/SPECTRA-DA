#!/usr/bin/env python3
"""Create aggregate Gate-1 diagnostics from a sealed evaluator report."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.trajectory_export.schema import (  # noqa: E402
    atomic_json,
    discover_candidate_records,
)


def chosen_by_source_validation(records: list[dict[str, Any]], method: str) -> str:
    candidates = [record["metadata"] for record in records if record["metadata"]["method"] == method]
    optimum = max(float(metadata["source_val_micro_f1"]) for metadata in candidates)
    return min(
        metadata["candidate_id"]
        for metadata in candidates
        if float(metadata["source_val_micro_f1"]) == optimum
    )


def ranking_reversals(
    records: list[dict[str, Any]],
    candidate_truth: dict[str, Any],
) -> tuple[set[tuple[str, str]], bool]:
    methods = sorted({record["metadata"]["method"] for record in records})
    oracle_risk = {}
    source_selected_risk = {}
    for method in methods:
        identifiers = [
            record["metadata"]["candidate_id"]
            for record in records
            if record["metadata"]["method"] == method
        ]
        oracle_risk[method] = min(candidate_truth[identifier]["target_error"] for identifier in identifiers)
        source_identifier = chosen_by_source_validation(records, method)
        source_selected_risk[method] = candidate_truth[source_identifier]["target_error"]

    reversed_pairs = set()
    for first, second in itertools.combinations(methods, 2):
        oracle_difference = oracle_risk[first] - oracle_risk[second]
        selected_difference = source_selected_risk[first] - source_selected_risk[second]
        if oracle_difference * selected_difference < 0:
            reversed_pairs.add((first, second))
    oracle_best = min(methods, key=lambda method: (oracle_risk[method], method))
    selected_best = min(methods, key=lambda method: (source_selected_risk[method], method))
    return reversed_pairs, oracle_best != selected_best


def build_diagnostics(sealed_report: dict[str, Any], candidate_root: Path) -> dict[str, Any]:
    aggregates = sealed_report["aggregates"]
    reports = sealed_report["tasks"]
    task_reports: dict[str, dict[str, dict[str, Any]]] = {}
    for report in reports:
        task_reports.setdefault(report["task"], {})[report["selector"]] = report

    selectors = sorted(aggregates)
    best_selector = min(
        selectors,
        key=lambda selector: (
            aggregates[selector]["mean_normalized_regret"],
            selector,
        ),
    )
    gap_over_one = {selector: 0 for selector in selectors}
    unique_reversed_pairs: set[tuple[str, str]] = set()
    reversal_tasks = 0
    top_method_change_tasks = 0
    methods: set[str] = set()

    for task, by_selector in sorted(task_reports.items()):
        for selector, report in by_selector.items():
            if report["oracle_micro_f1_gap_points"] > 1.0:
                gap_over_one[selector] += 1
        truth_report = by_selector.get("source_val") or next(iter(by_selector.values()))
        records = discover_candidate_records(candidate_root, task)
        methods.update(record["metadata"]["method"] for record in records)
        reversed_pairs, top_changed = ranking_reversals(records, truth_report["candidate_truth"])
        if reversed_pairs:
            reversal_tasks += 1
            unique_reversed_pairs.update(reversed_pairs)
        if top_changed:
            top_method_change_tasks += 1

    task_count = len(task_reports)
    last_or_source_over_one = sum(
        1
        for by_selector in task_reports.values()
        if max(
            by_selector["last_source_val"]["oracle_micro_f1_gap_points"],
            by_selector["source_val"]["oracle_micro_f1_gap_points"],
        )
        > 1.0
    )
    best_metrics = aggregates[best_selector]
    return {
        "schema_version": 1,
        "quick_pilot_not_paper_result": True,
        "task_count": task_count,
        "candidate_method_count": len(methods),
        "candidate_methods": sorted(methods),
        "best_label_free_selector": best_selector,
        "best_selector_mean_normalized_regret": best_metrics["mean_normalized_regret"],
        "best_selector_mean_oracle_gap_points": best_metrics["mean_oracle_f1_gap_points"],
        "best_selector_median_kendall_tau": best_metrics["median_kendall_tau"],
        "gap_over_1pt_task_count": gap_over_one,
        "last_or_source_gap_over_1pt_task_count": last_or_source_over_one,
        "last_or_source_gap_over_1pt_fraction": last_or_source_over_one / task_count,
        "ranking_reversal_task_count": reversal_tasks,
        "ranking_reversal_task_fraction": reversal_tasks / task_count,
        "unique_reversed_method_pair_count": len(unique_reversed_pairs),
        "unique_reversed_method_pairs": [list(pair) for pair in sorted(unique_reversed_pairs)],
        "top_method_changed_task_count": top_method_change_tasks,
        "gate1_problem_signal": {
            "last_or_source_gap_fraction_at_least_30pct": last_or_source_over_one / task_count >= 0.3,
            "at_least_two_method_pairs_reverse": len(unique_reversed_pairs) >= 2,
            "best_label_free_selector_has_nontrivial_gap": best_metrics["mean_oracle_f1_gap_points"] > 0.5,
            "candidate_pool_is_multimethod": len(methods) >= 4,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sealed-report", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--public-output", type=Path, required=True)
    parser.add_argument("--trusted-evaluator", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.trusted_evaluator:
        raise SystemExit("refusing sealed-report access without --trusted-evaluator")
    sealed = json.loads(args.sealed_report.resolve().read_text(encoding="utf-8"))
    diagnostics = build_diagnostics(sealed, args.candidate_root.resolve())
    atomic_json(diagnostics, args.public_output.resolve())
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
