#!/usr/bin/env python3
"""Evaluate selector outputs while keeping candidate-level target truth sealed."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from metrics import macro_f1, normalized_regret, rank_correlations, top_fraction_hit  # noqa: E402
from sealed_eval.access import SEALED_ROOT, load_target_labels  # noqa: E402
from scripts.trajectory_export.schema import (  # noqa: E402
    atomic_json,
    candidate_bank_hash,
    discover_candidate_records,
)

SELECTION_SCHEMA_VERSION = 1


def _mean(values: list[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and np.isfinite(value)]
    return float(np.mean(finite)) if finite else None


def _median(values: list[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and np.isfinite(value)]
    return float(statistics.median(finite)) if finite else None


def load_selection(path: Path) -> dict[str, Any]:
    selection = json.loads(path.read_text(encoding="utf-8"))
    if selection.get("schema_version") != SELECTION_SCHEMA_VERSION:
        raise ValueError(f"unsupported selection schema: {path}")
    if selection.get("label_access_count") != 0:
        raise ValueError(f"selector reports target-label access: {path}")
    if selection.get("protocol_violation_count") != 0:
        raise ValueError(f"selector reports protocol violations: {path}")
    if selection.get("score_direction") not in {"minimize", "maximize"}:
        raise ValueError(f"invalid score direction: {path}")
    if selection.get("score_semantics") not in {"ranking", "estimated_error"}:
        raise ValueError(f"invalid score semantics: {path}")
    return selection


def evaluate_one(
    selection_path: Path,
    candidate_root: Path,
    purpose: str,
    label_cache: dict[tuple[str, str], np.ndarray] | None = None,
) -> dict[str, Any]:
    selection = load_selection(selection_path)
    task = selection["task"]
    records = discover_candidate_records(candidate_root, task)
    bank_hash = candidate_bank_hash(records)
    if selection.get("candidate_bank_sha256") != bank_hash:
        raise ValueError(f"candidate-bank hash mismatch: {selection_path}")

    by_id = {record["metadata"]["candidate_id"]: record for record in records}
    scores = selection.get("candidate_scores", {})
    if set(scores) != set(by_id):
        missing = sorted(set(by_id) - set(scores))
        extra = sorted(set(scores) - set(by_id))
        raise ValueError(f"selector scores must cover the full bank; missing={missing[:3]} extra={extra[:3]}")
    selected_id = selection["selected_candidate_id"]
    if selected_id not in by_id:
        raise ValueError("selected candidate is absent from candidate bank")

    identifiers = sorted(by_id)
    raw_scores = np.asarray([float(scores[candidate]) for candidate in identifiers], dtype=np.float64)
    if not np.isfinite(raw_scores).all():
        raise ValueError("selector produced non-finite candidate scores")
    predicted_risks = raw_scores if selection["score_direction"] == "minimize" else -raw_scores
    selected_index = identifiers.index(selected_id)
    if predicted_risks[selected_index] > predicted_risks.min() + 1e-12:
        raise ValueError("selected candidate is not an optimum of the declared scores")

    first_target_path = by_id[identifiers[0]]["path"] / "target_public.npz"
    with np.load(first_target_path, allow_pickle=False) as first_target:
        expected_num_nodes = int(first_target["hard_predictions"].shape[0])
    cache_key = (task, bank_hash)
    if label_cache is not None and cache_key in label_cache:
        labels = label_cache[cache_key]
    else:
        labels = load_target_labels(
            task,
            purpose=purpose,
            candidate_bank_sha256=bank_hash,
            expected_num_nodes=expected_num_nodes,
        ).numpy()
        if label_cache is not None:
            label_cache[cache_key] = labels

    true_risks = []
    macro_scores = []
    candidate_truth: dict[str, Any] = {}
    for identifier in identifiers:
        with np.load(by_id[identifier]["path"] / "target_public.npz", allow_pickle=False) as artifact:
            predictions = artifact["hard_predictions"].astype(np.int64, copy=False)
        if predictions.shape != labels.shape:
            raise ValueError(f"prediction shape mismatch for {identifier}")
        accuracy = float(np.mean(predictions == labels))
        risk = 1.0 - accuracy
        candidate_macro = macro_f1(labels, predictions)
        true_risks.append(risk)
        macro_scores.append(candidate_macro)
        candidate_truth[identifier] = {
            "target_error": risk,
            "target_micro_f1": accuracy,
            "target_macro_f1": candidate_macro,
        }

    true_risks_array = np.asarray(true_risks, dtype=np.float64)
    selected_risk = float(true_risks_array[selected_index])
    oracle_index = int(np.argmin(true_risks_array))
    oracle_risk = float(true_risks_array[oracle_index])
    correlations = rank_correlations(predicted_risks, true_risks_array)
    risk_mae = None
    if selection["score_semantics"] == "estimated_error":
        risk_mae = float(np.mean(np.abs(predicted_risks - true_risks_array)))

    return {
        "task": task,
        "selector": selection["selector"],
        "selection_path": str(selection_path),
        "candidate_bank_sha256": bank_hash,
        "candidate_count": len(identifiers),
        "selected_candidate_id": selected_id,
        "oracle_candidate_id": identifiers[oracle_index],
        "selected_target_error": selected_risk,
        "selected_micro_f1": 1.0 - selected_risk,
        "selected_macro_f1": float(macro_scores[selected_index]),
        "oracle_target_error": oracle_risk,
        "oracle_micro_f1": 1.0 - oracle_risk,
        "oracle_micro_f1_gap": selected_risk - oracle_risk,
        "oracle_micro_f1_gap_points": 100.0 * (selected_risk - oracle_risk),
        "normalized_regret": normalized_regret(selected_risk, true_risks_array),
        "kendall_tau": correlations["kendall_tau"],
        "spearman_rho": correlations["spearman_rho"],
        "risk_estimation_mae": risk_mae,
        "top_5pct_hit": top_fraction_hit(selected_index, true_risks_array, fraction=0.05),
        "score_semantics": selection["score_semantics"],
        "candidate_truth": candidate_truth,
    }


def aggregate_reports(reports: list[dict[str, Any]], purpose: str) -> dict[str, Any]:
    selectors = sorted({report["selector"] for report in reports})
    if len(selectors) != 1:
        raise ValueError("one evaluator invocation must contain a single selector")
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": purpose,
        "selector": selectors[0],
        "task_count": len(reports),
        "candidate_count_total": sum(report["candidate_count"] for report in reports),
        "mean_normalized_regret": _mean([report["normalized_regret"] for report in reports]),
        "median_kendall_tau": _median([report["kendall_tau"] for report in reports]),
        "mean_spearman_rho": _mean([report["spearman_rho"] for report in reports]),
        "risk_estimation_mae": _mean([report["risk_estimation_mae"] for report in reports]),
        "mean_oracle_f1_gap": _mean([report["oracle_micro_f1_gap"] for report in reports]),
        "mean_oracle_f1_gap_points": _mean([report["oracle_micro_f1_gap_points"] for report in reports]),
        "top_5pct_hit_rate": _mean([float(report["top_5pct_hit"]) for report in reports]),
        "mean_selected_micro_f1": _mean([report["selected_micro_f1"] for report in reports]),
        "mean_selected_macro_f1": _mean([report["selected_macro_f1"] for report in reports]),
        "label_access_count": 0,
        "evaluator_target_label_read_count": len(reports),
        "protocol_violation_count": 0,
        "candidate_bank_sha256": [report["candidate_bank_sha256"] for report in reports],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("selection", nargs="+", type=Path)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--purpose", choices=("pilot", "development", "final"), required=True)
    parser.add_argument("--public-report", type=Path, required=True)
    parser.add_argument("--sealed-report", type=Path)
    parser.add_argument(
        "--trusted-evaluator",
        action="store_true",
        help="required acknowledgement that this process reads sealed target labels",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.trusted_evaluator:
        raise SystemExit("refusing target-label access without --trusted-evaluator")
    label_cache: dict[tuple[str, str], np.ndarray] = {}
    reports = [
        evaluate_one(
            path.resolve(),
            args.candidate_root.resolve(),
            args.purpose,
            label_cache=label_cache,
        )
        for path in args.selection
    ]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for report in reports:
        grouped.setdefault(report["selector"], []).append(report)
    aggregates = {
        selector: aggregate_reports(selector_reports, args.purpose)
        for selector, selector_reports in sorted(grouped.items())
    }
    if len(aggregates) == 1:
        public_report: dict[str, Any] = next(iter(aggregates.values()))
    else:
        public_report = {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "purpose": args.purpose,
            "selector_count": len(aggregates),
            "unique_task_bank_count": len(label_cache),
            "evaluator_target_label_read_count": len(label_cache),
            "label_access_count": 0,
            "protocol_violation_count": 0,
            "selectors": aggregates,
        }
    sealed_report = args.sealed_report
    if sealed_report is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        label = next(iter(aggregates)) if len(aggregates) == 1 else "selector_comparison"
        sealed_report = SEALED_ROOT / "evaluator_reports" / f"{args.purpose}_{label}_{stamp}.json"
    atomic_json(
        {"aggregates": aggregates, "tasks": reports},
        sealed_report.resolve(),
        mode=0o600,
    )
    atomic_json(public_report, args.public_report.resolve())
    print(json.dumps(public_report, indent=2))
    print(f"sealed candidate-level report: {sealed_report.resolve()}")


if __name__ == "__main__":
    main()
