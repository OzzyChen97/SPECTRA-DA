#!/usr/bin/env python3
"""Leave-one-task-out selector-choice diagnostic for open-development reports.

This script consumes an ``objective_v2.py`` report that already contains
open-development selector metrics.  It does not read candidate artifacts,
target labels, ``sealed_eval``, or final-label paths.  The goal is to prevent
choosing a reliable-selector hyperparameter from the same four tasks on which
it is reported.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

FORBIDDEN_PATH_PARTS = {".sealed", "sealed_eval", "final_12_labels"}
SHORTLIST_FUSION_MODES = {
    "transfer_shortlist_spectra_rerank",
    "spectra_shortlist_transfer_rerank",
}


def reject_forbidden_path(path: Path) -> None:
    forbidden = sorted(set(path.resolve().parts) & FORBIDDEN_PATH_PARTS)
    if forbidden:
        raise RuntimeError(f"forbidden path components: {forbidden}")


def atomic_json(document: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
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


def finite_mean(values: list[float]) -> float:
    if not values:
        raise ValueError("mean requires at least one value")
    return float(np.mean([float(value) for value in values]))


def cvar_tail(values: list[float], fraction: float = 0.20) -> float:
    if not values:
        raise ValueError("CVaR requires at least one value")
    ordered = sorted((float(value) for value in values), reverse=True)
    count = max(1, math.ceil(fraction * len(ordered)))
    return float(np.mean(ordered[:count]))


def task_reports(selector: dict[str, Any]) -> dict[str, dict[str, Any]]:
    reports = selector.get("tasks")
    if not isinstance(reports, list) or not reports:
        raise ValueError(f"selector {selector.get('selector')} has no task reports")
    return {str(report["task"]): report for report in reports}


def uses_shortlist(selector: dict[str, Any]) -> bool:
    modes = set(selector.get("fusion_modes") or [])
    if modes & SHORTLIST_FUSION_MODES:
        return True
    name = str(selector.get("selector", ""))
    return "_tsr_" in name or "_str_" in name


def subset_metrics(
    selector: dict[str, Any],
    transfer: dict[str, Any],
    tasks: list[str],
    *,
    mean_regret_max: float,
    worst_regret_max: float,
    task_noninferiority_min: float,
    shortlist_recall20_min: float,
) -> dict[str, Any]:
    selector_by_task = task_reports(selector)
    transfer_by_task = task_reports(transfer)
    missing = sorted(set(tasks) - set(selector_by_task))
    if missing:
        raise ValueError(f"selector {selector.get('selector')} missing tasks: {missing}")
    regrets = [float(selector_by_task[task]["normalized_regret"]) for task in tasks]
    micro = [float(selector_by_task[task]["selected_micro_f1"]) for task in tasks]
    transfer_regrets = [float(transfer_by_task[task]["normalized_regret"]) for task in tasks]
    transfer_micro = [float(transfer_by_task[task]["selected_micro_f1"]) for task in tasks]
    noninferior = [
        regrets[index] <= transfer_regrets[index] + 1.0e-12
        for index in range(len(tasks))
    ]
    oracle_recall20 = finite_mean(
        [float(selector_by_task[task]["oracle_recall_at_20pct"]) for task in tasks]
    )
    mean_regret = finite_mean(regrets)
    worst_regret = float(max(regrets))
    mean_micro = finite_mean(micro)
    transfer_mean_micro = finite_mean(transfer_micro)
    task_noninferiority_rate = float(np.mean(noninferior))
    shortlist_required = uses_shortlist(selector)
    pass_absolute = (
        mean_regret < mean_regret_max
        and worst_regret < worst_regret_max
        and mean_micro > transfer_mean_micro
        and task_noninferiority_rate >= task_noninferiority_min
    )
    pass_shortlist = (not shortlist_required) or oracle_recall20 >= shortlist_recall20_min
    return {
        "task_count": len(tasks),
        "mean_normalized_regret": mean_regret,
        "cvar_20pct_normalized_regret": cvar_tail(regrets, fraction=0.20),
        "worst_normalized_regret": worst_regret,
        "mean_selected_micro_f1": mean_micro,
        "transfer_mean_selected_micro_f1": transfer_mean_micro,
        "task_noninferiority_rate_vs_transfer": task_noninferiority_rate,
        "mean_oracle_recall_at_20pct": oracle_recall20,
        "shortlist_guardrail_required": shortlist_required,
        "eligible": bool(pass_absolute and pass_shortlist),
        "guards": {
            "mean_regret_below_max": mean_regret < mean_regret_max,
            "worst_regret_below_max": worst_regret < worst_regret_max,
            "micro_above_transfer": mean_micro > transfer_mean_micro,
            "task_noninferiority_rate_at_least_min": (
                task_noninferiority_rate >= task_noninferiority_min
            ),
            "shortlist_recall20_at_least_min": pass_shortlist,
        },
    }


def choose_selector(
    selectors: dict[str, dict[str, Any]],
    transfer: dict[str, Any],
    candidates: list[str],
    train_tasks: list[str],
    args: argparse.Namespace,
) -> tuple[str, dict[str, Any]]:
    scored: list[tuple[bool, float, float, float, str, dict[str, Any]]] = []
    for name in candidates:
        metrics = subset_metrics(
            selectors[name],
            transfer,
            train_tasks,
            mean_regret_max=args.mean_regret_max,
            worst_regret_max=args.worst_regret_max,
            task_noninferiority_min=args.task_noninferiority_min,
            shortlist_recall20_min=args.shortlist_recall20_min,
        )
        scored.append(
            (
                bool(metrics["eligible"]),
                -float(metrics["mean_normalized_regret"]),
                -float(metrics["task_noninferiority_rate_vs_transfer"]),
                -float(metrics["mean_selected_micro_f1"]),
                name,
                metrics,
            )
        )
    # Prefer eligible selectors; within that, minimize mean regret, maximize
    # task non-inferiority and Micro-F1, then use selector name deterministically.
    scored.sort(reverse=True)
    eligible, _, _, _, name, metrics = scored[0]
    metrics = dict(metrics)
    metrics["selection_reason"] = "eligible_best" if eligible else "fallback_best_mean_regret"
    return name, metrics


def run(args: argparse.Namespace) -> dict[str, Any]:
    reject_forbidden_path(args.objective_report)
    report = json.loads(args.objective_report.read_text(encoding="utf-8"))
    if int(report.get("label_access_count", 0)) != 0:
        raise RuntimeError("objective report must not report target-label access")
    if int(report.get("protocol_violation_count", 0)) != 0:
        raise RuntimeError("objective report reports protocol violations")
    selectors = report.get("selectors")
    if not isinstance(selectors, dict) or not selectors:
        raise ValueError("objective report must contain selectors")
    if args.transfer_selector not in selectors:
        raise ValueError(f"missing transfer selector: {args.transfer_selector}")
    transfer = selectors[args.transfer_selector]
    tasks = list(report.get("open_development_tasks") or sorted(task_reports(transfer)))
    if len(tasks) < 2:
        raise ValueError("leave-one-task-out requires at least two tasks")
    candidates = args.selector or [
        name for name in sorted(selectors) if name != args.transfer_selector
    ]
    missing = sorted(set(candidates) - set(selectors))
    if missing:
        raise ValueError(f"missing candidate selectors: {missing}")

    folds: list[dict[str, Any]] = []
    validation_regrets: list[float] = []
    validation_micro: list[float] = []
    validation_noninferior: list[bool] = []
    validation_recall20: list[float] = []
    for heldout in tasks:
        train_tasks = [task for task in tasks if task != heldout]
        selected, train_metrics = choose_selector(
            selectors,
            transfer,
            candidates,
            train_tasks,
            args,
        )
        selected_reports = task_reports(selectors[selected])
        transfer_reports = task_reports(transfer)
        validation_report = selected_reports[heldout]
        transfer_report = transfer_reports[heldout]
        validation_nregret = float(validation_report["normalized_regret"])
        validation_regrets.append(validation_nregret)
        validation_micro.append(float(validation_report["selected_micro_f1"]))
        noninferior = validation_nregret <= float(transfer_report["normalized_regret"]) + 1.0e-12
        validation_noninferior.append(noninferior)
        validation_recall20.append(float(validation_report["oracle_recall_at_20pct"]))
        folds.append(
            {
                "heldout_task": heldout,
                "train_tasks": train_tasks,
                "selected_selector": selected,
                "train_metrics": train_metrics,
                "validation": {
                    "normalized_regret": validation_nregret,
                    "transfer_normalized_regret": float(transfer_report["normalized_regret"]),
                    "noninferior_to_transfer": bool(noninferior),
                    "selected_micro_f1": float(validation_report["selected_micro_f1"]),
                    "transfer_selected_micro_f1": float(transfer_report["selected_micro_f1"]),
                    "oracle_recall_at_20pct": float(validation_report["oracle_recall_at_20pct"]),
                    "selected_candidate_id": validation_report["selected_candidate_id"],
                    "oracle_candidate_id": validation_report["oracle_candidate_id"],
                },
            }
        )

    aggregate = {
        "mean_validation_normalized_regret": finite_mean(validation_regrets),
        "cvar_20pct_validation_normalized_regret": cvar_tail(validation_regrets),
        "worst_validation_normalized_regret": float(max(validation_regrets)),
        "mean_validation_selected_micro_f1": finite_mean(validation_micro),
        "validation_task_noninferiority_rate_vs_transfer": float(
            np.mean(validation_noninferior)
        ),
        "mean_validation_oracle_recall_at_20pct": finite_mean(validation_recall20),
    }
    aggregate["promotion_ready"] = bool(
        aggregate["mean_validation_normalized_regret"] < args.mean_regret_max
        and aggregate["worst_validation_normalized_regret"] < args.worst_regret_max
        and aggregate["validation_task_noninferiority_rate_vs_transfer"]
        >= args.task_noninferiority_min
        and aggregate["mean_validation_oracle_recall_at_20pct"]
        >= args.shortlist_recall20_min
    )
    result = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "objective_report": str(args.objective_report.resolve()),
        "transfer_selector": args.transfer_selector,
        "candidate_selectors": candidates,
        "open_development_tasks": tasks,
        "folds": folds,
        "aggregate": aggregate,
        "thresholds": {
            "mean_regret_max": args.mean_regret_max,
            "worst_regret_max": args.worst_regret_max,
            "task_noninferiority_min": args.task_noninferiority_min,
            "shortlist_recall20_min": args.shortlist_recall20_min,
        },
        "label_access_count": 0,
        "protocol_violation_count": 0,
    }
    if args.output is not None:
        atomic_json(result, args.output)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objective-report", type=Path, required=True)
    parser.add_argument("--transfer-selector", default="transfer_score")
    parser.add_argument("--selector", action="append", dest="selector")
    parser.add_argument("--mean-regret-max", type=float, default=0.20)
    parser.add_argument("--worst-regret-max", type=float, default=0.30)
    parser.add_argument("--task-noninferiority-min", type=float, default=0.75)
    parser.add_argument("--shortlist-recall20-min", type=float, default=0.75)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    result = run(parse_args())
    print(
        json.dumps(
            {
                "selector_count": len(result["candidate_selectors"]),
                "fold_count": len(result["folds"]),
                "aggregate": result["aggregate"],
                "label_access_count": result["label_access_count"],
                "protocol_violation_count": result["protocol_violation_count"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
