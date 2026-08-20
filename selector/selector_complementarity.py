#!/usr/bin/env python3
"""Diagnose complementarity between label-free selectors on open-development tasks.

The diagnostic reads only selector JSON files and, optionally, an already
exported ``objective_v2.py`` open-development report.  It never reads candidate
artifacts, target-label files, ``sealed_eval``, or final-label paths.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from selector.reliable_selection import percentile_ranks, top_fraction_candidates

FORBIDDEN_PATH_PARTS = {".sealed", "sealed_eval", "final_12_labels"}
TOP_FRACTIONS = (0.05, 0.10, 0.20)


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


def pct_key(fraction: float) -> str:
    return f"{int(round(100 * fraction))}pct"


def load_json(path: Path) -> dict[str, Any]:
    reject_forbidden_path(path)
    document = json.loads(path.read_text(encoding="utf-8"))
    if int(document.get("label_access_count", 0)) != 0:
        raise RuntimeError(f"{path} reports target-label access")
    if int(document.get("protocol_violation_count", 0)) != 0:
        raise RuntimeError(f"{path} reports protocol violations")
    return document


def load_objective_reports(paths: list[Path] | None) -> list[dict[str, Any]]:
    reports = []
    for path in paths or []:
        report = load_json(path)
        selectors = report.get("selectors")
        if not isinstance(selectors, dict):
            raise ValueError("objective report must contain a selector aggregate map")
        reports.append(report)
    return reports


def find_selection(selection_roots: list[Path], task: str, selector: str) -> Path:
    matches = []
    for root in selection_roots:
        reject_forbidden_path(root)
        candidate = root / task / f"{selector}.json"
        if candidate.is_file():
            matches.append(candidate)
    if not matches:
        raise FileNotFoundError(f"missing selector '{selector}' for task '{task}'")
    if len(matches) > 1:
        raise ValueError(
            f"selector '{selector}' for task '{task}' appears in multiple roots: {matches}"
        )
    return matches[0]


def load_selections(
    selection_roots: list[Path],
    tasks: list[str],
    selectors: list[str],
) -> dict[str, dict[str, dict[str, Any]]]:
    by_task: dict[str, dict[str, dict[str, Any]]] = {}
    for task in tasks:
        by_selector: dict[str, dict[str, Any]] = {}
        for selector in selectors:
            document = load_json(find_selection(selection_roots, task, selector))
            if document.get("task") != task:
                raise ValueError(f"{task}/{selector} has mismatched task metadata")
            if document.get("selector") != selector:
                raise ValueError(f"{task}/{selector} has mismatched selector metadata")
            by_selector[selector] = document
        by_task[task] = by_selector
    return by_task


def aligned_candidates(selections: dict[str, dict[str, Any]]) -> list[str]:
    candidate_sets = {
        selector: set(document["candidate_scores"])
        for selector, document in selections.items()
    }
    if len({frozenset(value) for value in candidate_sets.values()}) != 1:
        raise ValueError(f"selector candidate sets do not align: {candidate_sets.keys()}")
    hashes = {
        str(document.get("candidate_bank_sha256"))
        for document in selections.values()
    }
    if len(hashes) != 1:
        raise ValueError(f"selector candidate bank hashes do not align: {hashes}")
    return sorted(next(iter(candidate_sets.values())))


def rank_vector(selection: dict[str, Any], candidates: list[str]) -> np.ndarray:
    ranks = percentile_ranks(selection["candidate_scores"], selection["score_direction"])
    return np.asarray([float(ranks[candidate]) for candidate in candidates], dtype=np.float64)


def spearman_from_rank_vectors(first: np.ndarray, second: np.ndarray) -> float:
    if first.size != second.size or first.size == 0:
        raise ValueError("rank vectors must be non-empty and aligned")
    first_centered = first - float(np.mean(first))
    second_centered = second - float(np.mean(second))
    denominator = float(np.linalg.norm(first_centered) * np.linalg.norm(second_centered))
    if denominator == 0.0:
        return 0.0
    return float(np.dot(first_centered, second_centered) / denominator)


def score_tie_stats(scores: dict[str, float], direction: str) -> dict[str, Any]:
    values = [float(value) for value in scores.values()]
    counts = Counter(values)
    if direction == "minimize":
        best = min(values)
        distinct = sorted(set(values))
    else:
        best = max(values)
        distinct = sorted(set(values), reverse=True)
    if len(distinct) <= 1:
        margin = math.inf
    else:
        margin = abs(float(distinct[1]) - float(distinct[0]))
    return {
        "unique_score_ratio": float(len(counts) / max(1, len(values))),
        "max_tie_group_size": int(max(counts.values())),
        "top_score_tie_group_size": int(counts[best]),
        "margin_to_next_distinct_score": float(margin),
    }


def parse_candidate_id(candidate_id: str) -> dict[str, str | int | None]:
    parts = candidate_id.split("__")
    method = parts[1] if len(parts) > 1 else None
    config = parts[2] if len(parts) > 2 else None
    seed = None
    epoch = None
    for part in parts:
        if part.startswith("seed-"):
            seed = part.removeprefix("seed-")
        if part.startswith("epoch-"):
            try:
                epoch = int(part.removeprefix("epoch-"))
            except ValueError:
                epoch = None
    return {"method": method, "config": config, "seed": seed, "epoch": epoch}


def normalized_entropy(counter: Counter[str | None]) -> float:
    total = sum(counter.values())
    if total <= 0 or len(counter) <= 1:
        return 0.0
    probabilities = [count / total for count in counter.values()]
    entropy = -sum(p * math.log(p) for p in probabilities if p > 0)
    return float(entropy / math.log(len(counter)))


def top_profile(selection: dict[str, Any], candidates: list[str], fraction: float) -> dict[str, Any]:
    ranks = percentile_ranks(selection["candidate_scores"], selection["score_direction"])
    top_ids = top_fraction_candidates(ranks, fraction=fraction)
    methods = Counter(parse_candidate_id(candidate)["method"] for candidate in top_ids)
    epochs = [
        parse_candidate_id(candidate)["epoch"]
        for candidate in top_ids
        if parse_candidate_id(candidate)["epoch"] is not None
    ]
    return {
        "fraction": fraction,
        "size": len(top_ids),
        "method_counts": dict(sorted(methods.items(), key=lambda item: (str(item[0]), item[1]))),
        "method_entropy": normalized_entropy(methods),
        "epoch_min": int(min(epochs)) if epochs else None,
        "epoch_median": float(np.median(epochs)) if epochs else None,
        "epoch_max": int(max(epochs)) if epochs else None,
    }


def selected_rank_under(
    selected_candidate: str,
    reference_selection: dict[str, Any],
) -> float:
    ranks = percentile_ranks(
        reference_selection["candidate_scores"],
        reference_selection["score_direction"],
    )
    return float(ranks[selected_candidate])


def objective_task_map(objective_reports: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for objective_report in objective_reports:
        for selector, aggregate in objective_report["selectors"].items():
            for task_report in aggregate.get("tasks", []):
                task = str(task_report["task"])
                if selector in result.setdefault(task, {}):
                    continue
                result[task][selector] = task_report
    return result


def selector_summary(
    task: str,
    selector: str,
    selection: dict[str, Any],
    candidates: list[str],
    objective_by_task: dict[str, dict[str, dict[str, Any]]],
    transfer_selector: str,
) -> dict[str, Any]:
    selected = str(selection["selected_candidate_id"])
    ranks = percentile_ranks(selection["candidate_scores"], selection["score_direction"])
    summary: dict[str, Any] = {
        "selector": selector,
        "selected_candidate_id": selected,
        "selected_candidate_metadata": parse_candidate_id(selected),
        "selected_percentile_rank": float(ranks[selected]),
        "score_ties": score_tie_stats(
            selection["candidate_scores"],
            selection["score_direction"],
        ),
        "top_profiles": {
            pct_key(fraction): top_profile(selection, candidates, fraction)
            for fraction in TOP_FRACTIONS
        },
    }
    objective_task = objective_by_task.get(task, {})
    if selector in objective_task:
        task_report = objective_task[selector]
        transfer_report = objective_task.get(transfer_selector)
        summary["open_dev_metrics"] = {
            key: task_report.get(key)
            for key in (
                "normalized_regret",
                "selected_micro_f1",
                "top_10pct_hit",
                "top_20pct_hit",
                "oracle_recall_at_20pct",
            )
            if key in task_report
        }
        if transfer_report is not None:
            summary["open_dev_metrics"]["noninferior_to_transfer"] = bool(
                float(task_report["normalized_regret"])
                <= float(transfer_report["normalized_regret"]) + 1.0e-12
            )
    return summary


def pair_summary(
    first_name: str,
    first: dict[str, Any],
    second_name: str,
    second: dict[str, Any],
    candidates: list[str],
) -> dict[str, Any]:
    first_ranks = rank_vector(first, candidates)
    second_ranks = rank_vector(second, candidates)
    selected_first = str(first["selected_candidate_id"])
    selected_second = str(second["selected_candidate_id"])
    summary: dict[str, Any] = {
        "selectors": [first_name, second_name],
        "same_selected_candidate": selected_first == selected_second,
        "rank_spearman": spearman_from_rank_vectors(first_ranks, second_ranks),
        "first_selected_rank_under_second": selected_rank_under(selected_first, second),
        "second_selected_rank_under_first": selected_rank_under(selected_second, first),
    }
    first_rank_map = percentile_ranks(first["candidate_scores"], first["score_direction"])
    second_rank_map = percentile_ranks(second["candidate_scores"], second["score_direction"])
    for fraction in TOP_FRACTIONS:
        first_top = top_fraction_candidates(first_rank_map, fraction=fraction)
        second_top = top_fraction_candidates(second_rank_map, fraction=fraction)
        intersection = first_top & second_top
        union = first_top | second_top
        key = pct_key(fraction)
        summary[f"top_{key}_overlap_count"] = len(intersection)
        summary[f"top_{key}_jaccard"] = float(len(intersection) / len(union)) if union else 0.0
        summary[f"top_{key}_overlap_rate_vs_first"] = (
            float(len(intersection) / len(first_top)) if first_top else 0.0
        )
        summary[f"top_{key}_overlap_rate_vs_second"] = (
            float(len(intersection) / len(second_top)) if second_top else 0.0
        )
    return summary


def aggregate_pairs(task_reports: list[dict[str, Any]]) -> dict[str, Any]:
    by_pair: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for task_report in task_reports:
        for pair in task_report["pairwise"]:
            by_pair.setdefault(tuple(pair["selectors"]), []).append(pair)
    aggregate: dict[str, Any] = {}
    for pair, reports in sorted(by_pair.items()):
        key = "__vs__".join(pair)
        aggregate[key] = {
            "task_count": len(reports),
            "same_selected_rate": float(np.mean([report["same_selected_candidate"] for report in reports])),
            "mean_rank_spearman": float(np.mean([report["rank_spearman"] for report in reports])),
        }
        for fraction in TOP_FRACTIONS:
            metric = f"top_{pct_key(fraction)}_jaccard"
            aggregate[key][f"mean_{metric}"] = float(np.mean([report[metric] for report in reports]))
    return aggregate


def run(args: argparse.Namespace) -> dict[str, Any]:
    objective_reports = load_objective_reports(args.objective_report)
    if args.task:
        tasks = args.task
    elif objective_reports:
        tasks = list(objective_reports[0].get("open_development_tasks") or [])
    else:
        raise ValueError("pass --task when no objective report is supplied")
    if not tasks:
        raise ValueError("no tasks to diagnose")
    if len(args.selector) < 2:
        raise ValueError("at least two selectors are required")

    selection_roots = [path.resolve() for path in args.selection_root]
    objective_by_task = objective_task_map(objective_reports)
    loaded = load_selections(selection_roots, tasks, args.selector)

    task_diagnostics: list[dict[str, Any]] = []
    for task in tasks:
        selections = loaded[task]
        candidates = aligned_candidates(selections)
        summaries = {
            selector: selector_summary(
                task,
                selector,
                selections[selector],
                candidates,
                objective_by_task,
                args.transfer_selector,
            )
            for selector in args.selector
        }
        pairwise = []
        for index, first_name in enumerate(args.selector):
            for second_name in args.selector[index + 1 :]:
                pairwise.append(
                    pair_summary(
                        first_name,
                        selections[first_name],
                        second_name,
                        selections[second_name],
                        candidates,
                    )
                )
        task_diagnostics.append(
            {
                "task": task,
                "candidate_count": len(candidates),
                "candidate_bank_sha256": next(
                    iter({document["candidate_bank_sha256"] for document in selections.values()})
                ),
                "selectors": summaries,
                "pairwise": pairwise,
            }
        )

    result = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "label-free selector complementarity diagnostic",
        "selection_roots": [str(path) for path in selection_roots],
        "objective_reports": [
            str(path.resolve()) for path in args.objective_report or []
        ],
        "selectors": args.selector,
        "transfer_selector": args.transfer_selector,
        "tasks": tasks,
        "task_diagnostics": task_diagnostics,
        "pairwise_aggregate": aggregate_pairs(task_diagnostics),
        "label_access_count": 0,
        "protocol_violation_count": 0,
    }
    if args.output is not None:
        reject_forbidden_path(args.output)
        atomic_json(result, args.output)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-root", type=Path, action="append", required=True)
    parser.add_argument("--objective-report", type=Path, action="append")
    parser.add_argument("--selector", action="append", required=True)
    parser.add_argument("--task", action="append")
    parser.add_argument("--transfer-selector", default="transfer_score")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    result = run(parse_args())
    print(
        json.dumps(
            {
                "task_count": len(result["tasks"]),
                "selectors": result["selectors"],
                "pairwise_aggregate": result["pairwise_aggregate"],
                "label_access_count": result["label_access_count"],
                "protocol_violation_count": result["protocol_violation_count"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
