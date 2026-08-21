#!/usr/bin/env python3
"""Open-real-target development objective for SPECTRA-Trust.

This is the AutoSOTA v3 objective.  It deliberately does not import
``sealed_eval`` or read target labels.  Instead, a trusted evaluator must first
export an open-development report for the four pre-registered Gate-1 tasks,
including candidate-level target errors.  Once those four tasks are declared
open development, this script can safely evaluate selector JSON files and tune
top-1 calibration without touching the final sealed transfers.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import statistics
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from metrics import normalized_regret, rank_correlations
from protocol.tasks import PILOT_TASKS

OPEN_DEVELOPMENT_TASKS = tuple(task.id for task in PILOT_TASKS)
FORBIDDEN_TRUTH_PATH_PARTS = {".sealed", "sealed_eval", "final_12_labels"}
DEFAULT_RUNTIME_BUDGET_SECONDS = 350.0
DEFAULT_SOURCE_SIM_CVAR_DEGRADATION_MAX = 0.05
SHORTLIST_RECALL_FRACTIONS = (0.05, 0.10, 0.20, 0.30, 0.50)
SHORTLIST_FUSION_MODES = {
    "transfer_shortlist_spectra_rerank",
    "spectra_shortlist_transfer_rerank",
    "shortlist_consensus_rerank",
    "trajectory_shortlist_consensus_rerank",
}
SOURCE_SIM_CVAR_KEYS = (
    "source_sim_leave_one_shift_family_cvar",
    "cvar_20pct_normalized_regret",
    "cvar_20pct_regret",
    "cvar_20pct",
    "cvar",
)


def atomic_json(document: dict[str, Any], path: Path) -> None:
    """Write JSON atomically without importing trajectory utilities or torch."""

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


def finite_mean(values: list[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and np.isfinite(value)]
    return float(np.mean(finite)) if finite else None


def finite_median(values: list[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and np.isfinite(value)]
    return float(statistics.median(finite)) if finite else None


def cvar_tail(values: list[float], fraction: float = 0.20) -> float:
    if not values:
        raise ValueError("CVaR requires at least one value")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("CVaR fraction must lie in (0, 1]")
    ordered = sorted((float(value) for value in values), reverse=True)
    count = max(1, math.ceil(fraction * len(ordered)))
    return float(np.mean(ordered[:count]))


def reject_forbidden_truth_path(path: Path) -> None:
    parts = set(path.resolve().parts)
    forbidden = sorted(parts & FORBIDDEN_TRUTH_PATH_PARTS)
    if forbidden:
        raise RuntimeError(
            "objective_v2 must read only an exported open-development truth report; "
            f"forbidden path components: {forbidden}"
        )


def reject_forbidden_report_path(path: Path) -> None:
    parts = set(path.resolve().parts)
    forbidden = sorted(parts & FORBIDDEN_TRUTH_PATH_PARTS)
    if forbidden:
        raise RuntimeError(
            "objective_v2 must not read sealed/final-label report paths; "
            f"forbidden path components: {forbidden}"
        )


def _candidate_error(entry: Any, *, candidate_id: str) -> float:
    if isinstance(entry, dict):
        for key in ("target_error", "risk", "error"):
            if key in entry:
                value = float(entry[key])
                break
        else:
            raise ValueError(f"candidate truth for {candidate_id} has no target_error")
    else:
        value = float(entry)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"invalid target error for {candidate_id}: {value}")
    return value


def load_open_dev_truth(path: Path) -> dict[str, dict[str, Any]]:
    """Load candidate-level truth for the declared open-development tasks."""

    reject_forbidden_truth_path(path)
    document = json.loads(path.read_text(encoding="utf-8"))
    if int(document.get("label_access_count", 0)) != 0:
        raise RuntimeError("open-dev truth report must not be a selector output")
    task_entries = document.get("tasks")
    if not isinstance(task_entries, list):
        raise ValueError("open-dev truth report must contain a tasks list")
    truth: dict[str, dict[str, Any]] = {}
    for entry in task_entries:
        task = str(entry["task"])
        candidate_truth = entry.get("candidate_truth")
        if not isinstance(candidate_truth, dict) or not candidate_truth:
            raise ValueError(f"task {task} lacks candidate-level truth")
        risks = {
            str(candidate_id): _candidate_error(value, candidate_id=str(candidate_id))
            for candidate_id, value in candidate_truth.items()
        }
        if task in truth:
            raise ValueError(f"duplicate task in open-dev truth report: {task}")
        truth[task] = {
            "task": task,
            "candidate_bank_sha256": entry.get("candidate_bank_sha256"),
            "candidate_count": len(risks),
            "risks": risks,
        }
    return truth


def load_selection(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != 1:
        raise ValueError(f"unsupported selection schema: {path}")
    if int(document.get("label_access_count", 0)) != 0:
        raise ValueError(f"selection reports target-label access: {path}")
    if int(document.get("protocol_violation_count", 0)) != 0:
        raise ValueError(f"selection reports protocol violation: {path}")
    if document.get("score_direction") not in {"minimize", "maximize"}:
        raise ValueError(f"invalid score direction: {path}")
    scores = document.get("candidate_scores")
    if not isinstance(scores, dict) or not scores:
        raise ValueError(f"selection has no candidate_scores: {path}")
    numeric = {str(candidate): float(score) for candidate, score in scores.items()}
    if not all(np.isfinite(value) for value in numeric.values()):
        raise ValueError(f"selection contains non-finite scores: {path}")
    document["candidate_scores"] = numeric
    document["selection_path"] = str(path)
    return document


def _selector_entries(document: dict[str, Any]) -> dict[str, dict[str, Any]]:
    selectors = document.get("selectors")
    if isinstance(selectors, dict):
        return {
            str(selector): entry
            for selector, entry in selectors.items()
            if isinstance(entry, dict)
        }
    if isinstance(selectors, list):
        entries: dict[str, dict[str, Any]] = {}
        for entry in selectors:
            if isinstance(entry, dict) and isinstance(entry.get("selector"), str):
                entries[str(entry["selector"])] = entry
        return entries
    return {}


def _source_sim_cvar(entry: dict[str, Any], *, selector: str) -> float:
    for key in SOURCE_SIM_CVAR_KEYS:
        if key in entry:
            value = float(entry[key])
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"invalid source-sim CVaR for {selector}: {value}")
            return value
    raise ValueError(
        f"source-sim report for {selector} has none of the supported CVaR keys: "
        f"{list(SOURCE_SIM_CVAR_KEYS)}"
    )


def load_source_sim_report(path: Path) -> dict[str, dict[str, Any]]:
    reject_forbidden_report_path(path)
    document = json.loads(path.read_text(encoding="utf-8"))
    if int(document.get("label_access_count", 0)) != 0:
        raise RuntimeError("source-sim guardrail report must not report target-label access")
    if int(document.get("protocol_violation_count", 0)) != 0:
        raise RuntimeError("source-sim guardrail report reports protocol violations")
    entries = _selector_entries(document)
    if not entries:
        raise ValueError("source-sim guardrail report must contain a selectors mapping/list")
    return entries


def score_to_predicted_risk(selection: dict[str, Any], candidate_ids: list[str]) -> np.ndarray:
    raw = np.asarray(
        [float(selection["candidate_scores"][candidate]) for candidate in candidate_ids],
        dtype=np.float64,
    )
    return raw if selection["score_direction"] == "minimize" else -raw


def score_tie_diagnostics(selection: dict[str, Any], candidate_ids: list[str]) -> dict[str, Any]:
    scores = [float(selection["candidate_scores"][candidate]) for candidate in candidate_ids]
    unique_scores = sorted(set(scores))
    counts = [scores.count(value) for value in unique_scores]
    direction = selection["score_direction"]
    optimum = min(scores) if direction == "minimize" else max(scores)
    return {
        "unique_score_count": len(unique_scores),
        "unique_score_ratio": float(len(unique_scores) / len(scores)),
        "max_tie_group_size": int(max(counts)),
        "top_score_tie_group_size": int(scores.count(optimum)),
    }


def chosen_candidate(selection: dict[str, Any]) -> str:
    scores = selection["candidate_scores"]
    direction = selection["score_direction"]
    optimum = min(scores.values()) if direction == "minimize" else max(scores.values())
    return min(candidate for candidate, score in scores.items() if score == optimum)


def top_score_tie_candidates(selection: dict[str, Any]) -> set[str]:
    scores = selection["candidate_scores"]
    direction = selection["score_direction"]
    optimum = min(scores.values()) if direction == "minimize" else max(scores.values())
    return {str(candidate) for candidate, score in scores.items() if score == optimum}


def indices_with_cutoff_ties(values: np.ndarray, *, fraction: float, lower_is_better: bool) -> set[int]:
    if values.ndim != 1:
        raise ValueError("tie-aware top set requires a one-dimensional vector")
    if values.size == 0:
        raise ValueError("tie-aware top set requires at least one candidate")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("top-set fraction must lie in (0, 1]")
    order = np.argsort(values if lower_is_better else -values, kind="stable")
    cutoff = max(1, math.ceil(fraction * values.size))
    boundary = float(values[int(order[cutoff - 1])])
    if lower_is_better:
        return {int(index) for index, value in enumerate(values) if float(value) <= boundary}
    return {int(index) for index, value in enumerate(values) if float(value) >= boundary}


def top_fraction_hit_tie_aware(selected_index: int, true_risks: np.ndarray, *, fraction: float) -> bool:
    return selected_index in indices_with_cutoff_ties(
        true_risks,
        fraction=fraction,
        lower_is_better=True,
    )


def ndcg_at_k(predicted_risks: np.ndarray, true_risks: np.ndarray, *, k: int = 10) -> float | None:
    if predicted_risks.shape != true_risks.shape or predicted_risks.ndim != 1:
        raise ValueError("NDCG inputs must be aligned one-dimensional arrays")
    cutoff = min(k, predicted_risks.size)
    worst = float(np.max(true_risks))
    relevance = np.maximum(worst - true_risks, 0.0)
    discounts = 1.0 / np.log2(np.arange(2, cutoff + 2, dtype=np.float64))
    order = np.argsort(predicted_risks, kind="stable")[:cutoff]
    ideal = np.argsort(-relevance, kind="stable")[:cutoff]
    dcg = float(np.sum(relevance[order] * discounts))
    idcg = float(np.sum(relevance[ideal] * discounts))
    if idcg <= 1.0e-12:
        return None
    return dcg / idcg


def top_weighted_kendall(
    predicted_risks: np.ndarray,
    true_risks: np.ndarray,
    *,
    temperature: float = 10.0,
) -> float | None:
    if predicted_risks.shape != true_risks.shape or predicted_risks.ndim != 1:
        raise ValueError("top-weighted Kendall inputs must be aligned vectors")
    if predicted_risks.size < 2:
        return None
    if temperature <= 0:
        raise ValueError("top-weighted Kendall temperature must be positive")
    true_rank = np.empty(true_risks.size, dtype=np.int64)
    true_rank[np.argsort(true_risks, kind="stable")] = np.arange(true_risks.size)
    discordant_weight = 0.0
    total_weight = 0.0
    for i in range(true_risks.size):
        for j in range(i + 1, true_risks.size):
            true_delta = true_risks[i] - true_risks[j]
            if true_delta == 0.0:
                continue
            weight = abs(float(true_delta)) * math.exp(
                -float(min(true_rank[i], true_rank[j])) / temperature
            )
            if weight == 0.0:
                continue
            predicted_delta = predicted_risks[i] - predicted_risks[j]
            if predicted_delta * true_delta < 0.0:
                discordant_weight += weight
            total_weight += weight
    if total_weight <= 0.0:
        return None
    return 1.0 - 2.0 * discordant_weight / total_weight


def _pct_key(fraction: float) -> str:
    return f"{int(round(100.0 * fraction))}pct"


def shortlist_recall_metrics(
    predicted_risks: np.ndarray,
    true_risks: np.ndarray,
    *,
    fractions: tuple[float, ...] = SHORTLIST_RECALL_FRACTIONS,
) -> dict[str, float]:
    if predicted_risks.shape != true_risks.shape or predicted_risks.ndim != 1:
        raise ValueError("shortlist recall inputs must be aligned vectors")
    model_count = predicted_risks.size
    if model_count == 0:
        raise ValueError("shortlist recall requires at least one candidate")
    true_order = np.argsort(true_risks, kind="stable")
    oracle_index = int(true_order[0])
    top5_count = max(1, math.ceil(0.05 * model_count))
    top5_boundary = float(true_risks[int(true_order[top5_count - 1])])
    true_top5 = {
        int(index)
        for index, risk in enumerate(true_risks)
        if float(risk) <= top5_boundary
    }
    metrics: dict[str, float] = {}
    for fraction in fractions:
        if not 0.0 < fraction <= 1.0:
            raise ValueError("shortlist recall fractions must lie in (0, 1]")
        predicted_top = indices_with_cutoff_ties(
            predicted_risks,
            fraction=fraction,
            lower_is_better=True,
        )
        key = _pct_key(fraction)
        metrics[f"oracle_recall_at_{key}"] = float(oracle_index in predicted_top)
        metrics[f"top5_recall_at_{key}"] = float(len(predicted_top & true_top5) / len(true_top5))
        metrics[f"predicted_shortlist_size_at_{key}"] = float(len(predicted_top))
    return metrics


def evaluate_selection(selection: dict[str, Any], truth: dict[str, Any]) -> dict[str, Any]:
    task = str(selection["task"])
    if task != truth["task"]:
        raise ValueError("selection task does not match truth task")
    if truth.get("candidate_bank_sha256") is not None and selection.get(
        "candidate_bank_sha256"
    ) != truth.get("candidate_bank_sha256"):
        raise ValueError(f"candidate-bank hash mismatch for {task}/{selection['selector']}")
    truth_risks = truth["risks"]
    if set(selection["candidate_scores"]) != set(truth_risks):
        raise ValueError(f"candidate coverage mismatch for {task}/{selection['selector']}")
    if int(selection.get("candidate_count", len(truth_risks))) != len(truth_risks):
        raise ValueError(f"candidate_count mismatch for {task}/{selection['selector']}")

    candidate_ids = sorted(truth_risks)
    true_risks = np.asarray([truth_risks[candidate] for candidate in candidate_ids], dtype=np.float64)
    predicted_risks = score_to_predicted_risk(selection, candidate_ids)
    selected_id = str(selection["selected_candidate_id"])
    declared_best = chosen_candidate(selection)
    if selected_id != declared_best:
        raise ValueError(f"selected candidate is not optimal for declared scores: {selection['selection_path']}")
    selected_index = candidate_ids.index(selected_id)
    selected_risk = float(true_risks[selected_index])
    oracle_index = int(np.argmin(true_risks))
    top_tie_ids = top_score_tie_candidates(selection)
    top_tie_indices = [candidate_ids.index(candidate) for candidate in sorted(top_tie_ids)]
    top_tie_risks = np.asarray([true_risks[index] for index in top_tie_indices], dtype=np.float64)
    correlations = rank_correlations(predicted_risks, true_risks)
    fusion_config = selection.get("fusion_config")
    fusion_mode = fusion_config.get("fusion_mode") if isinstance(fusion_config, dict) else None
    report = {
        "task": task,
        "selector": str(selection["selector"]),
        "selection_path": selection["selection_path"],
        "candidate_bank_sha256": selection.get("candidate_bank_sha256"),
        "candidate_count": len(candidate_ids),
        "selected_candidate_id": selected_id,
        "oracle_candidate_id": candidate_ids[oracle_index],
        "selected_target_error": selected_risk,
        "selected_micro_f1": 1.0 - selected_risk,
        "oracle_target_error": float(true_risks[oracle_index]),
        "oracle_micro_f1": 1.0 - float(true_risks[oracle_index]),
        "oracle_micro_f1_gap": selected_risk - float(true_risks[oracle_index]),
        "oracle_micro_f1_gap_points": 100.0 * (selected_risk - float(true_risks[oracle_index])),
        "top_score_tie_best_normalized_regret": normalized_regret(float(np.min(top_tie_risks)), true_risks),
        "top_score_tie_worst_normalized_regret": normalized_regret(float(np.max(top_tie_risks)), true_risks),
        "top_score_tie_expected_normalized_regret": normalized_regret(float(np.mean(top_tie_risks)), true_risks),
        "normalized_regret": normalized_regret(selected_risk, true_risks),
        "kendall_tau": correlations["kendall_tau"],
        "spearman_rho": correlations["spearman_rho"],
        "top_weighted_kendall": top_weighted_kendall(predicted_risks, true_risks),
        "ndcg_at_10": ndcg_at_k(predicted_risks, true_risks, k=10),
        "top_5pct_hit": top_fraction_hit_tie_aware(selected_index, true_risks, fraction=0.05),
        "top_10pct_hit": top_fraction_hit_tie_aware(selected_index, true_risks, fraction=0.10),
        "top_20pct_hit": top_fraction_hit_tie_aware(selected_index, true_risks, fraction=0.20),
        "selector_runtime_seconds": selection.get("selector_runtime_seconds"),
        "fusion_mode": fusion_mode,
    }
    report.update(score_tie_diagnostics(selection, candidate_ids))
    report.update(shortlist_recall_metrics(predicted_risks, true_risks))
    return report


def aggregate_selector(reports: list[dict[str, Any]]) -> dict[str, Any]:
    regrets = [float(report["normalized_regret"]) for report in reports]
    runtimes = [
        float(report["selector_runtime_seconds"])
        for report in reports
        if report.get("selector_runtime_seconds") is not None
        and np.isfinite(float(report["selector_runtime_seconds"]))
    ]
    return {
        "selector": reports[0]["selector"],
        "task_count": len(reports),
        "candidate_counts": sorted({int(report["candidate_count"]) for report in reports}),
        "candidate_count_total": sum(int(report["candidate_count"]) for report in reports),
        "mean_normalized_regret": finite_mean(regrets),
        "cvar_20pct_normalized_regret": cvar_tail(regrets, fraction=0.20),
        "worst_normalized_regret": float(max(regrets)),
        "median_kendall_tau": finite_median([report["kendall_tau"] for report in reports]),
        "mean_top_weighted_kendall": finite_mean(
            [report["top_weighted_kendall"] for report in reports]
        ),
        "mean_ndcg_at_10": finite_mean([report["ndcg_at_10"] for report in reports]),
        "top_5pct_hit_rate": finite_mean([float(report["top_5pct_hit"]) for report in reports]),
        "top_10pct_hit_rate": finite_mean([float(report["top_10pct_hit"]) for report in reports]),
        "top_20pct_hit_rate": finite_mean([float(report["top_20pct_hit"]) for report in reports]),
        "mean_unique_score_ratio": finite_mean(
            [float(report["unique_score_ratio"]) for report in reports]
        ),
        "max_tie_group_size": int(max(int(report["max_tie_group_size"]) for report in reports)),
        "max_top_score_tie_group_size": int(
            max(int(report["top_score_tie_group_size"]) for report in reports)
        ),
        "mean_top_score_tie_best_normalized_regret": finite_mean(
            [report["top_score_tie_best_normalized_regret"] for report in reports]
        ),
        "mean_top_score_tie_worst_normalized_regret": finite_mean(
            [report["top_score_tie_worst_normalized_regret"] for report in reports]
        ),
        "mean_top_score_tie_expected_normalized_regret": finite_mean(
            [report["top_score_tie_expected_normalized_regret"] for report in reports]
        ),
        "fusion_modes": sorted(
            {str(report["fusion_mode"]) for report in reports if report.get("fusion_mode") is not None}
        ),
        **{
            f"mean_oracle_recall_at_{_pct_key(fraction)}": finite_mean(
                [float(report[f"oracle_recall_at_{_pct_key(fraction)}"]) for report in reports]
            )
            for fraction in SHORTLIST_RECALL_FRACTIONS
        },
        **{
            f"mean_top5_recall_at_{_pct_key(fraction)}": finite_mean(
                [float(report[f"top5_recall_at_{_pct_key(fraction)}"]) for report in reports]
            )
            for fraction in SHORTLIST_RECALL_FRACTIONS
        },
        **{
            f"mean_predicted_shortlist_size_at_{_pct_key(fraction)}": finite_mean(
                [float(report[f"predicted_shortlist_size_at_{_pct_key(fraction)}"]) for report in reports]
            )
            for fraction in SHORTLIST_RECALL_FRACTIONS
        },
        "mean_selected_micro_f1": finite_mean([report["selected_micro_f1"] for report in reports]),
        "mean_oracle_f1_gap_points": finite_mean(
            [report["oracle_micro_f1_gap_points"] for report in reports]
        ),
        "selector_runtime_seconds": float(np.sum(runtimes)) if runtimes else None,
        "tasks": reports,
    }


def discover_selection_paths(
    roots: list[Path],
    *,
    selectors: list[str] | None,
    tasks: list[str],
) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    found_pairs: set[tuple[str, str]] = set()
    for root in roots:
        root = root.resolve()
        candidates: list[Path] = []
        if root.is_file():
            candidates = [root]
        else:
            for task in tasks:
                task_root = root / task
                if selectors:
                    for selector in selectors:
                        path = task_root / f"{selector}.json"
                        if path.is_file():
                            candidates.append(path)
                            found_pairs.add((task, selector))
                else:
                    candidates.extend(sorted(task_root.glob("*.json")))
        for path in candidates:
            resolved = path.resolve()
            if resolved in seen:
                continue
            if not resolved.is_file():
                raise FileNotFoundError(f"missing selection JSON: {resolved}")
            seen.add(resolved)
            paths.append(resolved)
    if selectors:
        missing = [
            f"{task}/{selector}.json"
            for task in tasks
            for selector in selectors
            if (task, selector) not in found_pairs
        ]
        if missing:
            raise FileNotFoundError(
                "missing selection JSON after searching all roots: "
                + ", ".join(missing[:10])
                + (" ..." if len(missing) > 10 else "")
            )
    if not paths:
        raise FileNotFoundError("no selection JSON files discovered")
    return paths


def localized_gain_share(
    selector_reports: list[dict[str, Any]],
    transfer_reports: list[dict[str, Any]],
) -> float | None:
    transfer_by_task = {report["task"]: report for report in transfer_reports}
    gains = []
    for report in selector_reports:
        transfer = transfer_by_task.get(report["task"])
        if transfer is None:
            return None
        gains.append(max(0.0, float(transfer["normalized_regret"]) - float(report["normalized_regret"])))
    total = float(np.sum(gains))
    if total <= 1.0e-12:
        return 0.0
    return float(max(gains) / total)


def task_noninferiority_rate(
    selector_reports: list[dict[str, Any]],
    transfer_reports: list[dict[str, Any]],
) -> float | None:
    transfer_by_task = {report["task"]: report for report in transfer_reports}
    comparisons = []
    for report in selector_reports:
        transfer = transfer_by_task.get(report["task"])
        if transfer is None:
            return None
        comparisons.append(
            float(report["normalized_regret"])
            <= float(transfer["normalized_regret"]) + 1.0e-12
        )
    if not comparisons:
        return None
    return float(np.mean(comparisons))


def add_objective_guardrails(
    aggregates: dict[str, dict[str, Any]],
    *,
    objective_selector: str | None,
    transfer_selector: str,
    runtime_budget_seconds: float,
) -> dict[str, Any]:
    if objective_selector is None:
        non_transfer = [name for name in aggregates if name != transfer_selector]
        objective_selector = non_transfer[0] if len(non_transfer) == 1 else None
    if objective_selector is None or objective_selector not in aggregates:
        return {
            "objective_selector": objective_selector,
            "status": "not_evaluated",
            "reason": "pass --objective-selector when multiple non-transfer selectors are present",
        }
    candidate = aggregates[objective_selector]
    transfer = aggregates.get(transfer_selector)
    runtime = candidate.get("selector_runtime_seconds")
    runtime_pass = runtime is None or float(runtime) <= runtime_budget_seconds
    fusion_modes = set(candidate.get("fusion_modes") or [])
    shortlist_guardrail_required = bool(fusion_modes & SHORTLIST_FUSION_MODES)
    if not fusion_modes and str(objective_selector or "").lower().find("shortlist") >= 0:
        shortlist_guardrail_required = True
    guardrails: dict[str, Any] = {
        "objective_selector": objective_selector,
        "transfer_selector": transfer_selector,
        "runtime_budget_seconds": runtime_budget_seconds,
        "runtime_guardrail_pass": runtime_pass,
        "shortlist_guardrail_required": shortlist_guardrail_required,
        "gate_a_pass": (
            float(candidate["mean_normalized_regret"]) < 0.20
            and float(candidate["top_5pct_hit_rate"]) >= 0.50
        ),
    }
    if transfer is not None:
        guardrails.update(
            {
                "transfer_mean_normalized_regret": transfer["mean_normalized_regret"],
                "transfer_worst_normalized_regret": transfer["worst_normalized_regret"],
                "transfer_top_10pct_hit_rate": transfer["top_10pct_hit_rate"],
                "transfer_oracle_recall_at_10pct": transfer["mean_oracle_recall_at_10pct"],
                "transfer_top5_recall_at_10pct": transfer["mean_top5_recall_at_10pct"],
                "transfer_mean_selected_micro_f1": transfer["mean_selected_micro_f1"],
                "beats_transfer_mean_regret": (
                    float(candidate["mean_normalized_regret"])
                    < float(transfer["mean_normalized_regret"])
                ),
                "beats_transfer_selected_micro_f1": (
                    float(candidate["mean_selected_micro_f1"])
                    > float(transfer["mean_selected_micro_f1"])
                ),
                "open_dev_mean_regret_absolute_guardrail_pass": (
                    float(candidate["mean_normalized_regret"]) < 0.20
                ),
                "open_dev_mean_regret_absolute_max": 0.20,
                "open_dev_worst_regret_absolute_guardrail_pass": (
                    float(candidate["worst_normalized_regret"]) < 0.30
                ),
                "open_dev_worst_regret_absolute_max": 0.30,
                "top_10_guardrail_pass": (
                    float(candidate["top_10pct_hit_rate"])
                    >= float(transfer["top_10pct_hit_rate"])
                ),
                "worst_task_delta_vs_transfer": (
                    float(candidate["worst_normalized_regret"])
                    - float(transfer["worst_normalized_regret"])
                ),
                "worst_task_guardrail_pass": (
                    float(candidate["worst_normalized_regret"])
                    <= float(transfer["worst_normalized_regret"])
                ),
                "oracle_recall_10pct_guardrail_pass": (
                    float(candidate["mean_oracle_recall_at_10pct"])
                    >= float(transfer["mean_oracle_recall_at_10pct"])
                ),
                "top5_recall_10pct_guardrail_pass": (
                    float(candidate["mean_top5_recall_at_10pct"])
                    >= float(transfer["mean_top5_recall_at_10pct"])
                ),
                "oracle_recall_20pct_absolute_guardrail_pass": (
                    float(candidate["mean_oracle_recall_at_20pct"]) >= 0.75
                ),
                "oracle_recall_20pct_absolute_min": 0.75,
                "localized_gain_share_vs_transfer": localized_gain_share(
                    candidate["tasks"],
                    transfer["tasks"],
                ),
                "task_noninferiority_rate_vs_transfer": task_noninferiority_rate(
                    candidate["tasks"],
                    transfer["tasks"],
                ),
                "task_noninferiority_rate_min": 0.75,
            }
        )
        guardrails["task_noninferiority_guardrail_pass"] = (
            guardrails["task_noninferiority_rate_vs_transfer"] is not None
            and float(guardrails["task_noninferiority_rate_vs_transfer"]) >= 0.75
        )
        if not shortlist_guardrail_required:
            guardrails["shortlist_guardrail_status"] = "diagnostic_only"
        guardrails["gate_b_pass"] = (
            guardrails["beats_transfer_mean_regret"]
            and guardrails["beats_transfer_selected_micro_f1"]
        )
    else:
        guardrails.update(
            {
                "top_10_guardrail_pass": None,
                "worst_task_guardrail_pass": None,
                "gate_b_pass": False,
                "reason": f"transfer selector '{transfer_selector}' not found",
            }
        )
    guardrails["promotion_gate_pass"] = bool(
        guardrails["gate_a_pass"] or guardrails["gate_b_pass"]
    )
    guardrails["promotion_ready"] = bool(
        guardrails["runtime_guardrail_pass"]
        and guardrails["promotion_gate_pass"]
        and guardrails.get("open_dev_mean_regret_absolute_guardrail_pass") is not False
        and guardrails.get("open_dev_worst_regret_absolute_guardrail_pass") is not False
        and guardrails.get("task_noninferiority_guardrail_pass") is not False
        and (
            not shortlist_guardrail_required
            or guardrails.get("top_10_guardrail_pass") is not False
        )
        and (
            not shortlist_guardrail_required
            or guardrails.get("oracle_recall_20pct_absolute_guardrail_pass") is not False
        )
        and guardrails.get("worst_task_guardrail_pass") is not False
    )
    guardrails["autosota_primary_value"] = candidate["mean_normalized_regret"]
    return guardrails


def add_source_sim_guardrail(
    guardrails: dict[str, Any],
    *,
    report: dict[str, dict[str, Any]] | None,
    objective_selector: str | None,
    reference_selector: str,
    max_degradation: float,
) -> None:
    if report is None:
        guardrails["source_sim_cvar_guardrail_pass"] = None
        guardrails["source_sim_cvar_guardrail_status"] = "not_evaluated"
        return
    if objective_selector is None:
        guardrails["source_sim_cvar_guardrail_pass"] = False
        guardrails["source_sim_cvar_guardrail_status"] = "missing_objective_selector"
        return
    if objective_selector not in report or reference_selector not in report:
        guardrails["source_sim_cvar_guardrail_pass"] = False
        guardrails["source_sim_cvar_guardrail_status"] = "missing_selector"
        guardrails["source_sim_missing_selectors"] = sorted(
            {
                selector
                for selector in (objective_selector, reference_selector)
                if selector not in report
            }
        )
        return
    candidate_cvar = _source_sim_cvar(report[objective_selector], selector=objective_selector)
    reference_cvar = _source_sim_cvar(report[reference_selector], selector=reference_selector)
    allowed = reference_cvar * (1.0 + max_degradation)
    guardrails.update(
        {
            "source_sim_cvar_guardrail_status": "evaluated",
            "source_sim_reference_selector": reference_selector,
            "source_sim_cvar_degradation_max": max_degradation,
            "source_sim_candidate_cvar": candidate_cvar,
            "source_sim_reference_cvar": reference_cvar,
            "source_sim_allowed_cvar": allowed,
            "source_sim_cvar_relative_degradation": (
                (candidate_cvar - reference_cvar) / max(reference_cvar, 1.0e-12)
            ),
            "source_sim_cvar_guardrail_pass": candidate_cvar <= allowed,
        }
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    tasks = args.tasks or list(OPEN_DEVELOPMENT_TASKS)
    truth = load_open_dev_truth(args.dev_truth_report.resolve())
    missing_truth = sorted(set(tasks) - set(truth))
    if missing_truth:
        raise ValueError(f"open-dev truth report is missing tasks: {missing_truth}")
    selection_paths = discover_selection_paths(
        args.selection_root,
        selectors=args.selector,
        tasks=tasks,
    )
    task_selector_reports: dict[str, list[dict[str, Any]]] = {}
    for path in selection_paths:
        selection = load_selection(path)
        task = str(selection["task"])
        if task not in tasks:
            continue
        report = evaluate_selection(selection, truth[task])
        task_selector_reports.setdefault(report["selector"], []).append(report)
    expected_tasks = set(tasks)
    aggregates: dict[str, dict[str, Any]] = {}
    for selector, reports in sorted(task_selector_reports.items()):
        covered = {report["task"] for report in reports}
        if covered != expected_tasks:
            raise ValueError(
                f"selector {selector} does not cover the open-dev task set; "
                f"missing={sorted(expected_tasks - covered)} extra={sorted(covered - expected_tasks)}"
            )
        aggregate = aggregate_selector(reports)
        if args.expected_candidate_count is not None and aggregate["candidate_counts"] != [
            args.expected_candidate_count
        ]:
            raise ValueError(
                f"selector {selector} candidate counts {aggregate['candidate_counts']} "
                f"do not match expected {args.expected_candidate_count}"
            )
        aggregates[selector] = aggregate
    guardrails = add_objective_guardrails(
        aggregates,
        objective_selector=args.objective_selector,
        transfer_selector=args.transfer_selector,
        runtime_budget_seconds=args.runtime_budget_seconds,
    )
    add_source_sim_guardrail(
        guardrails,
        report=(
            load_source_sim_report(getattr(args, "source_sim_report").resolve())
            if getattr(args, "source_sim_report", None) is not None
            else None
        ),
        objective_selector=guardrails.get("objective_selector"),
        reference_selector=getattr(args, "source_sim_reference_selector", "spectra_v2"),
        max_degradation=getattr(
            args,
            "source_sim_cvar_degradation_max",
            DEFAULT_SOURCE_SIM_CVAR_DEGRADATION_MAX,
        ),
    )
    if "promotion_ready" in guardrails:
        guardrails["promotion_ready"] = bool(
            guardrails["promotion_ready"]
            and guardrails.get("source_sim_cvar_guardrail_pass") is not False
        )
    result = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "objective_source": "open real-target development candidate truth; no sealed_eval import",
        "open_development_tasks": tasks,
        "selector_count": len(aggregates),
        "selectors": aggregates,
        "guardrails": guardrails,
        "label_access_count": 0,
        "protocol_violation_count": 0,
        "objective_runtime_seconds": time.perf_counter() - started,
    }
    if args.output is not None:
        atomic_json(result, args.output.resolve())
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-truth-report", type=Path, required=True)
    parser.add_argument("--selection-root", type=Path, action="append", required=True)
    parser.add_argument("--selector", action="append")
    parser.add_argument("--tasks", nargs="*")
    parser.add_argument("--objective-selector")
    parser.add_argument("--transfer-selector", default="transfer_score")
    parser.add_argument("--expected-candidate-count", type=int, default=None)
    parser.add_argument("--runtime-budget-seconds", type=float, default=DEFAULT_RUNTIME_BUDGET_SECONDS)
    parser.add_argument(
        "--source-sim-report",
        type=Path,
        help=(
            "Optional source-simulated leave-one-shift-family report used only "
            "for a CVaR guardrail. This must not be a sealed/final-label path."
        ),
    )
    parser.add_argument("--source-sim-reference-selector", default="spectra_v2")
    parser.add_argument(
        "--source-sim-cvar-degradation-max",
        type=float,
        default=DEFAULT_SOURCE_SIM_CVAR_DEGRADATION_MAX,
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    result = run(parse_args())
    summary = {
        "selector_count": result["selector_count"],
        "guardrails": result["guardrails"],
        "label_access_count": result["label_access_count"],
        "protocol_violation_count": result["protocol_violation_count"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
