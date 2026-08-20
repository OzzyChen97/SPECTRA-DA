from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest

from selector.loto_open_dev_selection import run


def _task_report(task: str, regret: float, transfer_regret: float | None = None) -> dict:
    return {
        "task": task,
        "selector": "selector",
        "candidate_count": 10,
        "selected_candidate_id": f"{task}_selected",
        "oracle_candidate_id": f"{task}_oracle",
        "normalized_regret": regret,
        "selected_micro_f1": 1.0 - 0.1 * regret,
        "oracle_recall_at_20pct": 1.0 if regret <= 0.2 else 0.0,
        "fusion_mode": None,
        "transfer_regret": transfer_regret,
    }


def _selector(name: str, regrets: dict[str, float], *, fusion_mode: str | None = None) -> dict:
    tasks = []
    for task, regret in regrets.items():
        report = _task_report(task, regret)
        report["selector"] = name
        if fusion_mode is not None:
            report["fusion_mode"] = fusion_mode
        tasks.append(report)
    return {
        "selector": name,
        "tasks": tasks,
        "fusion_modes": [fusion_mode] if fusion_mode is not None else [],
    }


def test_loto_selection_uses_only_training_tasks(tmp_path: Path) -> None:
    tasks = ["t1", "t2", "t3", "t4"]
    report = {
        "schema_version": 1,
        "open_development_tasks": tasks,
        "selectors": {
            "transfer_score": _selector(
                "transfer_score",
                {"t1": 0.20, "t2": 0.20, "t3": 0.20, "t4": 0.20},
            ),
            "stable": _selector(
                "stable",
                {"t1": 0.10, "t2": 0.10, "t3": 0.10, "t4": 0.10},
            ),
            "leaky_best_on_t4": _selector(
                "leaky_best_on_t4",
                {"t1": 0.19, "t2": 0.19, "t3": 0.19, "t4": 0.00},
            ),
        },
        "label_access_count": 0,
        "protocol_violation_count": 0,
    }
    path = tmp_path / "objective.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    result = run(
        Namespace(
            objective_report=path,
            transfer_selector="transfer_score",
            selector=["stable", "leaky_best_on_t4"],
            mean_regret_max=0.20,
            worst_regret_max=0.30,
            task_noninferiority_min=0.75,
            shortlist_recall20_min=0.75,
            output=None,
        )
    )

    heldout_t4 = next(fold for fold in result["folds"] if fold["heldout_task"] == "t4")
    assert heldout_t4["selected_selector"] == "stable"
    assert result["aggregate"]["promotion_ready"]


def test_loto_selection_reports_near_miss_not_ready(tmp_path: Path) -> None:
    tasks = ["t1", "t2", "t3", "t4"]
    report = {
        "schema_version": 1,
        "open_development_tasks": tasks,
        "selectors": {
            "transfer_score": _selector(
                "transfer_score",
                {"t1": 0.01, "t2": 0.02, "t3": 0.60, "t4": 0.60},
            ),
            "localized": _selector(
                "localized",
                {"t1": 0.08, "t2": 0.05, "t3": 0.00, "t4": 0.20},
                fusion_mode="spectra_shortlist_transfer_rerank",
            ),
        },
        "label_access_count": 0,
        "protocol_violation_count": 0,
    }
    path = tmp_path / "objective.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    result = run(
        Namespace(
            objective_report=path,
            transfer_selector="transfer_score",
            selector=["localized"],
            mean_regret_max=0.20,
            worst_regret_max=0.30,
            task_noninferiority_min=0.75,
            shortlist_recall20_min=0.75,
            output=None,
        )
    )

    assert result["aggregate"]["mean_validation_normalized_regret"] < 0.20
    assert result["aggregate"]["validation_task_noninferiority_rate_vs_transfer"] == pytest.approx(0.5)
    assert not result["aggregate"]["promotion_ready"]

