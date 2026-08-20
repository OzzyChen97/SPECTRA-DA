from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest

from selector.selector_complementarity import load_json, run


def _selection(
    task: str,
    selector: str,
    scores: dict[str, float],
    direction: str = "minimize",
) -> dict:
    selected = min(scores, key=scores.get) if direction == "minimize" else max(scores, key=scores.get)
    return {
        "schema_version": 1,
        "created_at": "2026-08-20T00:00:00+00:00",
        "task": task,
        "selector": selector,
        "candidate_bank_sha256": "bank",
        "candidate_count": len(scores),
        "candidate_scores": scores,
        "score_direction": direction,
        "score_semantics": "ranking",
        "selected_candidate_id": selected,
        "label_access_count": 0,
        "protocol_violation_count": 0,
    }


def _write_selection(root: Path, task: str, selector: str, scores: dict[str, float]) -> None:
    directory = root / task
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{selector}.json").write_text(
        json.dumps(_selection(task, selector, scores)),
        encoding="utf-8",
    )


def test_complementarity_reports_overlap_and_selected_cross_ranks(tmp_path: Path) -> None:
    root = tmp_path / "selections"
    task = "A_to_B"
    scores_a = {
        "A_to_B__M1__cfg__seed-1__epoch-0010": 0.0,
        "A_to_B__M2__cfg__seed-1__epoch-0020": 1.0,
        "A_to_B__M3__cfg__seed-1__epoch-0030": 2.0,
        "A_to_B__M4__cfg__seed-1__epoch-0040": 3.0,
    }
    scores_b = {
        "A_to_B__M1__cfg__seed-1__epoch-0010": 3.0,
        "A_to_B__M2__cfg__seed-1__epoch-0020": 2.0,
        "A_to_B__M3__cfg__seed-1__epoch-0030": 1.0,
        "A_to_B__M4__cfg__seed-1__epoch-0040": 0.0,
    }
    _write_selection(root, task, "selector_a", scores_a)
    _write_selection(root, task, "selector_b", scores_b)

    result = run(
        Namespace(
            selection_root=[root],
            objective_report=None,
            selector=["selector_a", "selector_b"],
            task=[task],
            transfer_selector="selector_a",
            output=None,
        )
    )

    diagnostic = result["task_diagnostics"][0]
    pair = diagnostic["pairwise"][0]
    assert pair["same_selected_candidate"] is False
    assert pair["rank_spearman"] == pytest.approx(-1.0)
    assert pair["first_selected_rank_under_second"] == pytest.approx(1.0)
    assert pair["second_selected_rank_under_first"] == pytest.approx(1.0)
    assert pair["top_20pct_jaccard"] == pytest.approx(0.0)
    assert diagnostic["selectors"]["selector_a"]["selected_candidate_metadata"]["method"] == "M1"
    assert diagnostic["selectors"]["selector_b"]["top_profiles"]["20pct"]["method_counts"] == {"M4": 1}
    assert result["label_access_count"] == 0
    assert result["protocol_violation_count"] == 0


def test_complementarity_attaches_open_dev_metrics_without_reading_labels(tmp_path: Path) -> None:
    root = tmp_path / "selections"
    task = "A_to_B"
    scores = {"a": 0.0, "b": 1.0, "c": 2.0}
    _write_selection(root, task, "transfer_score", scores)
    _write_selection(root, task, "candidate", {"a": 1.0, "b": 0.0, "c": 2.0})
    report = {
        "schema_version": 1,
        "open_development_tasks": [task],
        "selectors": {
            "transfer_score": {
                "tasks": [
                    {
                        "task": task,
                        "normalized_regret": 0.20,
                        "selected_micro_f1": 0.80,
                        "oracle_recall_at_20pct": 1.0,
                    }
                ]
            },
            "candidate": {
                "tasks": [
                    {
                        "task": task,
                        "normalized_regret": 0.10,
                        "selected_micro_f1": 0.90,
                        "oracle_recall_at_20pct": 1.0,
                    }
                ]
            },
        },
        "label_access_count": 0,
        "protocol_violation_count": 0,
    }
    report_path = tmp_path / "objective.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    result = run(
        Namespace(
            selection_root=[root],
            objective_report=[report_path],
            selector=["transfer_score", "candidate"],
            task=None,
            transfer_selector="transfer_score",
            output=None,
        )
    )

    metrics = result["task_diagnostics"][0]["selectors"]["candidate"]["open_dev_metrics"]
    assert metrics["normalized_regret"] == pytest.approx(0.10)
    assert metrics["noninferior_to_transfer"] is True


def test_complementarity_rejects_protocol_violating_selector(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    document = _selection("A_to_B", "bad", {"a": 0.0})
    document["protocol_violation_count"] = 1
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(RuntimeError, match="protocol violations"):
        load_json(path)
