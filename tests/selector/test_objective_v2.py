from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest

from selector.objective_v2 import (
    discover_selection_paths,
    load_open_dev_truth,
    run,
    shortlist_recall_metrics,
)

import numpy as np


def _write_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document), encoding="utf-8")


def _selection(
    *,
    task: str,
    selector: str,
    scores: dict[str, float],
    direction: str = "minimize",
    fusion_config: dict | None = None,
) -> dict:
    optimum = min(scores.values()) if direction == "minimize" else max(scores.values())
    selected = min(candidate for candidate, value in scores.items() if value == optimum)
    document = {
        "schema_version": 1,
        "task": task,
        "selector": selector,
        "candidate_bank_sha256": f"bank-{task}",
        "candidate_count": len(scores),
        "candidate_scores": scores,
        "score_direction": direction,
        "score_semantics": "ranking",
        "selected_candidate_id": selected,
        "label_access_count": 0,
        "protocol_violation_count": 0,
        "selector_runtime_seconds": 1.0,
    }
    if fusion_config is not None:
        document["fusion_config"] = fusion_config
    return document


def test_objective_v2_evaluates_open_dev_rank_fusion_inputs(tmp_path: Path) -> None:
    tasks = ["ACMv9_to_Citationv1", "USA_to_BRAZIL"]
    candidates = ["best", "middle", "bad"]
    truth = {
        "schema_version": 1,
        "tasks": [
            {
                "task": task,
                "candidate_bank_sha256": f"bank-{task}",
                "candidate_truth": {
                    "best": {"target_error": 0.10},
                    "middle": {"target_error": 0.20},
                    "bad": {"target_error": 0.50},
                },
            }
            for task in tasks
        ],
    }
    truth_path = tmp_path / "open_dev_truth.json"
    _write_json(truth_path, truth)

    spectra_root = tmp_path / "spectra"
    transfer_root = tmp_path / "transfer"
    for task in tasks:
        _write_json(
            spectra_root / task / "spectra_trust.json",
            _selection(
                task=task,
                selector="spectra_trust",
                scores={"best": 0.0, "middle": 0.5, "bad": 1.0},
            ),
        )
        _write_json(
            transfer_root / task / "transfer_score.json",
            _selection(
                task=task,
                selector="transfer_score",
                scores={"best": 0.2, "middle": 0.9, "bad": 0.1},
                direction="maximize",
            ),
        )

    result = run(
        Namespace(
            dev_truth_report=truth_path,
            selection_root=[spectra_root, transfer_root],
            selector=None,
            tasks=tasks,
            objective_selector="spectra_trust",
            transfer_selector="transfer_score",
            expected_candidate_count=3,
            runtime_budget_seconds=350.0,
            output=None,
        )
    )

    assert result["label_access_count"] == 0
    assert result["protocol_violation_count"] == 0
    assert result["selectors"]["spectra_trust"]["mean_normalized_regret"] == 0.0
    assert result["selectors"]["spectra_trust"]["top_10pct_hit_rate"] == 1.0
    assert result["selectors"]["spectra_trust"]["mean_oracle_recall_at_10pct"] == 1.0
    assert result["selectors"]["spectra_trust"]["mean_top5_recall_at_10pct"] == 1.0
    assert result["guardrails"]["oracle_recall_10pct_guardrail_pass"]
    assert result["guardrails"]["top5_recall_10pct_guardrail_pass"]
    assert result["guardrails"]["beats_transfer_mean_regret"]
    assert result["guardrails"]["worst_task_guardrail_pass"]
    assert result["guardrails"]["source_sim_cvar_guardrail_pass"] is None
    assert result["guardrails"]["gate_b_pass"]


def test_shortlist_recall_tracks_oracle_trajectory_separately() -> None:
    candidate_ids = [
        "A_to_B__M1__c1__seed-1__epoch-0010",
        "A_to_B__M1__c1__seed-1__epoch-0020",
        "A_to_B__M2__c2__seed-2__epoch-0010",
        "A_to_B__M3__c3__seed-3__epoch-0010",
    ]
    true_risks = np.asarray([0.1, 0.2, 0.3, 0.4])
    predicted_risks = np.asarray([1.0, 0.0, 2.0, 3.0])

    metrics = shortlist_recall_metrics(
        predicted_risks,
        true_risks,
        candidate_ids=candidate_ids,
        fractions=(0.25,),
    )

    assert metrics["oracle_recall_at_25pct"] == 0.0
    assert metrics["oracle_trajectory_recall_at_25pct"] == 1.0


def test_discover_selection_paths_searches_across_roots(tmp_path: Path) -> None:
    task = "ACMv9_to_Citationv1"
    baseline_root = tmp_path / "baseline"
    repaired_root = tmp_path / "repaired"
    _write_json(
        baseline_root / task / "transfer_score.json",
        _selection(
            task=task,
            selector="transfer_score",
            scores={"a": 0.1, "b": 0.2},
            direction="maximize",
        ),
    )
    _write_json(
        repaired_root / task / "spectra_reliable.json",
        _selection(
            task=task,
            selector="spectra_reliable",
            scores={"a": 0.1, "b": 0.2},
        ),
    )

    paths = discover_selection_paths(
        [baseline_root, repaired_root],
        selectors=["transfer_score", "spectra_reliable"],
        tasks=[task],
    )

    assert sorted(path.name for path in paths) == [
        "spectra_reliable.json",
        "transfer_score.json",
    ]


def test_objective_v2_all_selector_diagnostic_without_objective_selector(tmp_path: Path) -> None:
    task = "ACMv9_to_Citationv1"
    truth = {
        "schema_version": 1,
        "tasks": [
            {
                "task": task,
                "candidate_bank_sha256": f"bank-{task}",
                "candidate_truth": {
                    "best": {"target_error": 0.10},
                    "bad": {"target_error": 0.50},
                },
            }
        ],
    }
    truth_path = tmp_path / "open_dev_truth.json"
    _write_json(truth_path, truth)
    root = tmp_path / "selectors"
    for selector in ("transfer_score", "agreement_reference", "spectra_reliable"):
        _write_json(
            root / task / f"{selector}.json",
            _selection(
                task=task,
                selector=selector,
                scores={"best": 0.0, "bad": 1.0},
            ),
        )

    result = run(
        Namespace(
            dev_truth_report=truth_path,
            selection_root=[root],
            selector=["transfer_score", "agreement_reference", "spectra_reliable"],
            tasks=[task],
            objective_selector=None,
            transfer_selector="transfer_score",
            expected_candidate_count=2,
            runtime_budget_seconds=350.0,
            output=None,
        )
    )

    assert result["selector_count"] == 3
    assert result["guardrails"]["status"] == "not_evaluated"


def test_objective_v2_flags_worst_task_regression_vs_transfer_score(tmp_path: Path) -> None:
    tasks = ["ACMv9_to_Citationv1", "USA_to_BRAZIL"]
    truth = {
        "schema_version": 1,
        "tasks": [
            {
                "task": task,
                "candidate_bank_sha256": f"bank-{task}",
                "candidate_truth": {
                    "best": {"target_error": 0.10},
                    "middle": {"target_error": 0.20},
                    "bad": {"target_error": 0.50},
                },
            }
            for task in tasks
        ],
    }
    truth_path = tmp_path / "open_dev_truth.json"
    _write_json(truth_path, truth)

    spectra_root = tmp_path / "spectra"
    transfer_root = tmp_path / "transfer"
    _write_json(
        spectra_root / tasks[0] / "spectra_trust.json",
        _selection(
            task=tasks[0],
            selector="spectra_trust",
            scores={"best": 1.0, "middle": 0.5, "bad": 0.0},
        ),
    )
    _write_json(
        spectra_root / tasks[1] / "spectra_trust.json",
        _selection(
            task=tasks[1],
            selector="spectra_trust",
            scores={"best": 0.0, "middle": 0.5, "bad": 1.0},
        ),
    )
    for task in tasks:
        _write_json(
            transfer_root / task / "transfer_score.json",
            _selection(
                task=task,
                selector="transfer_score",
                scores={"best": 0.9, "middle": 0.8, "bad": 0.1},
                direction="maximize",
            ),
        )

    result = run(
        Namespace(
            dev_truth_report=truth_path,
            selection_root=[spectra_root, transfer_root],
            selector=None,
            tasks=tasks,
            objective_selector="spectra_trust",
            transfer_selector="transfer_score",
            expected_candidate_count=3,
            runtime_budget_seconds=350.0,
            output=None,
        )
    )

    assert result["selectors"]["spectra_trust"]["worst_normalized_regret"] == pytest.approx(1.0)
    assert result["selectors"]["transfer_score"]["worst_normalized_regret"] == 0.0
    assert result["guardrails"]["worst_task_delta_vs_transfer"] == pytest.approx(1.0)
    assert not result["guardrails"]["worst_task_guardrail_pass"]
    assert not result["guardrails"]["promotion_ready"]


def test_objective_v2_applies_source_sim_cvar_guardrail(tmp_path: Path) -> None:
    tasks = ["ACMv9_to_Citationv1", "USA_to_BRAZIL"]
    truth = {
        "schema_version": 1,
        "tasks": [
            {
                "task": task,
                "candidate_bank_sha256": f"bank-{task}",
                "candidate_truth": {
                    "best": {"target_error": 0.10},
                    "middle": {"target_error": 0.20},
                    "bad": {"target_error": 0.50},
                },
            }
            for task in tasks
        ],
    }
    truth_path = tmp_path / "open_dev_truth.json"
    _write_json(truth_path, truth)
    source_sim_report = {
        "schema_version": 1,
        "label_access_count": 0,
        "protocol_violation_count": 0,
        "selectors": {
            "spectra_trust": {"cvar_20pct_normalized_regret": 1.06},
            "spectra_v2": {"cvar_20pct_normalized_regret": 1.00},
        },
    }
    source_sim_path = tmp_path / "source_sim_report.json"
    _write_json(source_sim_path, source_sim_report)

    spectra_root = tmp_path / "spectra"
    transfer_root = tmp_path / "transfer"
    for task in tasks:
        _write_json(
            spectra_root / task / "spectra_trust.json",
            _selection(
                task=task,
                selector="spectra_trust",
                scores={"best": 0.0, "middle": 0.5, "bad": 1.0},
            ),
        )
        _write_json(
            transfer_root / task / "transfer_score.json",
            _selection(
                task=task,
                selector="transfer_score",
                scores={"best": 0.9, "middle": 0.8, "bad": 0.1},
                direction="maximize",
            ),
        )

    result = run(
        Namespace(
            dev_truth_report=truth_path,
            selection_root=[spectra_root, transfer_root],
            selector=None,
            tasks=tasks,
            objective_selector="spectra_trust",
            transfer_selector="transfer_score",
            expected_candidate_count=3,
            runtime_budget_seconds=350.0,
            source_sim_report=source_sim_path,
            source_sim_reference_selector="spectra_v2",
            source_sim_cvar_degradation_max=0.05,
            output=None,
        )
    )

    guardrails = result["guardrails"]
    assert guardrails["source_sim_cvar_guardrail_status"] == "evaluated"
    assert guardrails["source_sim_candidate_cvar"] == pytest.approx(1.06)
    assert guardrails["source_sim_reference_cvar"] == pytest.approx(1.00)
    assert guardrails["source_sim_allowed_cvar"] == pytest.approx(1.05)
    assert not guardrails["source_sim_cvar_guardrail_pass"]
    assert not guardrails["promotion_ready"]


def test_objective_v2_promotes_gate_b_even_when_gate_a_top5_fails(tmp_path: Path) -> None:
    task = "ACMv9_to_Citationv1"
    candidates = [f"c{i:02d}" for i in range(20)]
    truth = {
        "schema_version": 1,
        "tasks": [
            {
                "task": task,
                "candidate_bank_sha256": f"bank-{task}",
                "candidate_truth": {
                    candidate: {"target_error": 0.05 + 0.02 * index}
                    for index, candidate in enumerate(candidates)
                },
            }
        ],
    }
    truth_path = tmp_path / "open_dev_truth.json"
    _write_json(truth_path, truth)

    spectra_scores = {candidate: 10.0 + index for index, candidate in enumerate(candidates)}
    spectra_scores["c01"] = 0.0
    spectra_scores["c00"] = 0.1
    transfer_scores = {candidate: 20.0 - index for index, candidate in enumerate(candidates)}
    transfer_scores["c09"] = 100.0
    transfer_scores["c00"] = -100.0

    spectra_root = tmp_path / "spectra"
    transfer_root = tmp_path / "transfer"
    _write_json(
        spectra_root / task / "spectra_trust.json",
        _selection(task=task, selector="spectra_trust", scores=spectra_scores),
    )
    _write_json(
        transfer_root / task / "transfer_score.json",
        _selection(
            task=task,
            selector="transfer_score",
            scores=transfer_scores,
            direction="maximize",
        ),
    )

    result = run(
        Namespace(
            dev_truth_report=truth_path,
            selection_root=[spectra_root, transfer_root],
            selector=None,
            tasks=[task],
            objective_selector="spectra_trust",
            transfer_selector="transfer_score",
            expected_candidate_count=20,
            runtime_budget_seconds=350.0,
            output=None,
        )
    )

    guardrails = result["guardrails"]
    assert not guardrails["gate_a_pass"]
    assert guardrails["gate_b_pass"]
    assert guardrails["promotion_gate_pass"]
    assert guardrails["top_10_guardrail_pass"]
    assert guardrails["worst_task_guardrail_pass"]
    assert guardrails["source_sim_cvar_guardrail_pass"] is None
    assert guardrails["promotion_ready"]


def test_objective_v2_reports_shortlist_recall_without_top1_success(tmp_path: Path) -> None:
    task = "ACMv9_to_Citationv1"
    candidates = [f"c{i:02d}" for i in range(20)]
    truth = {
        "schema_version": 1,
        "tasks": [
            {
                "task": task,
                "candidate_bank_sha256": f"bank-{task}",
                "candidate_truth": {
                    candidate: {"target_error": 0.05 + 0.02 * index}
                    for index, candidate in enumerate(candidates)
                },
            }
        ],
    }
    truth_path = tmp_path / "open_dev_truth.json"
    _write_json(truth_path, truth)

    spectra_scores = {candidate: 10.0 + index for index, candidate in enumerate(candidates)}
    spectra_scores["c09"] = 0.0
    spectra_scores["c00"] = 0.1
    transfer_scores = {candidate: 20.0 - index for index, candidate in enumerate(candidates)}
    transfer_scores["c09"] = 100.0
    transfer_scores["c00"] = -100.0

    spectra_root = tmp_path / "spectra"
    transfer_root = tmp_path / "transfer"
    _write_json(
        spectra_root / task / "spectra_trust.json",
        _selection(task=task, selector="spectra_trust", scores=spectra_scores),
    )
    _write_json(
        transfer_root / task / "transfer_score.json",
        _selection(
            task=task,
            selector="transfer_score",
            scores=transfer_scores,
            direction="maximize",
        ),
    )

    result = run(
        Namespace(
            dev_truth_report=truth_path,
            selection_root=[spectra_root, transfer_root],
            selector=None,
            tasks=[task],
            objective_selector="spectra_trust",
            transfer_selector="transfer_score",
            expected_candidate_count=20,
            runtime_budget_seconds=350.0,
            output=None,
        )
    )

    spectra = result["selectors"]["spectra_trust"]
    transfer = result["selectors"]["transfer_score"]
    task_report = spectra["tasks"][0]
    assert spectra["mean_normalized_regret"] > 0.0
    assert task_report["selected_candidate_id"] == "c09"
    assert task_report["oracle_candidate_id"] == "c00"
    assert task_report["oracle_recall_at_5pct"] == 0.0
    assert task_report["oracle_recall_at_10pct"] == 1.0
    assert spectra["mean_oracle_recall_at_10pct"] == 1.0
    assert spectra["mean_top5_recall_at_10pct"] == 1.0
    assert transfer["mean_oracle_recall_at_10pct"] == 0.0
    assert result["guardrails"]["oracle_recall_10pct_guardrail_pass"]
    assert result["guardrails"]["top5_recall_10pct_guardrail_pass"]


def test_direct_selector_is_not_rejected_by_shortlist_guardrail(tmp_path: Path) -> None:
    task = "ACMv9_to_Citationv1"
    candidates = [f"c{i:02d}" for i in range(20)]
    truth = {
        "schema_version": 1,
        "tasks": [
            {
                "task": task,
                "candidate_bank_sha256": f"bank-{task}",
                "candidate_truth": {
                    candidate: {"target_error": 0.05 + 0.02 * index}
                    for index, candidate in enumerate(candidates)
                },
            }
        ],
    }
    truth_path = tmp_path / "open_dev_truth.json"
    _write_json(truth_path, truth)

    candidate_scores = {candidate: 100.0 + index for index, candidate in enumerate(candidates)}
    candidate_scores["c03"] = 0.0
    candidate_scores["c04"] = 0.1
    candidate_scores["c05"] = 0.2
    candidate_scores["c06"] = 0.3
    transfer_scores = {candidate: 10.0 + index for index, candidate in enumerate(candidates)}
    transfer_scores["c19"] = 100.0
    transfer_scores["c00"] = 99.0

    spectra_root = tmp_path / "spectra"
    transfer_root = tmp_path / "transfer"
    _write_json(
        spectra_root / task / "agreement_reference.json",
        _selection(task=task, selector="agreement_reference", scores=candidate_scores),
    )
    _write_json(
        transfer_root / task / "transfer_score.json",
        _selection(
            task=task,
            selector="transfer_score",
            scores=transfer_scores,
            direction="maximize",
        ),
    )

    result = run(
        Namespace(
            dev_truth_report=truth_path,
            selection_root=[spectra_root, transfer_root],
            selector=None,
            tasks=[task],
            objective_selector="agreement_reference",
            transfer_selector="transfer_score",
            expected_candidate_count=20,
            runtime_budget_seconds=350.0,
            output=None,
        )
    )

    guardrails = result["guardrails"]
    assert not guardrails["oracle_recall_10pct_guardrail_pass"]
    assert guardrails["shortlist_guardrail_required"] is False
    assert guardrails["shortlist_guardrail_status"] == "diagnostic_only"
    assert guardrails["gate_b_pass"]
    assert guardrails["promotion_ready"]


def test_shortlist_selector_is_rejected_by_shortlist_guardrail(tmp_path: Path) -> None:
    task = "ACMv9_to_Citationv1"
    candidates = [f"c{i:02d}" for i in range(20)]
    truth = {
        "schema_version": 1,
        "tasks": [
            {
                "task": task,
                "candidate_bank_sha256": f"bank-{task}",
                "candidate_truth": {
                    candidate: {"target_error": 0.05 + 0.02 * index}
                    for index, candidate in enumerate(candidates)
                },
            }
        ],
    }
    truth_path = tmp_path / "open_dev_truth.json"
    _write_json(truth_path, truth)

    candidate_scores = {candidate: 100.0 + index for index, candidate in enumerate(candidates)}
    candidate_scores["c03"] = 0.0
    candidate_scores["c04"] = 0.1
    candidate_scores["c05"] = 0.2
    candidate_scores["c06"] = 0.3
    transfer_scores = {candidate: 10.0 + index for index, candidate in enumerate(candidates)}
    transfer_scores["c19"] = 100.0
    transfer_scores["c00"] = 99.0

    spectra_root = tmp_path / "spectra"
    transfer_root = tmp_path / "transfer"
    _write_json(
        spectra_root / task / "spectra_reliable_tsr.json",
        _selection(
            task=task,
            selector="spectra_reliable_tsr",
            scores=candidate_scores,
            fusion_config={"fusion_mode": "transfer_shortlist_spectra_rerank"},
        ),
    )
    _write_json(
        transfer_root / task / "transfer_score.json",
        _selection(
            task=task,
            selector="transfer_score",
            scores=transfer_scores,
            direction="maximize",
        ),
    )

    result = run(
        Namespace(
            dev_truth_report=truth_path,
            selection_root=[spectra_root, transfer_root],
            selector=None,
            tasks=[task],
            objective_selector="spectra_reliable_tsr",
            transfer_selector="transfer_score",
            expected_candidate_count=20,
            runtime_budget_seconds=350.0,
            output=None,
        )
    )

    guardrails = result["guardrails"]
    assert not guardrails["oracle_recall_10pct_guardrail_pass"]
    assert guardrails["shortlist_guardrail_required"] is True
    assert guardrails["gate_b_pass"]
    assert not guardrails["promotion_ready"]


def test_objective_v2_reports_tie_diagnostics_and_tie_aware_shortlists(tmp_path: Path) -> None:
    task = "ACMv9_to_Citationv1"
    truth = {
        "schema_version": 1,
        "tasks": [
            {
                "task": task,
                "candidate_bank_sha256": f"bank-{task}",
                "candidate_truth": {
                    "best": {"target_error": 0.10},
                    "tied_worse": {"target_error": 0.20},
                    "middle": {"target_error": 0.30},
                    "bad": {"target_error": 0.50},
                },
            }
        ],
    }
    truth_path = tmp_path / "open_dev_truth.json"
    _write_json(truth_path, truth)

    spectra_root = tmp_path / "spectra"
    transfer_root = tmp_path / "transfer"
    _write_json(
        spectra_root / task / "agreement_reference.json",
        _selection(
            task=task,
            selector="agreement_reference",
            scores={"best": 0.0, "tied_worse": 0.0, "middle": 1.0, "bad": 2.0},
        ),
    )
    _write_json(
        transfer_root / task / "transfer_score.json",
        _selection(
            task=task,
            selector="transfer_score",
            scores={"best": 0.9, "tied_worse": 0.8, "middle": 0.7, "bad": 0.1},
            direction="maximize",
        ),
    )

    result = run(
        Namespace(
            dev_truth_report=truth_path,
            selection_root=[spectra_root, transfer_root],
            selector=None,
            tasks=[task],
            objective_selector="agreement_reference",
            transfer_selector="transfer_score",
            expected_candidate_count=4,
            runtime_budget_seconds=350.0,
            output=None,
        )
    )

    report = result["selectors"]["agreement_reference"]["tasks"][0]
    assert report["unique_score_ratio"] == pytest.approx(0.75)
    assert report["top_score_tie_group_size"] == 2
    assert report["top_score_tie_best_normalized_regret"] == pytest.approx(0.0)
    assert report["top_score_tie_worst_normalized_regret"] > 0.0
    assert report["oracle_recall_at_5pct"] == 1.0
    assert report["predicted_shortlist_size_at_5pct"] == 2.0


def test_objective_v2_rejects_sealed_truth_paths(tmp_path: Path) -> None:
    sealed = tmp_path / ".sealed" / "open_dev_truth.json"
    _write_json(sealed, {"tasks": []})

    with pytest.raises(RuntimeError, match="forbidden path"):
        load_open_dev_truth(sealed)
