from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from selector.selection_error_decomposition import decompose_selection, run


def _candidate(method: str, config: str, seed: int, epoch: int) -> str:
    return f"A_to_B__{method}__{config}__seed-{seed}__epoch-{epoch:04d}"


def _selection(scores: dict[str, float], selected: str) -> dict:
    return {
        "schema_version": 1,
        "task": "A_to_B",
        "selector": "test_selector",
        "candidate_bank_sha256": "bank",
        "candidate_count": len(scores),
        "candidate_scores": scores,
        "score_direction": "minimize",
        "selected_candidate_id": selected,
        "label_access_count": 0,
        "protocol_violation_count": 0,
    }


def test_decomposition_is_exact_at_trajectory_and_method_levels() -> None:
    oracle = _candidate("M1", "c1", 1, 10)
    same_method = _candidate("M1", "c2", 1, 10)
    trajectory_best = _candidate("M2", "c1", 1, 10)
    selected = _candidate("M2", "c1", 1, 20)
    risks = {
        oracle: 0.10,
        same_method: 0.15,
        trajectory_best: 0.20,
        selected: 0.35,
    }
    report = decompose_selection(
        _selection({candidate: float(index) for index, candidate in enumerate(risks)}, selected),
        {"task": "A_to_B", "risks": risks},
    )

    assert report["total_gap"] == pytest.approx(0.25)
    assert report["trajectory_gap"] == pytest.approx(0.10)
    assert report["checkpoint_gap"] == pytest.approx(0.15)
    assert report["method_gap"] == pytest.approx(0.10)
    assert report["within_method_gap"] == pytest.approx(0.15)
    assert report["trajectory_gap"] + report["checkpoint_gap"] == pytest.approx(
        report["total_gap"]
    )


def test_run_rejects_sealed_truth_path(tmp_path: Path) -> None:
    sealed = tmp_path / "sealed_eval" / "truth.json"
    sealed.parent.mkdir()
    sealed.write_text(json.dumps({"tasks": []}), encoding="utf-8")

    with pytest.raises(RuntimeError, match="forbidden path"):
        run(
            argparse.Namespace(
                dev_truth_report=sealed,
                selection_root=[tmp_path],
                selector=["test_selector"],
                tasks=["A_to_B"],
                output=None,
            )
        )
