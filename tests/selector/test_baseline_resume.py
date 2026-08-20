from __future__ import annotations

import json

from selector.baselines import build_selection_result
from selector.run_baseline_suite import is_valid_cached_selection


def _records() -> list[dict]:
    return [
        {"metadata": {"candidate_id": f"task__Method__config__seed-1__epoch-{index:04d}"}}
        for index in range(1, 4)
    ]


def test_valid_cached_selection_is_reused(tmp_path) -> None:
    records = _records()
    scores = {
        record["metadata"]["candidate_id"]: float(index)
        for index, record in enumerate(records)
    }
    result = build_selection_result(
        records,
        "task",
        "entropy",
        scores,
        "minimize",
        bank_hash="frozen-bank",
    )
    path = tmp_path / "entropy.json"
    path.write_text(json.dumps(result), encoding="utf-8")

    assert is_valid_cached_selection(
        path,
        task="task",
        selector="entropy",
        direction="minimize",
        candidate_ids=set(scores),
        bank_hash="frozen-bank",
    )


def test_cached_selection_must_match_bank_coverage_and_optimum(tmp_path) -> None:
    records = _records()
    scores = {
        record["metadata"]["candidate_id"]: float(index)
        for index, record in enumerate(records)
    }
    result = build_selection_result(
        records,
        "task",
        "entropy",
        scores,
        "minimize",
        bank_hash="frozen-bank",
    )
    path = tmp_path / "entropy.json"

    wrong_bank = dict(result, candidate_bank_sha256="stale-bank")
    path.write_text(json.dumps(wrong_bank), encoding="utf-8")
    assert not is_valid_cached_selection(
        path,
        task="task",
        selector="entropy",
        direction="minimize",
        candidate_ids=set(scores),
        bank_hash="frozen-bank",
    )

    incomplete = dict(result, candidate_scores=dict(list(scores.items())[:-1]))
    path.write_text(json.dumps(incomplete), encoding="utf-8")
    assert not is_valid_cached_selection(
        path,
        task="task",
        selector="entropy",
        direction="minimize",
        candidate_ids=set(scores),
        bank_hash="frozen-bank",
    )

    non_optimal = dict(result, selected_candidate_id=max(scores))
    path.write_text(json.dumps(non_optimal), encoding="utf-8")
    assert not is_valid_cached_selection(
        path,
        task="task",
        selector="entropy",
        direction="minimize",
        candidate_ids=set(scores),
        bank_hash="frozen-bank",
    )
