from __future__ import annotations

from selector.auxiliary_shortlist_selection import (
    build_auxiliary_shortlist_selection,
    fixed_auxiliary_shortlist,
)


def _selection(selector: str, scores: dict[str, float]) -> dict:
    return {
        "schema_version": 1,
        "task": "A_to_B",
        "selector": selector,
        "candidate_bank_sha256": "bank",
        "candidate_count": len(scores),
        "candidate_scores": scores,
        "score_direction": "minimize",
        "selected_candidate_id": min(scores, key=scores.get),
        "label_access_count": 0,
        "protocol_violation_count": 0,
    }


def test_midrank_uses_fixed_budget() -> None:
    candidates = {f"c{index}" for index in range(10)}
    first = {candidate: float(index) / 9.0 for index, candidate in enumerate(sorted(candidates))}
    second = {candidate: 1.0 - first[candidate] for candidate in candidates}

    shortlist, info = fixed_auxiliary_shortlist(
        candidates, [first, second], mode="midrank", budget=4, owner_fraction=0.1
    )

    assert len(shortlist) == 4
    assert info["budget_adjustment"] == "fixed consensus top-budget"


def test_union_is_filled_or_capped_to_exact_budget() -> None:
    candidates = {f"c{index}" for index in range(20)}
    first = {candidate: float(index) for index, candidate in enumerate(sorted(candidates))}
    second = dict(first)

    shortlist, info = fixed_auxiliary_shortlist(
        candidates, [first, second], mode="union", budget=8, owner_fraction=0.1
    )

    assert len(shortlist) == 8
    assert info["raw_union_size"] == 2
    assert info["budget_adjustment"] == "midrank_fill"


def test_auxiliary_shortlist_reranks_only_inside_budget() -> None:
    candidates = [f"c{index}" for index in range(8)]
    owner = {candidate: float(index) for index, candidate in enumerate(candidates)}
    transfer = {candidate: 1.0 for candidate in candidates}
    transfer["c7"] = -10.0
    transfer["c0"] = -5.0

    result = build_auxiliary_shortlist_selection(
        task="A_to_B",
        owners=[_selection("gamma025", owner)],
        reranker=_selection("transfer", transfer),
        selector_name="aux",
        mode="single",
        budget=4,
        owner_fraction=0.1,
    )

    assert result["selected_candidate_id"] == "c0"
    assert result["candidate_scores"]["c7"] > 1.0e11
    assert result["fusion_config"]["shortlist_size"] == 4
    assert result["label_access_count"] == 0
