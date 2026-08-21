from __future__ import annotations

from selector.coverage_floor_selection import (
    build_coverage_floor_selection,
    build_fixed_budget_shortlist,
    method_key,
)
from selector.reliable_selection import percentile_ranks
from selector.trajectory_shortlist_selection import trajectory_key


def _candidate(method: str, config: str, seed: int, epoch: int) -> str:
    return f"A_to_B__{method}__{config}__seed-{seed}__epoch-{epoch:04d}"


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


def _bank() -> list[str]:
    return [
        _candidate(method, config, seed, epoch)
        for method, config, seed in (
            ("M1", "c1", 1),
            ("M1", "c2", 2),
            ("M2", "c3", 3),
            ("M2", "c4", 4),
        )
        for epoch in (10, 20, 30)
    ]


def test_trajectory_floor_preserves_every_group_then_globally_fills() -> None:
    candidates = _bank()
    owner_scores = {candidate: float(index) for index, candidate in enumerate(candidates)}
    ranks = percentile_ranks(owner_scores, "minimize")

    shortlist, guaranteed = build_fixed_budget_shortlist(
        set(candidates), ranks, budget=7, mode="trajectory", group_quota=1
    )

    assert len(shortlist) == 7
    assert len(guaranteed) == 4
    assert all(len(values) == 1 for values in guaranteed.values())
    assert {trajectory_key(candidate) for candidate in shortlist} == set(guaranteed)


def test_method_mode_uses_equal_fixed_quota() -> None:
    candidates = _bank()
    ranks = {candidate: float(index) for index, candidate in enumerate(candidates)}

    shortlist, guaranteed = build_fixed_budget_shortlist(
        set(candidates), ranks, budget=6, mode="method", group_quota=3
    )

    assert len(shortlist) == 6
    assert {method_key(candidate) for candidate in shortlist} == {"M1", "M2"}
    assert all(len(values) == 3 for values in guaranteed.values())


def test_reranker_can_only_select_from_fixed_budget_shortlist() -> None:
    candidates = _bank()
    owner_scores = {candidate: float(index) for index, candidate in enumerate(candidates)}
    transfer_scores = {candidate: 1.0 for candidate in candidates}
    outside = candidates[-1]
    transfer_scores[outside] = -10.0
    inside = candidates[0]
    transfer_scores[inside] = -5.0

    result = build_coverage_floor_selection(
        task="A_to_B",
        shortlist_owner=_selection("agreement", owner_scores),
        reranker=_selection("transfer", transfer_scores),
        selector_name="floor",
        budget=4,
        mode="trajectory",
        group_quota=1,
    )

    assert result["selected_candidate_id"] == inside
    assert result["candidate_scores"][outside] > 1.0e11
    assert result["fusion_config"]["shortlist_size"] == 4
    assert result["label_access_count"] == 0
    assert result["protocol_violation_count"] == 0
