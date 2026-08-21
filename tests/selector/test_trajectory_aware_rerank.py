from __future__ import annotations

from selector.trajectory_aware_rerank import build_trajectory_aware_rerank


def _candidate(trajectory: str, epoch: int) -> str:
    method, config, seed = trajectory.split("/")
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


def test_k1_exactly_matches_candidate_level_rerank() -> None:
    candidates = [
        _candidate(trajectory, epoch)
        for trajectory in ("M1/c1/1", "M2/c2/2")
        for epoch in (10, 20, 30)
    ]
    owner = {candidate: float(index) for index, candidate in enumerate(candidates)}
    transfer = {candidate: 1.0 for candidate in candidates}
    transfer[_candidate("M2/c2/2", 10)] = 0.0

    result = build_trajectory_aware_rerank(
        task="A_to_B",
        shortlist_owner=_selection("agreement", owner),
        reranker=_selection("transfer", transfer),
        selector_name="k1",
        shortlist_fraction=1.0,
        top_k=1,
    )

    assert result["selected_candidate_id"] == _candidate("M2/c2/2", 10)
    assert result["fusion_config"]["top_k"] == 1


def test_k2_prefers_repeated_trajectory_evidence() -> None:
    candidates = [
        _candidate(trajectory, epoch)
        for trajectory in ("M1/c1/1", "M2/c2/2", "M3/c3/3")
        for epoch in (10, 20)
    ]
    owner = {candidate: 0.0 for candidate in candidates}
    transfer = {candidate: 1.0 for candidate in candidates}
    transfer[_candidate("M1/c1/1", 10)] = 0.00
    transfer[_candidate("M1/c1/1", 20)] = 1.00
    transfer[_candidate("M2/c2/2", 10)] = 0.10
    transfer[_candidate("M2/c2/2", 20)] = 0.20

    result = build_trajectory_aware_rerank(
        task="A_to_B",
        shortlist_owner=_selection("agreement", owner),
        reranker=_selection("transfer", transfer),
        selector_name="k2",
        shortlist_fraction=1.0,
        top_k=2,
    )

    assert result["selected_candidate_id"] == _candidate("M2/c2/2", 10)
    assert result["fusion_config"]["selected_trajectory"] == "M2__c2__seed-2"


def test_missing_slots_receive_worst_rank_padding() -> None:
    candidates = [
        _candidate("M1/c1/1", 10),
        _candidate("M2/c2/2", 10),
        _candidate("M2/c2/2", 20),
        _candidate("M3/c3/3", 10),
        _candidate("M3/c3/3", 20),
        _candidate("M3/c3/3", 30),
    ]
    owner = {candidate: 0.0 for candidate in candidates}
    transfer = {
        _candidate("M1/c1/1", 10): 0.0,
        _candidate("M2/c2/2", 10): 0.1,
        _candidate("M2/c2/2", 20): 0.2,
        _candidate("M3/c3/3", 10): 0.7,
        _candidate("M3/c3/3", 20): 0.8,
        _candidate("M3/c3/3", 30): 0.9,
    }

    result = build_trajectory_aware_rerank(
        task="A_to_B",
        shortlist_owner=_selection("agreement", owner),
        reranker=_selection("transfer", transfer),
        selector_name="k2",
        shortlist_fraction=1.0,
        top_k=2,
    )

    assert result["selected_candidate_id"] == _candidate("M2/c2/2", 10)
