from __future__ import annotations

from selector.trajectory_shortlist_selection import build_trajectory_shortlist


def _candidate(trajectory: str, epoch: int) -> str:
    method, config, seed = trajectory.split("/")
    return f"A_to_B__{method}__{config}__seed-{seed}__epoch-{epoch:04d}"


def _selection(selector: str, scores: dict[str, float]) -> dict:
    selected = min(scores, key=scores.get)
    return {
        "schema_version": 1,
        "task": "A_to_B",
        "selector": selector,
        "candidate_bank_sha256": "bank",
        "candidate_count": len(scores),
        "candidate_scores": scores,
        "score_direction": "minimize",
        "selected_candidate_id": selected,
        "label_access_count": 0,
        "protocol_violation_count": 0,
    }


def test_trajectory_shortlist_expands_selected_trajectory_before_rerank() -> None:
    trajectories = ["M1/c1/1", "M2/c2/2", "M3/c3/3"]
    candidates = [
        _candidate(trajectory, epoch)
        for trajectory in trajectories
        for epoch in (10, 20, 30)
    ]
    owner_scores = {candidate: 0.9 for candidate in candidates}
    for epoch, score in zip((10, 20, 30), (0.0, 0.1, 0.2)):
        owner_scores[_candidate("M2/c2/2", epoch)] = score
    rerank_scores = {candidate: 0.8 for candidate in candidates}
    rerank_scores[_candidate("M2/c2/2", 30)] = 0.0
    rerank_scores[_candidate("M1/c1/1", 10)] = -1.0

    result = build_trajectory_shortlist(
        task="A_to_B",
        shortlist_owners=[_selection("agreement", owner_scores)],
        reranker=_selection("transfer", rerank_scores),
        selector_name="trajectory_selector",
        trajectory_fraction=1.0 / 3.0,
        best_q=3,
    )

    assert result["selected_candidate_id"] == _candidate("M2/c2/2", 30)
    assert result["candidate_scores"][_candidate("M1/c1/1", 10)] > 1.0e11
    assert result["fusion_config"]["shortlist_trajectory_count"] == 1
    assert result["fusion_config"]["shortlist_candidate_count"] == 3


def test_multiple_owners_use_trajectory_midrank_consensus() -> None:
    trajectories = ["M1/c1/1", "M2/c2/2", "M3/c3/3"]
    candidates = [
        _candidate(trajectory, epoch)
        for trajectory in trajectories
        for epoch in (10, 20, 30)
    ]
    first = {candidate: 1.0 for candidate in candidates}
    second = {candidate: 1.0 for candidate in candidates}
    for epoch in (10, 20, 30):
        first[_candidate("M1/c1/1", epoch)] = 0.0
        second[_candidate("M2/c2/2", epoch)] = 0.0
    rerank = {candidate: 1.0 for candidate in candidates}
    rerank[_candidate("M1/c1/1", 20)] = 0.0

    result = build_trajectory_shortlist(
        task="A_to_B",
        shortlist_owners=[_selection("first", first), _selection("second", second)],
        reranker=_selection("transfer", rerank),
        selector_name="consensus",
        trajectory_fraction=2.0 / 3.0,
        best_q=3,
    )

    assert result["selected_candidate_id"] == _candidate("M1/c1/1", 20)
    assert result["fusion_config"]["shortlist_owners"] == ["first", "second"]
