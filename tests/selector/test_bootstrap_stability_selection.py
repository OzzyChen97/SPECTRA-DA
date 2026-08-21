from __future__ import annotations

import numpy as np

from selector.bootstrap_stability_selection import bootstrap_agreement_diagnostics


def _candidate(index: int) -> str:
    return f"A_to_B__M{index // 2}__c{index // 2}__seed-{index // 2}__epoch-{index:04d}"


def test_bootstrap_agreement_is_deterministic_and_fixed_budget() -> None:
    predictions = np.asarray(
        [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1],
            [0, 0, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
        ],
        dtype=np.int64,
    )
    candidate_ids = [_candidate(index) for index in range(4)]
    reranker_ranks = {candidate: float(index) for index, candidate in enumerate(candidate_ids)}

    first = bootstrap_agreement_diagnostics(
        predictions,
        candidate_ids,
        reranker_ranks,
        budget=2,
        bootstrap_count=8,
        node_fraction=0.75,
        seed=7,
    )
    second = bootstrap_agreement_diagnostics(
        predictions,
        candidate_ids,
        reranker_ranks,
        budget=2,
        bootstrap_count=8,
        node_fraction=0.75,
        seed=7,
    )

    assert first["stable_shortlist_indices"] == second["stable_shortlist_indices"]
    assert len(first["stable_shortlist_indices"]) == 2
    assert np.array_equal(first["inclusion_frequency"], second["inclusion_frequency"])
    assert 0.0 <= first["mean_shortlist_jaccard"] <= 1.0
    assert 0.0 <= first["mean_trajectory_shortlist_jaccard"] <= 1.0
    assert sum(first["selected_candidate_frequency"].values()) == 8


def test_bootstrap_agreement_rejects_misaligned_reranker() -> None:
    predictions = np.zeros((2, 4), dtype=np.int64)
    candidate_ids = [_candidate(0), _candidate(1)]

    try:
        bootstrap_agreement_diagnostics(
            predictions,
            candidate_ids,
            {candidate_ids[0]: 0.0},
            budget=1,
            bootstrap_count=2,
            node_fraction=0.5,
            seed=1,
        )
    except ValueError as exc:
        assert "reranker ranks" in str(exc)
    else:
        raise AssertionError("misaligned reranker ranks were accepted")
