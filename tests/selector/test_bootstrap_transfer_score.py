from __future__ import annotations

import numpy as np

from selector.bootstrap_transfer_score import (
    aggregate_transfer_stability,
    normalized_information_maximization,
)


def _candidate(index: int) -> str:
    return f"A_to_B__M{index}__c{index}__seed-{index}__epoch-0010"


def test_information_maximization_is_finite() -> None:
    probabilities = np.asarray([[0.9, 0.1], [0.2, 0.8], [0.5, 0.5]])
    value = normalized_information_maximization(probabilities)
    assert np.isfinite(value)


def test_transfer_stability_reports_fixed_budget_and_frequencies() -> None:
    candidate_ids = [_candidate(index) for index in range(4)]
    frozen = np.asarray([4.0, 3.0, 2.0, 1.0])
    bootstrap = np.asarray(
        [
            [4.0, 3.0, 4.0],
            [3.0, 4.0, 3.0],
            [2.0, 2.0, 2.0],
            [1.0, 1.0, 1.0],
        ]
    )

    report = aggregate_transfer_stability(
        candidate_ids, frozen, bootstrap, budget=2
    )

    assert report["mean_shortlist_jaccard"] == 1.0
    assert report["mean_trajectory_shortlist_jaccard"] == 1.0
    assert sum(report["selected_candidate_frequency"].values()) == 3
    assert sum(report["selected_trajectory_frequency"].values()) == 3
    assert report["max_selected_trajectory_frequency"] == 2.0 / 3.0
    assert all(
        0.0 <= value <= 1.0
        for value in report["candidate_inclusion_frequency"].values()
    )
