from __future__ import annotations

import pytest

from selector.shortlist_error_decomposition import decompose_shortlist


def _candidate(method: str, config: str, seed: int, epoch: int) -> str:
    return f"A_to_B__{method}__{config}__seed-{seed}__epoch-{epoch:04d}"


def test_shortlist_decomposition_is_exact() -> None:
    global_oracle = _candidate("M1", "c1", 1, 10)
    represented_trajectory_best = _candidate("M2", "c2", 2, 10)
    shortlisted_best = _candidate("M2", "c2", 2, 20)
    selected = _candidate("M3", "c3", 3, 10)
    risks = {
        global_oracle: 0.10,
        represented_trajectory_best: 0.20,
        shortlisted_best: 0.25,
        selected: 0.40,
    }
    exclusion = 1.0e12
    selection = {
        "task": "A_to_B",
        "selector": "shortlist",
        "candidate_scores": {
            global_oracle: exclusion + 0.1,
            represented_trajectory_best: exclusion + 0.2,
            shortlisted_best: 0.1,
            selected: 0.0,
        },
        "selected_candidate_id": selected,
        "fusion_config": {
            "shortlist_exclusion_score": exclusion,
            "shortlist_size": 2,
        },
    }

    report = decompose_shortlist(
        selection,
        {"task": "A_to_B", "risks": risks},
    )

    assert report["total_gap"] == pytest.approx(0.30)
    assert report["trajectory_coverage_gap"] == pytest.approx(0.10)
    assert report["checkpoint_coverage_gap"] == pytest.approx(0.05)
    assert report["reranking_gap"] == pytest.approx(0.15)
    assert (
        report["trajectory_coverage_gap"]
        + report["checkpoint_coverage_gap"]
        + report["reranking_gap"]
    ) == pytest.approx(report["total_gap"])
