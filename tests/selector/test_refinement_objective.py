from __future__ import annotations

import pytest

from selector.refinement_objective import covariance_rank_gate_allows


def _metadata(*, family: str | None, source_nodes: int, candidates: int = 5):
    return {
        "candidate_count": candidates,
        "selected_shifts": [
            {
                "calibration_family": family,
                "node_count": source_nodes,
            }
            for _ in range(3)
        ],
    }


def test_rank_gate_rejects_rank_deficient_feature_mask_grid() -> None:
    assert not covariance_rank_gate_allows(
        _metadata(family="feature_mask_grid", source_nodes=4, candidates=5)
    )


def test_rank_gate_accepts_identifiable_feature_mask_grid() -> None:
    assert covariance_rank_gate_allows(
        _metadata(family="feature_mask_grid", source_nodes=5, candidates=5)
    )


def test_rank_gate_does_not_suppress_other_calibration_families() -> None:
    assert covariance_rank_gate_allows(
        _metadata(family=None, source_nodes=2, candidates=5)
    )


def test_rank_gate_rejects_inconsistent_feature_mask_node_counts() -> None:
    metadata = _metadata(family="feature_mask_grid", source_nodes=5)
    metadata["selected_shifts"][1]["node_count"] = 4
    with pytest.raises(ValueError, match="inconsistent source node counts"):
        covariance_rank_gate_allows(metadata)
