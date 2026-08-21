from __future__ import annotations

import numpy as np

from selector.crossfit_agreement import crossfit_agreement_scores


def test_trajectory_crossfit_removes_self_reinforcing_vote_block() -> None:
    # Group a has three identical checkpoints and would dominate an ordinary
    # majority vote. Cross-fitting it out exposes group b's opposite reference.
    predictions = np.asarray(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
        ],
        dtype=np.int64,
    )
    scores = crossfit_agreement_scores(predictions, ["a", "a", "a", "b", "b"])

    np.testing.assert_allclose(scores[:3], 0.0)
    np.testing.assert_allclose(scores[3:], 0.0)


def test_crossfit_scores_candidates_against_other_groups() -> None:
    predictions = np.asarray(
        [
            [0, 1, 0, 1],
            [0, 1, 1, 1],
            [0, 1, 0, 1],
            [1, 1, 0, 0],
        ],
        dtype=np.int64,
    )
    scores = crossfit_agreement_scores(predictions, ["a", "a", "b", "c"])

    assert np.isfinite(scores).all()
    assert np.all((scores >= 0.0) & (scores <= 1.0))
    assert scores[0] != scores[1]
