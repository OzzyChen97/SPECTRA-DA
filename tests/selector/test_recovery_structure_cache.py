"""Exact-equivalence tests for cached covariance-recovery structures."""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.optimize import lsq_linear

from covariance_transport.transport import (
    _augmented_recovery_design,
    corrected_band_risk_recovery,
)
from selector.risk_recovery import pair_design


def _baseline_augmented_design(weights: np.ndarray, model_count: int) -> sparse.csr_matrix:
    design, _, _ = pair_design(model_count)
    weighted = design.multiply(weights[:, None])
    return sparse.vstack(
        [
            weighted,
            np.sqrt(1.0e-6) * sparse.eye(model_count, format="csr"),
            np.sqrt(model_count - 2) * sparse.eye(model_count, format="csr"),
        ],
        format="csr",
    )


def test_cached_design_is_exactly_equivalent_to_baseline_structure_and_scores() -> None:
    model_count = 9
    design, first, _ = pair_design(model_count)
    rng = np.random.default_rng(20260820)
    square_root_weights = rng.uniform(0.1, 2.0, design.shape[0])
    scales = (np.sqrt(1.0e-6), np.sqrt(model_count - 2))

    baseline = _baseline_augmented_design(square_root_weights, model_count)
    cached = _augmented_recovery_design(square_root_weights, model_count, scales)

    np.testing.assert_array_equal(cached.toarray(), baseline.toarray())
    observations = rng.uniform(0.0, 1.0, cached.shape[0])
    baseline_risk = lsq_linear(baseline, observations, bounds=(0.0, np.inf)).x
    cached_risk = lsq_linear(cached, observations, bounds=(0.0, np.inf)).x
    np.testing.assert_array_equal(cached_risk, baseline_risk)
    assert int(np.argmin(cached_risk)) == int(np.argmin(baseline_risk))


def test_cached_recovery_is_repeatable_with_identical_selection() -> None:
    model_count, band_count = 7, 3
    risks = np.linspace(0.05, 0.65, model_count * band_count).reshape(model_count, band_count)
    covariance = np.zeros((band_count, model_count, model_count))
    disagreement = np.empty_like(covariance)
    for band in range(band_count):
        disagreement[band] = risks[:, band, None] + risks[None, :, band]
        np.fill_diagonal(disagreement[band], 0.0)

    first, _ = corrected_band_risk_recovery(
        disagreement, risks, covariance,
        ridge=1.0e-6, pair_weight_power=1.0, prior_strength=model_count - 2,
    )
    second, _ = corrected_band_risk_recovery(
        disagreement, risks, covariance,
        ridge=1.0e-6, pair_weight_power=1.0, prior_strength=model_count - 2,
    )
    np.testing.assert_array_equal(second, first)
    assert int(np.argmin(second.sum(axis=1))) == int(np.argmin(first.sum(axis=1)))
