"""Equivalence tests for the local augmented recovery workspace."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np

from covariance_transport.transport import (
    _augmented_recovery_design,
    _augmented_recovery_observations,
    corrected_band_risk_recovery,
)
from selector.risk_recovery import pair_design


def test_augmented_design_matches_legacy_assembly_exactly() -> None:
    model_count = 9
    design, _, _ = pair_design(model_count)
    weights = np.linspace(0.125, 2.25, design.shape[0], dtype=np.float64)
    scales = (0.01, 3.5)

    actual = _augmented_recovery_design(weights, model_count, scales)
    legacy_data = np.concatenate(
        [
            np.repeat(weights, 2),
            np.full(model_count, scales[0]),
            np.full(model_count, scales[1]),
        ]
    )

    np.testing.assert_array_equal(actual.data, legacy_data)
    np.testing.assert_array_equal(actual.indices[: design.indices.size], design.indices)
    np.testing.assert_array_equal(actual.indptr[: design.indptr.size], design.indptr)


def test_augmented_observations_match_legacy_assembly_exactly() -> None:
    model_count = 7
    pair_count = model_count * (model_count - 1) // 2
    observations = np.linspace(0.01, 0.9, pair_count, dtype=np.float64)
    weights = np.linspace(0.2, 1.4, pair_count, dtype=np.float64)
    prior = np.linspace(0.05, 0.8, model_count, dtype=np.float64)
    prior_strength = 2.75

    actual = _augmented_recovery_observations(
        observations,
        weights,
        model_count,
        ridge=1e-6,
        prior_strength=prior_strength,
        prior_risk=prior,
    )
    expected = np.concatenate(
        [
            observations * weights,
            np.zeros(model_count),
            np.sqrt(prior_strength) * prior,
        ]
    )
    np.testing.assert_array_equal(actual, expected)


def _recover(seed: int) -> tuple[np.ndarray, list[dict[str, float]]]:
    rng = np.random.default_rng(seed)
    model_count = 8
    band_count = 3
    raw = rng.uniform(0.0, 0.2, size=(band_count, model_count, model_count))
    disagreements = 0.5 * (raw + raw.transpose(0, 2, 1))
    for band in range(band_count):
        np.fill_diagonal(disagreements[band], 0.0)
    prior = rng.uniform(0.02, 0.4, size=(model_count, band_count))
    covariance = rng.normal(0.0, 0.003, size=(band_count, model_count, model_count))
    covariance = 0.5 * (covariance + covariance.transpose(0, 2, 1))
    return corrected_band_risk_recovery(
        disagreements,
        prior,
        covariance,
        ridge=1e-6,
        pair_weight_power=1.0,
        prior_strength=float(model_count - 2),
    )


def test_workspace_reuse_is_deterministic_and_thread_local() -> None:
    expected = [_recover(seed) for seed in range(4)]
    with ThreadPoolExecutor(max_workers=4) as executor:
        actual = list(executor.map(_recover, range(4)))

    for (expected_risks, expected_diag), (actual_risks, actual_diag) in zip(
        expected, actual
    ):
        np.testing.assert_array_equal(actual_risks, expected_risks)
        assert actual_diag == expected_diag
