from __future__ import annotations

import numpy as np
from scipy.optimize import lsq_linear

from covariance_transport.transport import corrected_band_risk_recovery
from selector.risk_recovery import pair_design


def disagreement_matrix(observations: np.ndarray, model_count: int) -> np.ndarray:
    _, first, second = pair_design(model_count)
    matrix = np.zeros((1, model_count, model_count), dtype=np.float64)
    for value, left, right in zip(observations, first, second, strict=True):
        matrix[0, left, right] = value
        matrix[0, right, left] = value
    return matrix


def test_tukey_refit_materially_reduces_single_pair_outlier_error() -> None:
    model_count = 8
    true_risk = np.linspace(0.1, 0.8, model_count)
    design, first, second = pair_design(model_count)
    generator = np.random.default_rng(7)
    observations = true_risk[first] + true_risk[second]
    observations = observations + generator.normal(0.0, 0.01, size=observations.shape)
    observations[0] += 2.0

    plain = lsq_linear(design, observations, bounds=(0.0, np.inf)).x
    robust, diagnostics = corrected_band_risk_recovery(
        disagreement_matrix(observations, model_count),
        true_risk[:, None],
        np.zeros((1, model_count, model_count), dtype=np.float64),
        ridge=0.0,
        pair_weight_power=0.0,
    )

    plain_error = np.linalg.norm(plain - true_risk)
    robust_error = np.linalg.norm(robust[:, 0] - true_risk)
    # The frozen Tukey + endpoint-reliability refit reduces this deterministic
    # outlier error by about 47%. Require a material >=40% reduction while
    # avoiding a stale Huber-specific half-error threshold.
    assert robust_error < 0.6 * plain_error
    assert diagnostics[0]["min_pair_weight"] < 0.2


def test_exact_pair_system_is_preserved_when_residual_scale_is_zero() -> None:
    model_count = 6
    true_risk = np.linspace(0.1, 0.6, model_count)
    _, first, second = pair_design(model_count)
    observations = true_risk[first] + true_risk[second]

    recovered, diagnostics = corrected_band_risk_recovery(
        disagreement_matrix(observations, model_count),
        true_risk[:, None],
        np.zeros((1, model_count, model_count), dtype=np.float64),
        ridge=0.0,
        pair_weight_power=0.0,
    )

    np.testing.assert_allclose(recovered[:, 0], true_risk, atol=1e-10, rtol=0.0)
    assert diagnostics[0]["min_pair_weight"] == 1.0
    assert diagnostics[0]["max_pair_weight"] == 1.0


def test_robust_flag_disables_tail_clipping_and_refit() -> None:
    model_count = 8
    true_risk = np.linspace(0.1, 0.8, model_count)
    _, first, second = pair_design(model_count)
    observations = true_risk[first] + true_risk[second]
    observations[0] += 2.0
    matrix = disagreement_matrix(observations, model_count)
    covariance = np.zeros((1, model_count, model_count), dtype=np.float64)

    plain, _ = corrected_band_risk_recovery(
        matrix,
        true_risk[:, None],
        covariance,
        ridge=0.0,
        pair_weight_power=0.0,
        robust=False,
    )
    robust, _ = corrected_band_risk_recovery(
        matrix,
        true_risk[:, None],
        covariance,
        ridge=0.0,
        pair_weight_power=0.0,
        robust=True,
    )

    assert not np.allclose(plain, robust)
