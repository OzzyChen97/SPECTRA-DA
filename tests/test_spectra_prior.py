from __future__ import annotations

import numpy as np

from covariance_transport import corrected_band_risk_recovery
from selector.spectra_cal import curvature_prior_strength


def _disagreement_from_risks(risks: np.ndarray) -> np.ndarray:
    matrix = risks[:, None] + risks[None, :]
    np.fill_diagonal(matrix, 0.0)
    return matrix[None, :, :]


def test_curvature_prior_uses_pair_design_scale_inside_support() -> None:
    disagreements = np.zeros((3, 7, 7), dtype=np.float64)
    assert curvature_prior_strength(disagreements, descriptor_rmse=2.0) == 5.0


def test_curvature_prior_turns_off_outside_two_sigma_support() -> None:
    disagreements = np.zeros((3, 7, 7), dtype=np.float64)
    assert curvature_prior_strength(disagreements, descriptor_rmse=2.0001) == 0.0


def test_prior_centering_reduces_recovery_error_under_pair_noise() -> None:
    true_risk = np.asarray([0.08, 0.16, 0.27, 0.39], dtype=np.float64)
    disagreements = _disagreement_from_risks(true_risk)
    noise = np.asarray(
        [
            [0.0, 0.08, -0.04, 0.05],
            [0.08, 0.0, 0.06, -0.03],
            [-0.04, 0.06, 0.0, 0.07],
            [0.05, -0.03, 0.07, 0.0],
        ],
        dtype=np.float64,
    )
    disagreements[0] = np.maximum(disagreements[0] + noise, 0.0)
    transported_risks = true_risk[:, None]
    transported_covariances = np.zeros((1, 4, 4), dtype=np.float64)

    base, _ = corrected_band_risk_recovery(
        disagreements,
        transported_risks,
        transported_covariances,
        ridge=0.0,
        pair_weight_power=0.0,
        prior_strength=0.0,
    )
    centered, _ = corrected_band_risk_recovery(
        disagreements,
        transported_risks,
        transported_covariances,
        ridge=0.0,
        pair_weight_power=0.0,
        prior_strength=2.0,
    )

    base_error = np.linalg.norm(base[:, 0] - true_risk)
    centered_error = np.linalg.norm(centered[:, 0] - true_risk)
    assert centered_error < base_error
