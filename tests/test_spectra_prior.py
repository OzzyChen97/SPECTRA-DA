from __future__ import annotations

import numpy as np

from covariance_transport import corrected_band_risk_recovery
from selector.spectra_cal import (
    calibrated_selector_name,
    covariance_shrinkage_gamma,
    curvature_prior_strength,
    pair_sum_consistency_gamma,
    recover_with_transport,
    shrink_transported_covariances,
)


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


def test_covariance_shrinkage_modes_are_explicit_and_label_free() -> None:
    covariances = np.arange(18, dtype=np.float64).reshape(2, 3, 3)

    assert covariance_shrinkage_gamma(99.0, mode="none") == 1.0
    assert covariance_shrinkage_gamma(1.5, mode="support_gate") == 1.0
    assert covariance_shrinkage_gamma(2.5, mode="support_gate") == 0.0
    assert covariance_shrinkage_gamma(99.0, mode="fixed", fixed_gamma=0.25) == 0.25

    shrunk, diagnostics = shrink_transported_covariances(
        covariances,
        descriptor_rmse=2.5,
        mode="support_gate",
    )
    np.testing.assert_allclose(shrunk, np.zeros_like(covariances))
    assert diagnostics["mode"] == "support_gate"
    assert diagnostics["gamma"] == 0.0


def test_calibrated_selector_name_can_be_overridden_for_gamma_sweeps() -> None:
    assert calibrated_selector_name(spectral_mode="banded", robust=False) == "spectra_cal"
    assert (
        calibrated_selector_name(spectral_mode="global", robust=True)
        == "spectra_global_robust"
    )
    assert (
        calibrated_selector_name(
            spectral_mode="banded",
            robust=False,
            output_selector="spectra_cov_gamma050",
        )
        == "spectra_cov_gamma050"
    )


def test_support_gate_removes_unsupported_covariance_from_recovery() -> None:
    true_risk = np.asarray([[0.10], [0.18], [0.28], [0.42]], dtype=np.float64)
    covariance = 0.08 * np.sqrt(true_risk @ true_risk.T)
    np.fill_diagonal(covariance, true_risk[:, 0])
    disagreements = true_risk[:, 0, None] + true_risk[None, :, 0] - 2.0 * covariance
    np.fill_diagonal(disagreements, 0.0)
    disagreements = disagreements[None, :, :]

    common = {
        "shift_deltas": np.zeros((2, 2), dtype=np.float64),
        "target_delta": np.asarray([10.0, 0.0], dtype=np.float64),
        "shift_band_risks": np.repeat(true_risk[None, :, :], 2, axis=0),
        "disagreements": disagreements,
        "transport_regularization": 0.1,
        "descriptor_floor": 0.05,
        "risk_ridge": 0.0,
        "pair_weight_power": 0.0,
    }
    gated, _, gated_diagnostics = recover_with_transport(
        **common,
        shift_band_covariances=np.repeat(covariance[None, None, :, :], 2, axis=0),
        covariance_shrinkage_mode="support_gate",
    )
    zero_covariance, _, _ = recover_with_transport(
        **common,
        shift_band_covariances=np.zeros((2, 1, 4, 4), dtype=np.float64),
        covariance_shrinkage_mode="none",
    )

    np.testing.assert_allclose(gated, zero_covariance, atol=1.0e-12, rtol=0.0)
    assert gated_diagnostics["covariance_shrinkage"]["gamma"] == 0.0


def test_pair_consistency_shrinkage_prefers_self_consistent_covariance() -> None:
    true_risk = np.asarray([[0.07], [0.13], [0.31], [0.46]], dtype=np.float64)
    covariance = np.asarray(
        [
            [0.07, 0.009, 0.041, 0.006],
            [0.009, 0.13, 0.014, 0.058],
            [0.041, 0.014, 0.31, 0.027],
            [0.006, 0.058, 0.027, 0.46],
        ],
        dtype=np.float64,
    )
    disagreements = true_risk[:, 0, None] + true_risk[None, :, 0] - 2.0 * covariance
    np.fill_diagonal(disagreements, 0.0)

    gamma, reports = pair_sum_consistency_gamma(
        disagreements[None, :, :],
        covariance[None, :, :],
    )

    assert gamma == 1.0
    assert reports[-1]["normalized_residual"] < 1.0e-24


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
