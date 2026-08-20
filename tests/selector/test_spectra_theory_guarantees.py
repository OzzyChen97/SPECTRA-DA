from __future__ import annotations

import numpy as np
import pytest
import torch

from covariance_transport import (
    corrected_band_risk_recovery,
    match_shift_convex_combination,
    transport_statistics,
)
from selector.risk_recovery import pair_design
from selector.spectra_cal import bootstrap_uncertainty
from spectral_filters import apply_tight_frame, frame_approximation_diagnostics


def test_approximate_tight_frame_controls_graph_signal_energy() -> None:
    edge_index = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5, 6],
            [1, 2, 3, 4, 5, 6, 0],
        ],
        dtype=torch.long,
    )
    generator = torch.Generator().manual_seed(20260819)
    signals = torch.randn(7, 5, generator=generator, dtype=torch.float64)

    filtered, coefficients = apply_tight_frame(
        edge_index,
        signals,
        num_bands=3,
        sigma=0.55,
        order=8,
    )
    diagnostics = frame_approximation_diagnostics(
        coefficients,
        num_bands=3,
        sigma=0.55,
    )

    input_energy = torch.sum(signals.square()).item()
    frame_energy = torch.sum(filtered.square()).item()
    relative_error = abs(frame_energy - input_energy) / input_energy
    assert relative_error <= diagnostics["max_frame_error"] + 1.0e-12


def test_exact_covariance_correction_recovers_band_risks() -> None:
    model_count, band_count = 7, 3
    risks = np.stack(
        [
            np.linspace(0.08, 0.38, model_count),
            np.linspace(0.05, 0.29, model_count),
            np.linspace(0.11, 0.47, model_count),
        ],
        axis=1,
    )
    covariances = np.zeros((band_count, model_count, model_count), dtype=np.float64)
    disagreements = np.zeros_like(covariances)
    for band in range(band_count):
        covariance = 0.15 * np.sqrt(
            risks[:, band, None] * risks[None, :, band]
        )
        np.fill_diagonal(covariance, risks[:, band])
        covariances[band] = covariance
        disagreements[band] = (
            risks[:, band, None]
            + risks[None, :, band]
            - 2.0 * covariance
        )
        np.fill_diagonal(disagreements[band], 0.0)

    recovered, diagnostics = corrected_band_risk_recovery(
        disagreements,
        risks,
        covariances,
        ridge=0.0,
        pair_weight_power=0.0,
        prior_strength=0.0,
    )

    np.testing.assert_allclose(recovered, risks, atol=1.0e-10, rtol=0.0)
    assert max(report["pair_rmse"] for report in diagnostics) < 1.0e-10


@pytest.mark.parametrize("model_count", [3, 5, 9, 17])
def test_covariance_misspecification_obeys_recovery_bound(model_count: int) -> None:
    design, _, _ = pair_design(model_count)
    dense_design = design.toarray()
    risks = np.linspace(0.1, 0.8, model_count)
    generator = np.random.default_rng(1000 + model_count)
    covariance_error = generator.normal(
        0.0,
        0.01,
        size=dense_design.shape[0],
    )
    observations = dense_design @ risks + 2.0 * covariance_error

    recovered, *_ = np.linalg.lstsq(dense_design, observations, rcond=None)
    recovery_error = np.linalg.norm(recovered - risks)
    theorem_bound = (
        2.0
        / np.sqrt(model_count - 2)
        * np.linalg.norm(covariance_error)
    )
    assert recovery_error <= theorem_bound + 1.0e-12


def test_shift_matching_recovers_an_exact_simplex_combination() -> None:
    shifts = np.eye(3, dtype=np.float64)
    expected = np.asarray([0.2, 0.3, 0.5], dtype=np.float64)
    target = expected @ shifts

    alpha, diagnostics = match_shift_convex_combination(
        shifts,
        target,
        regularization=0.0,
        descriptor_floor=0.01,
    )

    np.testing.assert_allclose(alpha, expected, atol=1.0e-8, rtol=0.0)
    assert np.all(alpha >= 0.0)
    assert alpha.sum() == pytest.approx(1.0, abs=1.0e-12)
    assert diagnostics["descriptor_rmse"] < 1.0e-8
    assert 1.0 <= diagnostics["effective_shift_count"] <= shifts.shape[0]


def test_transport_statistics_uses_the_same_convex_weights() -> None:
    alpha = np.asarray([0.15, 0.35, 0.5], dtype=np.float64)
    risks = np.arange(3 * 4 * 2, dtype=np.float64).reshape(3, 4, 2) / 100.0
    covariances = np.arange(
        3 * 2 * 4 * 4,
        dtype=np.float64,
    ).reshape(3, 2, 4, 4) / 1000.0

    transported_risks, transported_covariances = transport_statistics(
        alpha,
        risks,
        covariances,
    )

    np.testing.assert_allclose(transported_risks, np.tensordot(alpha, risks, axes=(0, 0)))
    np.testing.assert_allclose(
        transported_covariances,
        np.tensordot(alpha, covariances, axes=(0, 0)),
    )


def test_identical_shift_bank_has_zero_bootstrap_uncertainty() -> None:
    shift_count, model_count, band_count = 5, 6, 2
    risks = np.stack(
        [
            np.linspace(0.06, 0.31, model_count),
            np.linspace(0.09, 0.39, model_count),
        ],
        axis=1,
    )
    shift_risks = np.repeat(risks[None, :, :], shift_count, axis=0)
    shift_covariances = np.zeros(
        (shift_count, band_count, model_count, model_count),
        dtype=np.float64,
    )
    disagreements = np.zeros((band_count, model_count, model_count), dtype=np.float64)
    for band in range(band_count):
        disagreements[band] = risks[:, band, None] + risks[None, :, band]
        np.fill_diagonal(disagreements[band], 0.0)
    arrays = {
        "shift_deltas": np.zeros((shift_count, 4), dtype=np.float64),
        "target_delta": np.zeros(4, dtype=np.float64),
        "band_risks": shift_risks,
        "band_covariances": shift_covariances,
    }

    uncertainty = bootstrap_uncertainty(
        arrays=arrays,
        disagreements=disagreements,
        samples=16,
        seed=20260819,
        transport_regularization=0.1,
        descriptor_floor=0.05,
        risk_ridge=0.0,
        pair_weight_power=0.0,
    )

    np.testing.assert_allclose(uncertainty, np.zeros(model_count), atol=1.0e-12, rtol=0.0)
