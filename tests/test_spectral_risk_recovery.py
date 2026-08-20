from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from selector.risk_recovery import (
    closed_form_recovery,
    pair_design,
    pairwise_band_disagreement,
)
from spectral_filters import (
    chebyshev_coefficients,
    frame_approximation_diagnostics,
    tight_window_values,
)


def test_tight_windows_partition_spectral_energy() -> None:
    eigenvalues = np.linspace(0.0, 2.0, 1001)
    windows = tight_window_values(eigenvalues, num_bands=3, sigma=0.55)
    assert np.sum(np.square(windows), axis=0) == pytest.approx(np.ones_like(eigenvalues), abs=1e-12)


def test_registered_chebyshev_frame_meets_guardrail() -> None:
    coefficients = chebyshev_coefficients(num_bands=3, sigma=0.55, order=8)
    diagnostics = frame_approximation_diagnostics(
        coefficients,
        num_bands=3,
        sigma=0.55,
    )
    assert diagnostics["max_frame_error"] < 0.01


def test_complete_pair_design_has_claimed_minimum_singular_value() -> None:
    model_count = 9
    design, _, _ = pair_design(model_count)
    gram = (design.T @ design).toarray()
    expected = (model_count - 2) * np.eye(model_count) + np.ones((model_count, model_count))
    assert gram == pytest.approx(expected)
    assert np.linalg.svd(design.toarray(), compute_uv=False).min() == pytest.approx(
        np.sqrt(model_count - 2)
    )


def test_complete_pair_design_is_cached_by_committee_size() -> None:
    first = pair_design(11)
    second = pair_design(11)
    assert first[0] is second[0]
    assert first[1] is second[1]
    assert first[2] is second[2]


def test_closed_form_exactly_recovers_decorrelated_pair_risks() -> None:
    risks = np.asarray([0.07, 0.11, 0.19, 0.31, 0.42])
    disagreement = risks[:, None] + risks[None, :]
    np.fill_diagonal(disagreement, 0.0)
    assert closed_form_recovery(disagreement) == pytest.approx(risks)


def test_band_disagreement_identity_includes_error_covariance() -> None:
    generator = torch.Generator().manual_seed(23)
    model_count, band_count, nodes, classes = 6, 3, 17, 4
    filtered_errors = torch.randn(
        model_count,
        band_count,
        nodes,
        classes,
        generator=generator,
    )
    disagreement = pairwise_band_disagreement(filtered_errors, nodes)
    for band in range(band_count):
        flat = filtered_errors[:, band].reshape(model_count, -1)
        risks = torch.sum(flat.square(), dim=1) / nodes
        covariance = flat @ flat.T / nodes
        expected = risks[:, None] + risks[None, :] - 2.0 * covariance
        assert disagreement[band] == pytest.approx(expected.clamp_min(0.0), abs=1e-5)


def test_one_hot_squared_risk_is_twice_classification_error() -> None:
    labels = torch.tensor([0, 1, 2, 1, 0, 2, 2, 1])
    predictions = torch.tensor([0, 2, 2, 1, 1, 2, 0, 1])
    label_matrix = F.one_hot(labels, num_classes=3).float()
    prediction_matrix = F.one_hot(predictions, num_classes=3).float()
    squared_risk = torch.sum((prediction_matrix - label_matrix).square()) / labels.numel()
    classification_error = torch.mean((predictions != labels).float())
    assert 0.5 * squared_risk.item() == pytest.approx(classification_error.item())
