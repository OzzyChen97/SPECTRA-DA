from __future__ import annotations

import numpy as np
import torch

from selector.baselines import (
    agreement_on_the_line_scores,
    generalization_disagreement_equality,
    soft_neighborhood_density,
)
from selector.dev import dev_risk, domain_importance_weights
from selector.transfer_score import (
    classifier_uniformity,
    normalized_information_maximization,
    transfer_score,
)


def test_blocked_snd_matches_released_full_matrix_formula() -> None:
    probabilities = np.asarray(
        [
            [0.80, 0.15, 0.05],
            [0.75, 0.20, 0.05],
            [0.10, 0.75, 0.15],
            [0.05, 0.20, 0.75],
            [0.10, 0.15, 0.75],
        ],
        dtype=np.float32,
    )
    values = torch.nn.functional.normalize(torch.from_numpy(probabilities), p=2, dim=1)
    similarities = values @ values.T / 0.05
    similarities.fill_diagonal_(-1.0 / 0.05)
    neighbors = torch.softmax(similarities, dim=1)
    expected = float((-(neighbors * torch.log(neighbors + 1e-5)).sum(dim=1)).mean().item())

    actual = soft_neighborhood_density(
        probabilities,
        temperature=0.05,
        block_size=2,
        device="cpu",
    )
    assert actual == pytest.approx(expected, abs=1e-6)


def test_aol_variants_are_finite_and_label_free_on_target() -> None:
    rng = np.random.default_rng(17)
    model_count = 7
    source_nodes = 800
    target_nodes = 900
    classes = 4
    source_labels = rng.integers(classes, size=source_nodes)
    target_latent = rng.integers(classes, size=target_nodes)
    source_predictions = []
    target_predictions = []
    for index in range(model_count):
        source = source_labels.copy()
        target = target_latent.copy()
        source_flip = rng.random(source_nodes) < (0.12 + 0.05 * index)
        target_flip = rng.random(target_nodes) < (0.18 + 0.055 * index)
        source[source_flip] = rng.integers(classes, size=int(source_flip.sum()))
        target[target_flip] = rng.integers(classes, size=int(target_flip.sum()))
        source_predictions.append(source)
        target_predictions.append(target)

    aline_s, aline_d = agreement_on_the_line_scores(
        np.stack(source_predictions),
        source_labels,
        np.stack(target_predictions),
    )
    assert aline_s.shape == (model_count,)
    assert aline_d.shape == (model_count,)
    assert np.isfinite(aline_s).all()
    assert np.isfinite(aline_d).all()
    assert ((0.0 <= aline_s) & (aline_s <= 1.0)).all()
    assert ((0.0 <= aline_d) & (aline_d <= 1.0)).all()


def test_aol_rejects_too_small_committees() -> None:
    predictions = np.zeros((2, 10), dtype=np.int64)
    labels = np.zeros(10, dtype=np.int64)
    with pytest.raises(ValueError, match="at least three"):
        agreement_on_the_line_scores(predictions, labels, predictions)


def test_dev_is_deterministic_and_finite() -> None:
    rng = np.random.default_rng(5)
    source = rng.normal(0.0, 1.0, size=(90, 6)).astype(np.float32)
    target = rng.normal(0.4, 1.0, size=(100, 6)).astype(np.float32)
    errors = (source[:, 0] > 0.0).astype(np.float64)
    first = domain_importance_weights(source, target, source, seed=19)
    second = domain_importance_weights(source, target, source, seed=19)
    assert np.array_equal(first, second)
    assert np.isfinite(first).all()
    assert np.isfinite(dev_risk(first, errors))


def test_transfer_score_components() -> None:
    simplex = np.asarray(
        [[1.0, 0.0], [-0.5, np.sqrt(3.0) / 2.0], [-0.5, -np.sqrt(3.0) / 2.0]]
    )
    assert classifier_uniformity(simplex) == pytest.approx(0.0, abs=1e-12)
    probabilities = np.asarray([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8]])
    assert normalized_information_maximization(probabilities) > 0.0
    rng = np.random.default_rng(8)
    embeddings = rng.normal(size=(80, 2)).astype(np.float32)
    tiled_probabilities = np.tile(probabilities, (20, 1))
    head = np.asarray([[1.0, 0.0], [-1.0, 0.0]])
    first = transfer_score(
        embeddings,
        tiled_probabilities,
        head,
        candidate_id="candidate",
    )
    second = transfer_score(
        embeddings,
        tiled_probabilities,
        head,
        candidate_id="candidate",
    )
    assert first == second
    assert np.isfinite(first)


def test_gde_compares_only_matching_independent_seeds(tmp_path) -> None:
    records = []
    predictions = {
        ("cfg_a", 1): np.asarray([0, 0, 1, 1]),
        ("cfg_a", 2): np.asarray([0, 1, 1, 0]),
        ("cfg_b", 1): np.asarray([1, 1, 0, 0]),
        ("cfg_b", 2): np.asarray([1, 1, 0, 0]),
    }
    for (config, seed), values in predictions.items():
        directory = tmp_path / config / str(seed)
        directory.mkdir(parents=True)
        np.savez_compressed(directory / "target_public.npz", hard_predictions=values)
        records.append(
            {
                "path": directory,
                "metadata": {
                    "candidate_id": f"{config}-{seed}",
                    "method": "method",
                    "config_id": config,
                    "epoch": 10,
                    "seed": seed,
                },
            }
        )
    scores = generalization_disagreement_equality(records)
    assert scores["cfg_a-1"] == pytest.approx(0.5)
    assert scores["cfg_a-2"] == pytest.approx(0.5)
    assert scores["cfg_b-1"] == pytest.approx(0.0)
    assert scores["cfg_b-2"] == pytest.approx(0.0)


import pytest
