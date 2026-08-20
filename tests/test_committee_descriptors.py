from __future__ import annotations

import numpy as np
import pytest

from shift_simulator.committee_descriptors import (
    COMMITTEE_DESCRIPTOR_NAMES,
    build_descriptor_report,
    committee_descriptor,
    pairwise_disagreement_matrix,
)


def _record(
    tmp_path,
    identifier: str,
    probabilities: np.ndarray,
    predictions: np.ndarray,
    *,
    method: str,
    config_id: str = "cfg",
    seed: int = 1,
    epoch: int = 10,
) -> dict:
    directory = tmp_path / identifier
    directory.mkdir()
    np.savez_compressed(
        directory / "target_public.npz",
        probabilities=probabilities.astype(np.float64),
        hard_predictions=predictions.astype(np.int64),
        logits=np.log(np.clip(probabilities, 1.0e-6, 1.0)),
        embeddings=np.ones((predictions.shape[0], 2), dtype=np.float64),
    )
    return {
        "path": directory,
        "metadata": {
            "candidate_id": identifier,
            "task": "A_to_B",
            "method": method,
            "config_id": config_id,
            "seed": seed,
            "epoch": epoch,
        },
    }


def test_pairwise_disagreement_matrix_uses_hard_predictions() -> None:
    predictions = np.asarray(
        [
            [0, 0, 1, 1],
            [0, 1, 1, 1],
            [1, 1, 0, 0],
        ],
        dtype=np.int64,
    )

    disagreement = pairwise_disagreement_matrix(predictions, class_count=2)

    np.testing.assert_allclose(
        disagreement,
        np.asarray(
            [
                [0.0, 0.25, 1.0],
                [0.25, 0.0, 0.75],
                [1.0, 0.75, 0.0],
            ]
        ),
    )


def test_committee_descriptor_is_finite_and_label_free(tmp_path) -> None:
    probabilities = [
        np.asarray([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]]),
        np.asarray([[0.7, 0.3], [0.4, 0.6], [0.2, 0.8], [0.2, 0.8]]),
        np.asarray([[0.1, 0.9], [0.2, 0.8], [0.8, 0.2], [0.9, 0.1]]),
    ]
    records = [
        _record(tmp_path, "m1_e10", probabilities[0], np.asarray([0, 0, 1, 1]), method="M1", epoch=10),
        _record(tmp_path, "m1_e20", probabilities[1], np.asarray([0, 1, 1, 1]), method="M1", epoch=20),
        _record(tmp_path, "m2_e10", probabilities[2], np.asarray([1, 1, 0, 0]), method="M2", seed=2, epoch=10),
    ]

    descriptor, names = committee_descriptor(records)

    assert names == list(COMMITTEE_DESCRIPTOR_NAMES)
    assert descriptor.shape == (len(COMMITTEE_DESCRIPTOR_NAMES),)
    assert np.isfinite(descriptor).all()
    values = dict(zip(names, descriptor))
    assert values["checkpoint_drift_mean"] == pytest.approx(0.25)
    assert values["between_method_disagreement"] > values["within_method_disagreement"]


def test_committee_descriptor_rejects_label_like_target_fields(tmp_path) -> None:
    directory = tmp_path / "bad"
    directory.mkdir()
    np.savez_compressed(
        directory / "target_public.npz",
        probabilities=np.asarray([[0.5, 0.5], [0.2, 0.8]]),
        hard_predictions=np.asarray([0, 1]),
        labels=np.asarray([0, 1]),
    )
    records = [
        {
            "path": directory,
            "metadata": {
                "candidate_id": "bad",
                "method": "M",
                "config_id": "cfg",
                "seed": 1,
                "epoch": 1,
            },
        },
        {
            "path": directory,
            "metadata": {
                "candidate_id": "bad2",
                "method": "M",
                "config_id": "cfg",
                "seed": 1,
                "epoch": 2,
            },
        },
    ]

    with pytest.raises(RuntimeError, match="label-like"):
        committee_descriptor(records)


def test_build_descriptor_report_records_protocol_boundary(tmp_path, monkeypatch) -> None:
    probabilities = np.asarray([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2]])
    records = [
        _record(tmp_path, "a", probabilities, np.asarray([0, 1, 0]), method="M1", epoch=1),
        _record(tmp_path, "b", probabilities[::-1], np.asarray([1, 0, 1]), method="M2", seed=2, epoch=1),
    ]
    monkeypatch.setattr(
        "shift_simulator.committee_descriptors.candidate_bank_hash",
        lambda records: "bank",
    )

    report = build_descriptor_report(records, task="A_to_B")

    assert report["candidate_bank_sha256"] == "bank"
    assert report["descriptor_family"] == "committee"
    assert len(report["descriptor"]) == len(COMMITTEE_DESCRIPTOR_NAMES)
    assert report["label_access_count"] == 0
    assert report["protocol_violation_count"] == 0
