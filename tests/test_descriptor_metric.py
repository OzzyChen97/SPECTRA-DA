from __future__ import annotations

import argparse
import json

import numpy as np
import pytest

from shift_simulator.descriptor_metric import (
    correction_vectors,
    descriptor_dimension_audit,
    evaluate_descriptor_metric,
    robust_scale_descriptors,
    run,
    shift_family_from_name,
)


def test_correction_vectors_are_zero_when_covariance_is_zero() -> None:
    band_risks = np.asarray(
        [
            [[0.10], [0.20], [0.30], [0.40]],
            [[0.15], [0.25], [0.35], [0.45]],
        ],
        dtype=np.float64,
    )
    band_covariances = np.zeros((2, 1, 4, 4), dtype=np.float64)

    corrections = correction_vectors(band_risks, band_covariances)

    np.testing.assert_allclose(corrections, np.zeros((2, 4)), atol=1.0e-10)


def test_learned_diagonal_metric_identifies_recovery_relevant_dimension() -> None:
    descriptor_axis = np.arange(6, dtype=np.float64)
    nuisance_axis = np.asarray([0.0, 100.0, 0.0, 100.0, 0.0, 100.0])
    descriptors = np.stack([descriptor_axis, nuisance_axis], axis=1)
    corrections = np.stack(
        [
            0.10 * descriptor_axis,
            -0.05 * descriptor_axis,
            0.03 * descriptor_axis,
        ],
        axis=1,
    )

    report = evaluate_descriptor_metric(
        descriptors,
        corrections,
        descriptor_names=["candidate_behavior_axis", "large_scale_nuisance"],
        families=["feature_mask", "feature_mask", "edge_dropout", "edge_dropout", "homophily", "homophily"],
        ridge=1.0e-8,
    )

    assert report["learned_metric_correction_spearman"] is not None
    assert report["learned_metric_correction_spearman"] > 0.95
    assert report["diagonal_metric_weights"][0] > 100.0 * report["diagonal_metric_weights"][1]
    assert report["top_weighted_descriptors"][0]["name"] == "candidate_behavior_axis"


def test_shift_family_parser_keeps_multiword_shift_types() -> None:
    assert shift_family_from_name("feature_mask_035") == "feature_mask"
    assert shift_family_from_name("edge_dropout_010") == "edge_dropout"
    assert shift_family_from_name("label_prior_temperature_2") == "label_prior"
    assert shift_family_from_name("conditional_structure_drop") == "conditional_structure"


def test_robust_scaling_clips_heavy_tail_and_audits_contributions() -> None:
    raw_descriptors = np.asarray(
        [
            [1.0, 0.0, np.nan],
            [2.0, 0.0, 1.0],
            [3.0, 0.0, 1.0],
            [1000.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    processed, preprocessing = robust_scale_descriptors(
        raw_descriptors,
        transform="signed_log1p",
        robust_scale=True,
        clip=2.0,
    )
    audit = descriptor_dimension_audit(
        raw_descriptors,
        processed,
        descriptor_names=["heavy_tail", "constant", "partly_missing"],
        target_descriptor=np.asarray([4.0, 0.0, 1.0], dtype=np.float64),
        transform="signed_log1p",
        robust_scale=True,
        clip=2.0,
    )

    assert preprocessing["filled_nonfinite_count"] == 1
    assert np.isfinite(processed).all()
    assert np.max(np.abs(processed)) <= 2.0
    assert audit[2]["missing_or_nonfinite_count"] == 1
    assert audit[0]["target_nearest_distance_share"] is not None
    assert audit[0]["target_nearest_distance_share"] > 0.99


def test_descriptor_metric_run_writes_label_free_report(tmp_path) -> None:
    calibration_dir = tmp_path / "calibration"
    calibration_dir.mkdir()
    metadata = {
        "task": "A_to_B",
        "calibration_id": "synthetic",
        "candidate_bank_sha256": "bank",
        "candidate_count": 4,
        "target_label_access_count": 0,
        "protocol_violation_count": 0,
        "descriptor_names": ["scale", "nuisance"],
        "shift_specs": [
            {"name": "feature_mask_010"},
            {"name": "feature_mask_020"},
            {"name": "edge_dropout_010"},
            {"name": "edge_dropout_020"},
        ],
    }
    (calibration_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    np.savez_compressed(
        calibration_dir / "calibration.npz",
        shift_deltas=np.asarray(
            [
                [0.0, 2.0],
                [1.0, 0.0],
                [2.0, 2.0],
                [3.0, 0.0],
            ],
            dtype=np.float64,
        ),
        target_delta=np.asarray([4.0, 0.0], dtype=np.float64),
        band_risks=np.asarray(
            [
                [[0.10], [0.20], [0.30], [0.40]],
                [[0.12], [0.22], [0.32], [0.42]],
                [[0.14], [0.24], [0.34], [0.44]],
                [[0.16], [0.26], [0.36], [0.46]],
            ],
            dtype=np.float64,
        ),
        band_covariances=np.zeros((4, 1, 4, 4), dtype=np.float64),
    )
    output = tmp_path / "descriptor_metric.json"

    report = run(
        argparse.Namespace(
            calibration_dir=calibration_dir,
            descriptor_key="shift_deltas",
            target_descriptor_key="target_delta",
            descriptor_transform="none",
            no_robust_scale=False,
            robust_clip=8.0,
            ridge=1.0e-6,
            output=output,
        )
    )

    written = json.loads(output.read_text(encoding="utf-8"))
    assert report["label_access_count"] == 0
    assert report["protocol_violation_count"] == 0
    assert written["objective"] == "descriptor_distance_vs_recovery_relevant_correction_distance"
    assert written["descriptor_key"] == "shift_deltas"
    assert written["target_descriptor_key"] == "target_delta"
    assert written["descriptor_preprocessing"]["robust_scale"] is True
    assert written["descriptor_dimension_audit"][0]["target_nearest_distance_share"] is not None


def test_descriptor_metric_rejects_protocol_violating_artifact(tmp_path) -> None:
    calibration_dir = tmp_path / "calibration"
    calibration_dir.mkdir()
    (calibration_dir / "metadata.json").write_text(
        json.dumps(
            {
                "target_label_access_count": 0,
                "protocol_violation_count": 1,
            }
        ),
        encoding="utf-8",
    )
    np.savez_compressed(
        calibration_dir / "calibration.npz",
        shift_deltas=np.zeros((2, 1), dtype=np.float64),
        band_risks=np.zeros((2, 3, 1), dtype=np.float64),
        band_covariances=np.zeros((2, 1, 3, 3), dtype=np.float64),
    )

    with pytest.raises(RuntimeError, match="protocol violations"):
        run(
            argparse.Namespace(
                calibration_dir=calibration_dir,
                descriptor_key="shift_deltas",
                target_descriptor_key="target_delta",
                descriptor_transform="none",
                no_robust_scale=False,
                robust_clip=8.0,
                ridge=1.0e-6,
                output=None,
            )
        )
