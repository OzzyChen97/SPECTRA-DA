from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.trajectory_export.schema import sha256_file
from selector.spectra_cal import load_calibration_sidecars


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _fixture(tmp_path: Path, *, target_label_access_count: int = 0) -> tuple[Path, dict]:
    directory = tmp_path / "sidecar"
    directory.mkdir()
    arrays_path = directory / "calibration_sidecar.npz"
    np.savez_compressed(
        arrays_path,
        shift_deltas=np.zeros((2, 3), dtype=np.float64),
        band_risks=np.zeros((2, 4, 3), dtype=np.float32),
        band_covariances=np.zeros((2, 3, 4, 4), dtype=np.float32),
    )
    artifact_hash = sha256_file(arrays_path)
    spectral_config = {"num_bands": 3, "sigma": 0.55, "chebyshev_order": 8}
    base_metadata = {
        "candidate_bank_sha256": "candidate-bank",
        "candidate_ids": ["a", "b", "c", "d"],
        "spectral_config": spectral_config,
        "artifact_sha256": {"calibration.npz": "parent-calibration"},
    }
    metadata = {
        "task": "A_to_B",
        "candidate_count": 4,
        "candidate_bank_sha256": "candidate-bank",
        "candidate_ids": ["a", "b", "c", "d"],
        "parent_calibration_sha256": "parent-calibration",
        "spectral_config": spectral_config,
        "selected_shifts": [
            {
                "calibration_family": "target_matched",
                "candidate_index": index,
                "node_count": 10,
            }
            for index in range(2)
        ],
        "target_label_access_count": target_label_access_count,
        "protocol_violation_count": 0,
        "artifact_sha256": {arrays_path.name: artifact_hash},
    }
    _write_json(directory / "metadata.json", metadata)
    manifest = {
        "schema_version": 1,
        "rank_gate": "source_nodes_gte_candidate_count_for_feature_mask_grid",
        "target_label_access_count": 0,
        "protocol_violation_count": 0,
        "tasks": {
            "A_to_B": [
                {"path": str(directory), "artifact_sha256": artifact_hash}
            ]
        },
    }
    manifest_path = tmp_path / "manifest.json"
    _write_json(manifest_path, manifest)
    return manifest_path, base_metadata


def test_load_calibration_sidecars_binds_and_merges_source_only_artifacts(
    tmp_path: Path,
) -> None:
    manifest_path, base_metadata = _fixture(tmp_path)
    arrays, names, accepted, skipped = load_calibration_sidecars(
        manifest_path,
        task="A_to_B",
        base_metadata=base_metadata,
    )

    assert arrays["shift_deltas"].shape == (2, 3)
    assert arrays["band_risks"].shape == (2, 4, 3)
    assert arrays["band_covariances"].shape == (2, 3, 4, 4)
    assert names == [
        "sidecar0:target_matched:0",
        "sidecar0:target_matched:1",
    ]
    assert accepted[0]["shift_count"] == 2
    assert skipped == []


def test_load_calibration_sidecars_rejects_target_label_access(tmp_path: Path) -> None:
    manifest_path, base_metadata = _fixture(
        tmp_path,
        target_label_access_count=1,
    )

    with pytest.raises(RuntimeError, match="target-label access"):
        load_calibration_sidecars(
            manifest_path,
            task="A_to_B",
            base_metadata=base_metadata,
        )
