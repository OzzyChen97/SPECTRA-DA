from __future__ import annotations

import math

import pytest

from scripts.trajectory_export.schema import _validate_metadata


def valid_metadata() -> dict[str, object]:
    return {
        "artifact_sha256": {},
        "candidate_id": "candidate",
        "config": {},
        "config_id": "config",
        "cuda_visible_devices": "7",
        "epoch": 10,
        "family": "Citation",
        "logical_device": "cuda:0",
        "max_memory_bytes": 1024,
        "method": "ADAlign",
        "physical_gpu": 7,
        "seed": 2026,
        "source": "ACMv9",
        "source_graph_sha256": "source-graph",
        "source_num_nodes": 10,
        "source_split_sha256": "source-split",
        "source_train_nodes": 6,
        "source_val_macro_f1": 0.5,
        "source_val_micro_f1": 0.6,
        "source_val_nodes": 4,
        "target": "Citationv1",
        "target_entropy": 0.7,
        "target_graph_sha256": "target-graph",
        "target_label_access_count": 0,
        "target_num_nodes": 12,
        "target_public_has_labels": False,
        "task": "ACMv9_to_Citationv1",
        "train_alignment_loss": 0.1,
        "train_source_loss": 0.2,
        "train_total_loss": 0.3,
        "trajectory_elapsed_seconds": 1.0,
    }


def test_checkpoint_metadata_requires_all_trajectory_diagnostics() -> None:
    metadata = valid_metadata()
    _validate_metadata(metadata)
    del metadata["target_entropy"]
    with pytest.raises(ValueError, match="target_entropy"):
        _validate_metadata(metadata)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("train_source_loss", math.nan),
        ("train_alignment_loss", math.inf),
        ("trajectory_elapsed_seconds", 0.0),
        ("max_memory_bytes", -1),
        ("source_val_micro_f1", 1.1),
    ],
)
def test_checkpoint_metadata_rejects_invalid_diagnostics(field: str, value: object) -> None:
    metadata = valid_metadata()
    metadata[field] = value
    with pytest.raises(ValueError):
        _validate_metadata(metadata)
