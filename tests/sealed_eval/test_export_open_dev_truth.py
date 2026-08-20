from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from sealed_eval import export_open_dev_truth as exporter


def _candidate(root: Path, identifier: str, predictions: np.ndarray) -> dict:
    directory = root / identifier
    directory.mkdir(parents=True)
    np.savez_compressed(
        directory / "target_public.npz",
        hard_predictions=predictions.astype(np.int64),
    )
    return {
        "path": directory,
        "metadata": {
            "candidate_id": identifier,
            "task": "ACMv9_to_Citationv1",
        },
    }


def test_export_open_dev_truth_is_restricted_to_gate1_tasks() -> None:
    with pytest.raises(ValueError, match="restricted to Gate-1"):
        exporter.validate_open_development_tasks(["ACMv9_to_DBLPv7"])


def test_export_open_dev_truth_builds_candidate_level_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = [
        _candidate(tmp_path, "best", np.asarray([0, 1, 1, 0])),
        _candidate(tmp_path, "bad", np.asarray([1, 1, 0, 0])),
    ]

    monkeypatch.setattr(
        exporter,
        "discover_candidate_records",
        lambda candidate_root, task, verify_hashes=True: records,
    )
    monkeypatch.setattr(exporter, "candidate_bank_hash", lambda records: "bank")
    monkeypatch.setattr(
        exporter,
        "load_target_labels",
        lambda task, purpose, candidate_bank_sha256, expected_num_nodes: np.asarray([0, 1, 1, 0]),
    )

    report = exporter.export_open_development_truth(
        Namespace(
            trusted_evaluator=True,
            candidate_root=tmp_path,
            tasks=["ACMv9_to_Citationv1"],
            purpose="development_open_truth",
            metadata_only_candidate_check=False,
            expected_candidates_per_task=2,
        )
    )

    assert report["label_access_count"] == 0
    assert report["evaluator_target_label_read_count"] == 1
    assert report["final_sealed_tasks_exposed"] == 0
    task = report["tasks"][0]
    assert task["oracle_candidate_id"] == "best"
    assert task["candidate_truth"]["best"]["target_error"] == 0.0
    assert task["candidate_truth"]["bad"]["target_error"] == 0.5


def test_export_open_dev_truth_requires_trusted_evaluator(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="trusted-evaluator"):
        exporter.export_open_development_truth(
            Namespace(
                trusted_evaluator=False,
                candidate_root=tmp_path,
                tasks=["ACMv9_to_Citationv1"],
                purpose="development_open_truth",
                metadata_only_candidate_check=False,
                expected_candidates_per_task=None,
            )
        )
