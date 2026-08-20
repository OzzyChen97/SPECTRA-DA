from __future__ import annotations

import json
from argparse import Namespace
from types import SimpleNamespace

import pytest

from scripts.package_external_evaluation import package, validate_selection
from scripts.trajectory_export.schema import REQUIRED_METADATA_KEYS


def _selection() -> dict:
    return {
        "task": "A_to_B",
        "selector": "entropy",
        "candidate_bank_sha256": "bank",
        "candidate_count": 3,
        "candidate_scores": {"a": 0.3, "b": 0.1, "c": 0.2},
        "score_direction": "minimize",
        "selected_candidate_id": "b",
        "label_access_count": 0,
        "protocol_violation_count": 0,
    }


def test_validate_selection_accepts_complete_label_free_scores() -> None:
    assert (
        validate_selection(
            _selection(),
            task="A_to_B",
            bank_hash="bank",
            candidate_ids=["a", "b", "c"],
        )
        == "entropy"
    )


def test_validate_selection_rejects_target_truth_fields() -> None:
    document = _selection()
    document["diagnostics"] = {"oracle_candidate_index": 1}
    with pytest.raises(RuntimeError, match="forbidden target-truth"):
        validate_selection(
            document,
            task="A_to_B",
            bank_hash="bank",
            candidate_ids=["a", "b", "c"],
        )


def test_validate_selection_rejects_wrong_arg_opt() -> None:
    document = _selection()
    document["selected_candidate_id"] = "a"
    with pytest.raises(ValueError, match="arg-opt mismatch"):
        validate_selection(
            document,
            task="A_to_B",
            bank_hash="bank",
            candidate_ids=["a", "b", "c"],
        )


def test_package_can_include_one_requested_selector_with_metadata_only_check(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr("scripts.package_external_evaluation.TASKS", [SimpleNamespace(id="A_to_B")])
    candidate_root = tmp_path / "candidates"
    checkpoint = candidate_root / "A_to_B" / "method" / "config" / "seed_1" / "checkpoint_1"
    checkpoint.mkdir(parents=True)
    for name in ("source_val.npz", "target_public.npz", "model_state.pt"):
        (checkpoint / name).write_bytes(b"placeholder")

    metadata = {key: "x" for key in REQUIRED_METADATA_KEYS}
    metadata.update(
        {
            "artifact_sha256": {
                "source_val.npz": "source",
                "target_public.npz": "target",
                "model_state.pt": "state",
            },
            "candidate_id": "a",
            "config": {},
            "config_id": "cfg",
            "epoch": 1,
            "method": "Method",
            "schema_version": 1,
            "seed": 1,
            "source_graph_sha256": "source_graph",
            "source_split_sha256": "source_split",
            "target_graph_sha256": "target_graph",
            "target_label_access_count": 0,
            "target_public_has_labels": False,
            "task": "A_to_B",
        }
    )
    (checkpoint / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    selection_root = tmp_path / "selections" / "A_to_B"
    selection_root.mkdir(parents=True)
    selected = {
        **_selection(),
        "task": "A_to_B",
        "selector": "spectra_robust",
        "candidate_count": 1,
        "candidate_scores": {"a": 0.1},
        "selected_candidate_id": "a",
    }
    ignored = {**selected, "selector": "spectra_global_robust"}

    from scripts.trajectory_export.schema import candidate_bank_hash
    from scripts.package_external_evaluation import discover_candidate_metadata_records

    bank_hash = candidate_bank_hash(
        discover_candidate_metadata_records(candidate_root, "A_to_B")
    )
    selected["candidate_bank_sha256"] = bank_hash
    ignored["candidate_bank_sha256"] = bank_hash
    (selection_root / "spectra_robust.json").write_text(
        json.dumps(selected), encoding="utf-8"
    )
    (selection_root / "spectra_global_robust.json").write_text(
        json.dumps(ignored), encoding="utf-8"
    )

    manifest = package(
        Namespace(
            candidate_root=candidate_root,
            selection_root=tmp_path / "selections",
            output_root=tmp_path / "submission",
            archive=None,
            expected_candidates_per_task=1,
            selector=["spectra_robust"],
            trust_candidate_metadata_hashes=False,
            metadata_only_candidate_check=True,
        )
    )

    assert manifest["selectors"] == ["spectra_robust"]
    assert manifest["selector_count"] == 1
    assert manifest["candidate_validation_mode"] == "metadata_only"
    assert (tmp_path / "submission" / "selections" / "A_to_B" / "spectra_robust.json").is_file()
    assert not (
        tmp_path / "submission" / "selections" / "A_to_B" / "spectra_global_robust.json"
    ).exists()
