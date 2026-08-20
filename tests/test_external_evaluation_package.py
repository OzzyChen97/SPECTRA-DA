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


def _single_candidate_bank(tmp_path, monkeypatch):
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

    from scripts.package_external_evaluation import discover_candidate_metadata_records
    from scripts.trajectory_export.schema import candidate_bank_hash

    bank_hash = candidate_bank_hash(
        discover_candidate_metadata_records(candidate_root, "A_to_B")
    )
    return candidate_root, bank_hash


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


def test_package_includes_multiple_selectors_by_default_with_metadata_only_check(
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

    from scripts.package_external_evaluation import discover_candidate_metadata_records
    from scripts.trajectory_export.schema import candidate_bank_hash

    bank_hash = candidate_bank_hash(
        discover_candidate_metadata_records(candidate_root, "A_to_B")
    )
    baseline_root = tmp_path / "baseline_selections" / "A_to_B"
    spectra_root = tmp_path / "spectra_selections" / "A_to_B"
    baseline_root.mkdir(parents=True)
    spectra_root.mkdir(parents=True)
    baseline_manifest = tmp_path / "baseline_selections" / "baseline_suite_manifest.json"
    reliable_manifest = tmp_path / "spectra_selections" / "reliable_freeze_manifest.json"
    baseline_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "purpose": "label-free baseline selector suite",
                "label_access_count": 0,
                "protocol_violation_count": 0,
            }
        ),
        encoding="utf-8",
    )
    reliable_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "purpose": "frozen reliable selector submission",
                "label_access_count": 0,
                "protocol_violation_count": 0,
            }
        ),
        encoding="utf-8",
    )
    for selector in ("transfer_score", "entropy", "spectra_reliable"):
        selection = {
            **_selection(),
            "task": "A_to_B",
            "selector": selector,
            "candidate_bank_sha256": bank_hash,
            "candidate_count": 1,
            "candidate_scores": {"a": 0.1},
            "selected_candidate_id": "a",
        }
        target_root = spectra_root if selector == "spectra_reliable" else baseline_root
        (target_root / f"{selector}.json").write_text(
            json.dumps(selection),
            encoding="utf-8",
        )

    manifest = package(
        Namespace(
            candidate_root=candidate_root,
            selection_root=[tmp_path / "baseline_selections", tmp_path / "spectra_selections"],
            output_root=tmp_path / "submission_multi",
            archive=None,
            expected_candidates_per_task=1,
            selector=None,
            require_selector=["transfer_score"],
            min_selector_count=2,
            trust_candidate_metadata_hashes=False,
            metadata_only_candidate_check=True,
        )
    )

    assert manifest["selectors"] == ["entropy", "spectra_reliable", "transfer_score"]
    assert manifest["selector_count"] == 3
    assert manifest["selection_root_count"] == 2
    assert manifest["selection_root_manifest_count"] == 2
    provenance = {
        entry["manifest_file"]: entry
        for entry in manifest["selection_root_manifests"]
    }
    assert provenance["baseline_suite_manifest.json"]["purpose"] == "label-free baseline selector suite"
    assert provenance["baseline_suite_manifest.json"]["label_access_count"] == 0
    assert provenance["reliable_freeze_manifest.json"]["purpose"] == "frozen reliable selector submission"
    assert provenance["reliable_freeze_manifest.json"]["protocol_violation_count"] == 0
    assert len(provenance["baseline_suite_manifest.json"]["manifest_sha256"]) == 64
    for selector in manifest["selectors"]:
        assert (
            tmp_path / "submission_multi" / "selections" / "A_to_B" / f"{selector}.json"
        ).is_file()


def test_package_rejects_single_selector_final_guard(tmp_path, monkeypatch) -> None:
    candidate_root, bank_hash = _single_candidate_bank(tmp_path, monkeypatch)
    selection_task_root = tmp_path / "selections" / "A_to_B"
    selection_task_root.mkdir(parents=True)
    selection = {
        **_selection(),
        "task": "A_to_B",
        "selector": "spectra_robust",
        "candidate_bank_sha256": bank_hash,
        "candidate_count": 1,
        "candidate_scores": {"a": 0.1},
        "selected_candidate_id": "a",
    }
    (selection_task_root / "spectra_robust.json").write_text(
        json.dumps(selection),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing required selectors"):
        package(
            Namespace(
                candidate_root=candidate_root,
                selection_root=tmp_path / "selections",
                output_root=tmp_path / "submission_single_selector",
                archive=None,
                expected_candidates_per_task=1,
                selector=None,
                require_selector=["transfer_score"],
                min_selector_count=2,
                trust_candidate_metadata_hashes=False,
                metadata_only_candidate_check=True,
            )
        )


def test_package_rejects_unsafe_selection_root_provenance(tmp_path, monkeypatch) -> None:
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

    from scripts.package_external_evaluation import discover_candidate_metadata_records
    from scripts.trajectory_export.schema import candidate_bank_hash

    bank_hash = candidate_bank_hash(
        discover_candidate_metadata_records(candidate_root, "A_to_B")
    )
    selection_task_root = tmp_path / "selections" / "A_to_B"
    selection_task_root.mkdir(parents=True)
    selection = {
        **_selection(),
        "task": "A_to_B",
        "selector": "entropy",
        "candidate_bank_sha256": bank_hash,
        "candidate_count": 1,
        "candidate_scores": {"a": 0.1},
        "selected_candidate_id": "a",
    }
    (selection_task_root / "entropy.json").write_text(json.dumps(selection), encoding="utf-8")
    (tmp_path / "selections" / "baseline_suite_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "purpose": "label-free baseline selector suite",
                "label_access_count": 1,
                "protocol_violation_count": 0,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="not sealed-safe"):
        package(
            Namespace(
                candidate_root=candidate_root,
                selection_root=tmp_path / "selections",
                output_root=tmp_path / "submission_unsafe",
                archive=None,
                expected_candidates_per_task=1,
                selector=None,
                trust_candidate_metadata_hashes=False,
                metadata_only_candidate_check=True,
            )
        )
