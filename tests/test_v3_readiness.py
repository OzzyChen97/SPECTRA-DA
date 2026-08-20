from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

from scripts import check_v3_readiness as readiness


def _write_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document), encoding="utf-8")


def _selection(
    *,
    task: str,
    selector: str,
    bank: str,
    count: int,
    direction: str = "minimize",
) -> dict:
    scores = {f"c{i}": float(i) for i in range(count)}
    selected = "c0" if direction == "minimize" else f"c{count - 1}"
    return {
        "schema_version": 1,
        "task": task,
        "selector": selector,
        "candidate_bank_sha256": bank,
        "candidate_count": count,
        "candidate_scores": scores,
        "score_direction": direction,
        "score_semantics": "ranking",
        "selected_candidate_id": selected,
        "label_access_count": 0,
        "protocol_violation_count": 0,
    }


def _truth_report(count: int) -> dict:
    return {
        "schema_version": 1,
        "scope": "open_development_gate1_candidate_truth",
        "open_development_tasks": list(readiness.OPEN_DEVELOPMENT_TASKS),
        "task_count": len(readiness.OPEN_DEVELOPMENT_TASKS),
        "candidate_count_total": count * len(readiness.OPEN_DEVELOPMENT_TASKS),
        "label_access_count": 0,
        "protocol_violation_count": 0,
        "final_sealed_tasks_exposed": 0,
        "tasks": [
            {
                "task": task,
                "candidate_bank_sha256": f"bank-{task}",
                "candidate_count": count,
                "candidate_truth": {
                    f"c{i}": {"target_error": float(i) / count}
                    for i in range(count)
                },
            }
            for task in readiness.OPEN_DEVELOPMENT_TASKS
        ],
    }


def test_v3_readiness_accepts_synthetic_complete_artifacts(tmp_path: Path) -> None:
    candidate_count = 3
    truth_path = tmp_path / "open_dev" / "truth.json"
    _write_json(truth_path, _truth_report(candidate_count))

    spectra_root = tmp_path / "spectra"
    transfer_root = tmp_path / "transfer"
    for task in readiness.OPEN_DEVELOPMENT_TASKS:
        bank = f"bank-{task}"
        _write_json(
            spectra_root / task / "spectra_trust.json",
            _selection(
                task=task,
                selector="spectra_trust",
                bank=bank,
                count=candidate_count,
            ),
        )
        _write_json(
            transfer_root / task / "transfer_score.json",
            _selection(
                task=task,
                selector="transfer_score",
                bank=bank,
                count=candidate_count,
                direction="maximize",
            ),
        )

    submission_root = tmp_path / "submission"
    selection_files = {}
    final_selectors = ["spectra_reliable", "transfer_score"]
    for task in [task.id for task in readiness.TASKS]:
        for selector in final_selectors:
            relative = Path("selections") / task / f"{selector}.json"
            path = submission_root / relative
            _write_json(
                path,
                {
                    "schema_version": 1,
                    "task": task,
                    "selector": selector,
                    "candidate_count": candidate_count,
                    "label_access_count": 0,
                    "protocol_violation_count": 0,
                },
            )
            selection_files[relative.as_posix()] = readiness.sha256_file(path)

    manifest_path = submission_root / "submission_manifest.json"
    _write_json(
        manifest_path,
        {
            "schema_version": 1,
            "purpose": "one-time external sealed evaluation submission",
            "task_count": len(readiness.TASKS),
            "candidate_count_per_task": candidate_count,
            "selector_count": len(final_selectors),
            "selectors": final_selectors,
            "selection_files": selection_files,
            "target_label_access_count": 0,
            "protocol_violation_count": 0,
            "contains_target_labels": False,
        },
    )

    result = readiness.run(
        Namespace(
            open_dev_truth=truth_path,
            open_dev_selection_root=[spectra_root, transfer_root],
            final_submission_manifest=manifest_path,
            expected_open_dev_candidates=candidate_count,
            expected_final_candidates=candidate_count,
            required_open_dev_selector=["spectra_trust", "transfer_score"],
            required_final_selector=["transfer_score"],
            min_final_selector_count=2,
        )
    )

    assert result["ok"]
    assert result["label_access_count"] == 0
    assert result["protocol_violation_count"] == 0


def test_v3_readiness_rejects_manifest_without_packaged_selection_files(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "submission" / "submission_manifest.json"
    _write_json(
        manifest_path,
        {
            "schema_version": 1,
            "purpose": "one-time external sealed evaluation submission",
            "task_count": len(readiness.TASKS),
            "candidate_count_per_task": 3,
            "selector_count": 2,
            "selectors": ["spectra_reliable", "transfer_score"],
            "selection_files": {},
            "target_label_access_count": 0,
            "protocol_violation_count": 0,
            "contains_target_labels": False,
        },
    )

    result = readiness.audit_final_submission_manifest(
        manifest_path,
        expected_candidate_count=3,
        min_selector_count=2,
        required_selectors=["transfer_score"],
    )

    assert not result["ok"]
    assert any("missing packaged selection files" in error for error in result["errors"])


def test_v3_readiness_rejects_final_task_in_open_dev_truth(tmp_path: Path) -> None:
    report = _truth_report(count=2)
    report["tasks"].append(
        {
            "task": readiness.FINAL_TASKS[0],
            "candidate_bank_sha256": "bank-final",
            "candidate_count": 2,
            "candidate_truth": {
                "c0": {"target_error": 0.0},
                "c1": {"target_error": 1.0},
            },
        }
    )
    report["final_sealed_tasks_exposed"] = 1
    truth_path = tmp_path / "open_dev" / "truth.json"
    _write_json(truth_path, report)

    result = readiness.audit_open_dev_truth(truth_path, expected_candidate_count=2)

    assert not result["ok"]
    assert any("final sealed tasks exposed" in error for error in result["errors"])
