#!/usr/bin/env python3
"""Freeze and validate a task-complete source-only calibration-sidecar manifest."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from protocol.tasks import TASKS  # noqa: E402
from scripts.trajectory_export.schema import atomic_json, sha256_file  # noqa: E402
from selector.run_spectra_suite import calibration_directory  # noqa: E402
from selector.spectra_cal import (  # noqa: E402
    load_calibration,
    load_calibration_sidecars,
)


def _entry(directory: Path) -> dict[str, str]:
    metadata_path = directory / "metadata.json"
    arrays_path = directory / "calibration_sidecar.npz"
    if not metadata_path.is_file() or not arrays_path.is_file():
        raise FileNotFoundError(f"incomplete sidecar directory: {directory}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_hash = metadata.get("artifact_sha256", {}).get(arrays_path.name)
    if not expected_hash or sha256_file(arrays_path) != expected_hash:
        raise ValueError(f"sidecar artifact hash mismatch: {directory}")
    return {"path": str(directory.resolve()), "artifact_sha256": expected_hash}


def build(args: argparse.Namespace) -> dict[str, Any]:
    expected_tasks = [task.id for task in TASKS]
    entries: dict[str, list[dict[str, str]]] = {}
    if args.base_manifest is not None:
        base = json.loads(args.base_manifest.resolve().read_text(encoding="utf-8"))
        if base.get("schema_version") != 1:
            raise ValueError("unsupported base sidecar manifest schema")
        if base.get("target_label_access_count") != 0:
            raise RuntimeError("base sidecar manifest reports target-label access")
        if base.get("protocol_violation_count") != 0:
            raise RuntimeError("base sidecar manifest reports a protocol violation")
        entries.update(
            {
                str(task): [dict(entry) for entry in task_entries]
                for task, task_entries in base.get("tasks", {}).items()
            }
        )

    for root in args.sidecar_root:
        root = root.resolve()
        if not root.is_dir():
            raise FileNotFoundError(root)
        for task_directory in sorted(path for path in root.iterdir() if path.is_dir()):
            if task_directory.name.startswith("."):
                continue
            task = task_directory.name
            if task not in expected_tasks:
                raise ValueError(f"unexpected sidecar task: {task}")
            entries.setdefault(task, []).append(_entry(task_directory))

    if set(entries) != set(expected_tasks):
        missing = sorted(set(expected_tasks) - set(entries))
        extra = sorted(set(entries) - set(expected_tasks))
        raise ValueError(f"manifest task mismatch: missing={missing}, extra={extra}")
    if any(not task_entries for task_entries in entries.values()):
        raise ValueError("every task must contain at least one sidecar")

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "name": args.name,
        "description": (
            "Frozen source-labelled calibration sidecars for 16-task SPECTRA-DA "
            "deployment; no target labels are read."
        ),
        "rank_gate": "source_nodes_gte_candidate_count_for_feature_mask_grid",
        "task_count": len(entries),
        "sidecar_count": sum(len(value) for value in entries.values()),
        "target_label_access_count": 0,
        "protocol_violation_count": 0,
        "tasks": {task: entries[task] for task in expected_tasks},
    }

    calibration_root = args.calibration_root.resolve()
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        encoding="utf-8",
    ) as temporary:
        json.dump(manifest, temporary, sort_keys=True)
        temporary.flush()
        temporary_path = Path(temporary.name)
        validation = []
        for task in expected_tasks:
            metadata, _ = load_calibration(
                calibration_directory(calibration_root, task)
            )
            _, _, accepted, skipped = load_calibration_sidecars(
                temporary_path,
                task=task,
                base_metadata=metadata,
            )
            validation.append(
                {
                    "task": task,
                    "accepted_sidecar_count": len(accepted),
                    "skipped_sidecar_count": len(skipped),
                    "accepted_shift_count": sum(
                        int(item["shift_count"]) for item in accepted
                    ),
                }
            )
    manifest["validation"] = validation
    manifest["accepted_sidecar_count"] = sum(
        item["accepted_sidecar_count"] for item in validation
    )
    manifest["skipped_sidecar_count"] = sum(
        item["skipped_sidecar_count"] for item in validation
    )
    manifest["accepted_shift_count"] = sum(
        item["accepted_shift_count"] for item in validation
    )
    atomic_json(manifest, args.output.resolve())
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-root", type=Path, required=True)
    parser.add_argument("--base-manifest", type=Path)
    parser.add_argument("--sidecar-root", action="append", type=Path, default=[])
    parser.add_argument("--name", default="spectra_da_calibration_sidecars_v2")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    result = build(parse_args())
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "name",
                    "task_count",
                    "sidecar_count",
                    "accepted_sidecar_count",
                    "skipped_sidecar_count",
                    "accepted_shift_count",
                    "target_label_access_count",
                    "protocol_violation_count",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
