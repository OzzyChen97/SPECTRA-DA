#!/usr/bin/env python3
"""Validate and package frozen label-free selections for an external evaluator."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from protocol.tasks import TASKS  # noqa: E402
from scripts.trajectory_export.schema import (  # noqa: E402
    REQUIRED_METADATA_KEYS,
    atomic_json,
    candidate_bank_hash,
    discover_candidate_records,
    sha256_file,
)


FORBIDDEN_RESULT_KEYS = {
    "candidate_truth",
    "oracle_candidate_index",
    "oracle_f1",
    "target_f1",
    "target_labels",
    "target_macro_f1",
    "target_micro_f1",
    "true_risk",
    "y_target",
}


def redact_local_paths(value: Any) -> Any:
    """Return a JSON-safe copy with machine-local absolute paths redacted."""

    if isinstance(value, dict):
        return {key: redact_local_paths(child) for key, child in value.items()}
    if isinstance(value, list):
        return [redact_local_paths(child) for child in value]
    if isinstance(value, str) and (
        value.startswith("/mnt/")
        or value.startswith("/tmp/")
        or value.startswith(str(REPO.parent))
    ):
        return "<redacted-local-path>"
    return value


def discover_candidate_metadata_records(
    output_root: Path,
    task: str,
) -> list[dict[str, Any]]:
    """Discover candidates using metadata and artifact presence only."""

    task_root = output_root / task
    records: list[dict[str, Any]] = []
    if not task_root.is_dir():
        raise FileNotFoundError(f"candidate task directory does not exist: {task_root}")
    for metadata_path in sorted(task_root.glob("**/checkpoint_*/metadata.json")):
        directory = metadata_path.parent
        for artifact in (
            metadata_path,
            directory / "source_val.npz",
            directory / "target_public.npz",
            directory / "model_state.pt",
        ):
            if not artifact.is_file():
                raise FileNotFoundError(f"missing trajectory artifact: {artifact}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        missing = sorted(REQUIRED_METADATA_KEYS - set(metadata))
        if missing:
            raise ValueError(
                f"checkpoint metadata is missing required fields: {metadata_path}: {missing}"
            )
        if metadata.get("schema_version") != 1:
            raise ValueError(f"unsupported schema version: {metadata_path}")
        if metadata.get("task") != task:
            raise ValueError(f"candidate task mismatch: {directory}")
        if metadata.get("target_label_access_count") != 0:
            raise RuntimeError(f"candidate reports target-label access: {directory}")
        if metadata.get("target_public_has_labels") is not False:
            raise RuntimeError(f"candidate target artifact is not label-free: {directory}")
        records.append({"path": directory, "metadata": metadata})
    if not records:
        raise ValueError(f"no candidate checkpoints found under {task_root}")
    identifiers = [record["metadata"]["candidate_id"] for record in records]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError(f"duplicate candidate identifiers under {task_root}")
    return records


def _walk_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        keys = {str(key).lower() for key in value}
        for child in value.values():
            keys.update(_walk_keys(child))
        return keys
    if isinstance(value, list):
        keys: set[str] = set()
        for child in value:
            keys.update(_walk_keys(child))
        return keys
    return set()


def validate_selection(
    document: dict[str, Any],
    *,
    task: str,
    bank_hash: str,
    candidate_ids: list[str],
) -> str:
    if document.get("task") != task:
        raise ValueError(f"selection task mismatch: expected {task}")
    selector = document.get("selector")
    if not isinstance(selector, str) or not selector:
        raise ValueError(f"selection has no selector name: {task}")
    if document.get("candidate_bank_sha256") != bank_hash:
        raise ValueError(f"selection candidate-bank mismatch: {task}/{selector}")
    if document.get("candidate_count") != len(candidate_ids):
        raise ValueError(f"selection candidate-count mismatch: {task}/{selector}")
    if document.get("label_access_count") != 0:
        raise RuntimeError(f"selection reports label access: {task}/{selector}")
    if document.get("protocol_violation_count") != 0:
        raise RuntimeError(f"selection reports a protocol violation: {task}/{selector}")
    forbidden = sorted(_walk_keys(document) & FORBIDDEN_RESULT_KEYS)
    if forbidden:
        raise RuntimeError(
            f"selection exposes forbidden target-truth keys: {task}/{selector}: {forbidden}"
        )
    scores = document.get("candidate_scores")
    if not isinstance(scores, dict) or set(scores) != set(candidate_ids):
        raise ValueError(f"selection score coverage mismatch: {task}/{selector}")
    numeric_scores = {name: float(value) for name, value in scores.items()}
    if not all(math.isfinite(value) for value in numeric_scores.values()):
        raise ValueError(f"selection contains a non-finite score: {task}/{selector}")
    direction = document.get("score_direction")
    if direction == "minimize":
        optimum = min(numeric_scores.values())
    elif direction == "maximize":
        optimum = max(numeric_scores.values())
    else:
        raise ValueError(f"unknown score direction: {task}/{selector}")
    expected = min(
        name for name, value in numeric_scores.items() if value == optimum
    )
    if document.get("selected_candidate_id") != expected:
        raise ValueError(f"selection arg-opt mismatch: {task}/{selector}")
    return selector


def package(args: argparse.Namespace) -> dict[str, Any]:
    candidate_root = args.candidate_root.resolve()
    selection_root = args.selection_root.resolve()
    output_root = args.output_root.resolve()
    archive = args.archive.resolve() if args.archive is not None else None
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite package: {output_root}")
    if archive is not None and archive.exists():
        raise FileExistsError(f"refusing to overwrite archive: {archive}")
    temporary = output_root.parent / f".{output_root.name}.tmp-{os.getpid()}"
    if temporary.exists():
        raise FileExistsError(f"temporary package already exists: {temporary}")
    temporary.mkdir(parents=True)

    expected_tasks = [task.id for task in TASKS]
    file_manifest: dict[str, str] = {}
    bank_manifest: dict[str, str] = {}
    selector_set: set[str] | None = None
    try:
        for task in expected_tasks:
            if args.metadata_only_candidate_check:
                records = discover_candidate_metadata_records(candidate_root, task)
            else:
                records = discover_candidate_records(
                    candidate_root,
                    task,
                    verify_hashes=not args.trust_candidate_metadata_hashes,
                )
            if len(records) != args.expected_candidates_per_task:
                raise ValueError(
                    f"candidate count mismatch for {task}: "
                    f"expected {args.expected_candidates_per_task}, observed {len(records)}"
                )
            candidate_ids = [record["metadata"]["candidate_id"] for record in records]
            bank_hash = candidate_bank_hash(records)
            bank_manifest[task] = bank_hash
            source_directory = selection_root / task
            paths = sorted(source_directory.glob("*.json"))
            if not paths:
                raise FileNotFoundError(f"no selections for task {task}")
            if args.selector:
                allowed = set(args.selector)
                filtered_paths = []
                observed = set()
                for path in paths:
                    selector = json.loads(path.read_text(encoding="utf-8")).get("selector")
                    if selector in allowed:
                        filtered_paths.append(path)
                        observed.add(selector)
                paths = filtered_paths
                missing = sorted(allowed - observed)
                if missing:
                    raise FileNotFoundError(
                        f"missing requested selectors for {task}: {missing}"
                    )
            destination = temporary / "selections" / task
            destination.mkdir(parents=True)
            task_selectors: set[str] = set()
            for path in paths:
                document = json.loads(path.read_text(encoding="utf-8"))
                selector = validate_selection(
                    document,
                    task=task,
                    bank_hash=bank_hash,
                    candidate_ids=candidate_ids,
                )
                if selector in task_selectors:
                    raise ValueError(f"duplicate selector for {task}: {selector}")
                task_selectors.add(selector)
                target = destination / f"{selector}.json"
                atomic_json(redact_local_paths(document), target)
                relative = str(target.relative_to(temporary))
                file_manifest[relative] = sha256_file(target)
            if selector_set is None:
                selector_set = task_selectors
            elif task_selectors != selector_set:
                raise ValueError(
                    f"selector coverage differs for {task}: "
                    f"expected={sorted(selector_set)}, observed={sorted(task_selectors)}"
                )

        manifest = {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "purpose": "one-time external sealed evaluation submission",
            "task_count": len(expected_tasks),
            "candidate_count_per_task": args.expected_candidates_per_task,
            "selector_count": len(selector_set or ()),
            "selectors": sorted(selector_set or ()),
            "candidate_bank_sha256": bank_manifest,
            "selection_files": file_manifest,
            "target_label_access_count": 0,
            "protocol_violation_count": 0,
            "contains_candidate_artifacts": False,
            "contains_target_labels": False,
            "local_paths_redacted": True,
            "candidate_validation_mode": (
                "metadata_only"
                if args.metadata_only_candidate_check
                else (
                    "schema_with_metadata_hashes"
                    if args.trust_candidate_metadata_hashes
                    else "strict_schema_and_artifact_hashes"
                )
            ),
            "evaluator_contract": (
                "The receiving service must bind these hashes to its own frozen "
                "candidate bank, evaluate once, retain candidate-level truth "
                "privately, and return aggregate reports only."
            ),
        }
        atomic_json(manifest, temporary / "submission_manifest.json")
        temporary.replace(output_root)
    except BaseException:
        # Preserve an interrupted staging directory for audit/recovery rather
        # than deleting evidence or partially publishing a final package.
        raise

    if archive is not None:
        archive.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive, mode="w:gz") as handle:
            handle.add(output_root, arcname=output_root.name)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--selection-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--expected-candidates-per-task", type=int, default=675)
    parser.add_argument(
        "--trust-candidate-metadata-hashes",
        action="store_true",
        help=(
            "Use candidate metadata artifact hashes to bind the bank without "
            "rehashing every large checkpoint file. The default strict mode "
            "rehashes all artifacts."
        ),
    )
    parser.add_argument(
        "--metadata-only-candidate-check",
        action="store_true",
        help=(
            "Only validate candidate metadata, artifact presence, bank hashes, "
            "and selection coverage. This is intended for fast submission "
            "packaging after a separate trajectory export audit."
        ),
    )
    parser.add_argument(
        "--selector",
        action="append",
        help=(
            "Selector name to include. May be repeated. "
            "By default, every selector JSON found for each task is packaged."
        ),
    )
    return parser.parse_args()


def main() -> None:
    result = package(parse_args())
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "purpose",
                    "task_count",
                    "candidate_count_per_task",
                    "selector_count",
                    "selectors",
                    "target_label_access_count",
                    "protocol_violation_count",
                    "contains_target_labels",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
