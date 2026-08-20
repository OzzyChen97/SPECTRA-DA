#!/usr/bin/env python3
"""Freeze one pre-registered reliable-grid selector into a submission root."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from selector.reliable_selection import atomic_json, choose, load_selection  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_grid_manifest(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != 1:
        raise ValueError(f"unsupported reliable grid manifest schema: {path}")
    selectors = document.get("selectors")
    if not isinstance(selectors, list) or not selectors:
        raise ValueError(f"reliable grid manifest has no selectors: {path}")
    return document


def validate_grid_selector(manifest: dict[str, Any], selector: str) -> dict[str, Any]:
    matches = [entry for entry in manifest["selectors"] if entry.get("selector") == selector]
    if len(matches) != 1:
        raise ValueError(f"selector is not uniquely registered in grid manifest: {selector}")
    return matches[0]


def freeze_selector(
    *,
    grid_root: Path,
    output_root: Path,
    selector: str,
    manifest_path: Path,
    tasks: list[str] | None,
    force: bool,
) -> dict[str, Any]:
    if output_root.exists():
        if not force:
            raise FileExistsError(f"refusing to overwrite frozen selector root: {output_root}")
        if not output_root.is_dir():
            raise FileExistsError(f"output path exists but is not a directory: {output_root}")
    else:
        output_root.mkdir(parents=True)

    manifest = load_grid_manifest(manifest_path)
    selector_config = validate_grid_selector(manifest, selector)
    task_names = tasks or list(manifest.get("tasks") or [])
    if not task_names:
        task_names = sorted(path.name for path in grid_root.iterdir() if path.is_dir())
    if not task_names:
        raise FileNotFoundError(f"no tasks to freeze under {grid_root}")

    frozen_files: dict[str, str] = {}
    candidate_banks: dict[str, str] = {}
    for task in task_names:
        source = grid_root / task / f"{selector}.json"
        if not source.is_file():
            raise FileNotFoundError(f"missing reliable selector output: {source}")
        document = load_selection(source)
        if document.get("task") != task:
            raise ValueError(f"task mismatch in selector output: {source}")
        if document.get("selector") != selector:
            raise ValueError(f"selector name mismatch in selector output: {source}")
        scores = document["candidate_scores"]
        if document.get("selected_candidate_id") != choose(scores, document["score_direction"]):
            raise ValueError(f"selector arg-opt mismatch: {source}")

        destination = output_root / task / f"{selector}.json"
        atomic_json(document, destination)
        frozen_files[str(destination.relative_to(output_root))] = sha256_file(destination)
        candidate_banks[task] = str(document["candidate_bank_sha256"])

    freeze_manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "frozen single reliable selector for one-time sealed evaluation",
        "source_grid_root": str(grid_root.resolve()),
        "source_grid_manifest": str(manifest_path.resolve()),
        "source_grid_manifest_sha256": sha256_file(manifest_path),
        "selector": selector,
        "selector_config": selector_config,
        "task_count": len(task_names),
        "tasks": task_names,
        "candidate_bank_sha256": candidate_banks,
        "selection_files": frozen_files,
        "label_access_count": 0,
        "protocol_violation_count": 0,
        "contains_target_labels": False,
    }
    atomic_json(freeze_manifest, output_root / "reliable_freeze_manifest.json")
    return freeze_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-root", type=Path, required=True)
    parser.add_argument("--selector", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--grid-manifest",
        type=Path,
        help="defaults to GRID_ROOT/reliable_grid_manifest.json",
    )
    parser.add_argument("--task", action="append", dest="tasks")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    grid_root = args.grid_root.resolve()
    manifest_path = (args.grid_manifest or grid_root / "reliable_grid_manifest.json").resolve()
    result = freeze_selector(
        grid_root=grid_root,
        output_root=args.output_root.resolve(),
        selector=args.selector,
        manifest_path=manifest_path,
        tasks=args.tasks,
        force=args.force,
    )
    print(
        json.dumps(
            {
                "selector": result["selector"],
                "task_count": result["task_count"],
                "output_root": str(args.output_root.resolve()),
                "manifest": str(args.output_root.resolve() / "reliable_freeze_manifest.json"),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
