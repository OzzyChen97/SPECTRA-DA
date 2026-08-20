#!/usr/bin/env python3
"""Preflight audit for reliability-aware selector fusion inputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from selector.reliable_selection import load_selection, validate_pair  # noqa: E402


def _available_tasks(selection_root: Path) -> list[str]:
    return sorted(path.name for path in selection_root.iterdir() if path.is_dir())


def _audit_task(
    spectra_root: Path,
    transfer_root: Path,
    task: str,
    spectra_selector: str,
    transfer_selector: str,
    require_uncertainty: bool,
) -> dict[str, Any]:
    spectra_path = spectra_root / task / f"{spectra_selector}.json"
    transfer_path = transfer_root / task / f"{transfer_selector}.json"
    errors: list[str] = []
    if not spectra_path.is_file():
        errors.append(f"missing {spectra_path}")
    if not transfer_path.is_file():
        errors.append(f"missing {transfer_path}")
    if errors:
        return {"task": task, "ok": False, "errors": errors}

    try:
        spectra = load_selection(spectra_path)
        transfer = load_selection(transfer_path)
        candidates = validate_pair(spectra, transfer)
        uncertainty_source = (
            "candidate_transport_uncertainty"
            if "candidate_transport_uncertainty" in spectra
            else "candidate_uncertainty"
            if "candidate_uncertainty" in spectra
            else "none"
        )
        if require_uncertainty and uncertainty_source == "none":
            raise ValueError("missing candidate uncertainty required by --require-uncertainty")
        return {
            "task": task,
            "ok": True,
            "candidate_count": len(candidates),
            "candidate_bank_sha256": spectra["candidate_bank_sha256"],
            "spectra_selector": spectra.get("selector"),
            "transfer_selector": transfer.get("selector"),
            "uncertainty_source": uncertainty_source,
            "label_access_count": 0,
            "protocol_violation_count": 0,
        }
    except Exception as exc:
        return {"task": task, "ok": False, "errors": [str(exc)]}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-root", type=Path, required=True)
    parser.add_argument(
        "--spectra-root",
        type=Path,
        help="optional root for SPECTRA JSON files; defaults to --selection-root",
    )
    parser.add_argument(
        "--transfer-root",
        type=Path,
        help="optional root for Transfer Score JSON files; defaults to --selection-root",
    )
    parser.add_argument("--spectra-selector", default="spectra_robust")
    parser.add_argument("--transfer-selector", default="transfer_score")
    parser.add_argument("--task", action="append", dest="tasks")
    parser.add_argument("--require-uncertainty", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selection_root = args.selection_root.resolve()
    spectra_root = (args.spectra_root or selection_root).resolve()
    transfer_root = (args.transfer_root or selection_root).resolve()
    if not spectra_root.is_dir():
        raise FileNotFoundError(f"spectra root does not exist: {spectra_root}")
    if not transfer_root.is_dir():
        raise FileNotFoundError(f"transfer root does not exist: {transfer_root}")
    tasks = args.tasks or _available_tasks(spectra_root)
    if not tasks:
        raise FileNotFoundError(f"no task directories under {selection_root}")

    task_reports = [
        _audit_task(
            spectra_root,
            transfer_root,
            task,
            args.spectra_selector,
            args.transfer_selector,
            args.require_uncertainty,
        )
        for task in tasks
    ]
    failures = [report for report in task_reports if not report["ok"]]
    summary = {
        "selection_root": str(selection_root),
        "spectra_root": str(spectra_root),
        "transfer_root": str(transfer_root),
        "spectra_selector": args.spectra_selector,
        "transfer_selector": args.transfer_selector,
        "task_count": len(task_reports),
        "ok_task_count": len(task_reports) - len(failures),
        "failure_count": len(failures),
        "label_free_post_selector_ready": not failures,
        "tasks": task_reports,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
