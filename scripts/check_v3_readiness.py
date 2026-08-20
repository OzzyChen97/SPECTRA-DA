#!/usr/bin/env python3
"""Audit readiness for v3 open-development and sealed-package evaluation.

This script is deliberately label-safe. It reads only exported open-development
truth reports, selector JSON files, and submission manifests. It refuses sealed
or final-label paths and does not import the sealed evaluator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from protocol.tasks import PILOT_TASKS, TASKS  # noqa: E402


OPEN_DEVELOPMENT_TASKS = tuple(task.id for task in PILOT_TASKS)
FINAL_TASKS = tuple(task.id for task in TASKS if task.id not in set(OPEN_DEVELOPMENT_TASKS))
FORBIDDEN_PATH_PARTS = {".sealed", "sealed_eval", "final_12_labels"}
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


def path_is_forbidden(path: Path) -> bool:
    return bool(set(path.parts) & FORBIDDEN_PATH_PARTS)


def read_json(path: Path) -> dict[str, Any]:
    if path_is_forbidden(path):
        raise RuntimeError(f"refusing forbidden path: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _ok_report(name: str, **fields: Any) -> dict[str, Any]:
    return {"name": name, "ok": True, "status": "ready", "errors": [], **fields}


def _missing_report(name: str, path: Path, **fields: Any) -> dict[str, Any]:
    return {
        "name": name,
        "ok": False,
        "status": "missing",
        "path": str(path),
        "errors": [f"missing {path}"],
        **fields,
    }


def _failure_report(name: str, errors: list[str], **fields: Any) -> dict[str, Any]:
    return {"name": name, "ok": False, "status": "failed", "errors": errors, **fields}


def audit_open_dev_truth(path: Path, expected_candidate_count: int) -> dict[str, Any]:
    name = "open_dev_truth"
    if not path.exists():
        return _missing_report(name, path)
    try:
        document = read_json(path)
    except Exception as exc:
        return _failure_report(name, [str(exc)], path=str(path))

    errors: list[str] = []
    tasks = [str(task.get("task")) for task in document.get("tasks", [])]
    expected_tasks = set(OPEN_DEVELOPMENT_TASKS)
    observed_tasks = set(tasks)
    missing_tasks = sorted(expected_tasks - observed_tasks)
    extra_tasks = sorted(observed_tasks - expected_tasks)
    final_leakage = sorted(observed_tasks & set(FINAL_TASKS))
    if missing_tasks:
        errors.append(f"missing open-development tasks: {missing_tasks}")
    if extra_tasks:
        errors.append(f"unexpected non-open-development tasks: {extra_tasks}")
    if final_leakage:
        errors.append(f"final sealed tasks exposed in open-dev truth: {final_leakage}")
    if document.get("final_sealed_tasks_exposed") != 0:
        errors.append(
            "final_sealed_tasks_exposed must be 0, observed "
            f"{document.get('final_sealed_tasks_exposed')}"
        )
    if document.get("label_access_count") != 0:
        errors.append(f"label_access_count must be 0, observed {document.get('label_access_count')}")
    if document.get("protocol_violation_count") != 0:
        errors.append(
            "protocol_violation_count must be 0, observed "
            f"{document.get('protocol_violation_count')}"
        )

    task_counts: dict[str, int] = {}
    for task_report in document.get("tasks", []):
        task = str(task_report.get("task"))
        count = int(task_report.get("candidate_count", -1))
        truth = task_report.get("candidate_truth")
        task_counts[task] = count
        if count != expected_candidate_count:
            errors.append(
                f"{task} candidate_count={count}, expected {expected_candidate_count}"
            )
        if not isinstance(truth, dict) or len(truth) != count:
            observed = len(truth) if isinstance(truth, dict) else "non-dict"
            errors.append(f"{task} candidate_truth size={observed}, candidate_count={count}")

    fields = {
        "path": str(path),
        "expected_tasks": list(OPEN_DEVELOPMENT_TASKS),
        "observed_tasks": tasks,
        "expected_candidate_count": expected_candidate_count,
        "candidate_count_per_task": task_counts,
        "final_sealed_tasks_exposed": document.get("final_sealed_tasks_exposed"),
    }
    if errors:
        return _failure_report(name, errors, **fields)
    return _ok_report(name, **fields)


def _load_selection(path: Path) -> dict[str, Any]:
    document = read_json(path)
    forbidden = sorted(_walk_keys(document) & FORBIDDEN_RESULT_KEYS)
    if forbidden:
        raise RuntimeError(f"{path} exposes forbidden target-truth keys: {forbidden}")
    return document


def _argopt(scores: dict[str, Any], direction: str) -> str:
    numeric_scores = {candidate: float(value) for candidate, value in scores.items()}
    if not all(math.isfinite(value) for value in numeric_scores.values()):
        raise ValueError("candidate_scores contains non-finite values")
    if direction == "minimize":
        optimum = min(numeric_scores.values())
    elif direction == "maximize":
        optimum = max(numeric_scores.values())
    else:
        raise ValueError(f"unknown score_direction: {direction}")
    return min(candidate for candidate, value in numeric_scores.items() if value == optimum)


def _selection_paths_for_task(selection_roots: list[Path], task: str) -> list[Path]:
    paths: list[Path] = []
    for root in selection_roots:
        task_root = root / task
        if task_root.is_dir():
            paths.extend(sorted(task_root.glob("*.json")))
    return paths


def audit_selector_roots(
    selection_roots: list[Path],
    *,
    tasks: tuple[str, ...],
    expected_candidate_count: int,
    required_selectors: list[str],
    name: str,
) -> dict[str, Any]:
    missing_roots = [str(root) for root in selection_roots if not root.is_dir()]
    if missing_roots:
        return _failure_report(name, [f"missing selection roots: {missing_roots}"])

    errors: list[str] = []
    task_reports: dict[str, Any] = {}
    for task in tasks:
        paths = _selection_paths_for_task(selection_roots, task)
        by_selector: dict[str, dict[str, Any]] = {}
        bank_hashes: set[str] = set()
        counts: set[int] = set()
        task_errors: list[str] = []
        for path in paths:
            try:
                document = _load_selection(path)
                selector = str(document.get("selector"))
                if selector in by_selector:
                    task_errors.append(f"duplicate selector {selector}")
                    continue
                scores = document.get("candidate_scores")
                count = int(document.get("candidate_count", -1))
                if document.get("task") != task:
                    task_errors.append(f"{path} task={document.get('task')} expected {task}")
                if count != expected_candidate_count:
                    task_errors.append(
                        f"{selector} candidate_count={count}, expected {expected_candidate_count}"
                    )
                if not isinstance(scores, dict) or len(scores) != count:
                    observed = len(scores) if isinstance(scores, dict) else "non-dict"
                    task_errors.append(f"{selector} score_count={observed}, candidate_count={count}")
                elif document.get("selected_candidate_id") != _argopt(
                    scores,
                    str(document.get("score_direction")),
                ):
                    task_errors.append(f"{selector} selected_candidate_id is not the arg-opt")
                if document.get("label_access_count") != 0:
                    task_errors.append(f"{selector} label_access_count={document.get('label_access_count')}")
                if document.get("protocol_violation_count") != 0:
                    task_errors.append(
                        f"{selector} protocol_violation_count={document.get('protocol_violation_count')}"
                    )
                if not isinstance(document.get("candidate_bank_sha256"), str):
                    task_errors.append(f"{selector} missing candidate_bank_sha256")
                else:
                    bank_hashes.add(str(document["candidate_bank_sha256"]))
                counts.add(count)
                by_selector[selector] = {"path": str(path), "candidate_count": count}
            except Exception as exc:
                task_errors.append(str(exc))
        missing_selectors = sorted(set(required_selectors) - set(by_selector))
        if missing_selectors:
            task_errors.append(f"missing required selectors: {missing_selectors}")
        if len(bank_hashes) > 1:
            task_errors.append(f"selectors disagree on candidate_bank_sha256: {sorted(bank_hashes)}")
        if task_errors:
            errors.extend(f"{task}: {error}" for error in task_errors)
        task_reports[task] = {
            "selector_count": len(by_selector),
            "selectors": sorted(by_selector),
            "candidate_counts": sorted(counts),
            "candidate_bank_sha256_count": len(bank_hashes),
            "errors": task_errors,
        }

    fields = {
        "selection_roots": [str(root) for root in selection_roots],
        "task_count": len(tasks),
        "expected_candidate_count": expected_candidate_count,
        "required_selectors": required_selectors,
        "tasks": task_reports,
    }
    if errors:
        return _failure_report(name, errors, **fields)
    return _ok_report(name, **fields)


def audit_final_submission_manifest(
    path: Path,
    *,
    expected_candidate_count: int,
    min_selector_count: int,
    required_selectors: list[str],
) -> dict[str, Any]:
    name = "final_submission_manifest"
    if not path.exists():
        return _missing_report(name, path)
    try:
        document = read_json(path)
    except Exception as exc:
        return _failure_report(name, [str(exc)], path=str(path))

    errors: list[str] = []
    selectors = list(document.get("selectors", []))
    selector_set = set(selectors)
    if len(selector_set) != len(selectors):
        errors.append(f"selectors contains duplicates: {selectors}")
    if int(document.get("task_count", -1)) != len(TASKS):
        errors.append(f"task_count={document.get('task_count')}, expected {len(TASKS)}")
    if int(document.get("candidate_count_per_task", -1)) != expected_candidate_count:
        errors.append(
            "candidate_count_per_task="
            f"{document.get('candidate_count_per_task')}, expected {expected_candidate_count}"
        )
    if int(document.get("selector_count", -1)) < min_selector_count:
        errors.append(
            f"selector_count={document.get('selector_count')}, expected at least {min_selector_count}"
        )
    if int(document.get("selector_count", -1)) != len(selectors):
        errors.append(
            f"selector_count={document.get('selector_count')} does not match "
            f"len(selectors)={len(selectors)}"
        )
    missing_selectors = sorted(set(required_selectors) - set(selectors))
    if missing_selectors:
        errors.append(f"missing required final selectors: {missing_selectors}")
    if document.get("target_label_access_count") != 0:
        errors.append(
            "target_label_access_count must be 0, observed "
            f"{document.get('target_label_access_count')}"
        )
    if document.get("protocol_violation_count") != 0:
        errors.append(
            "protocol_violation_count must be 0, observed "
            f"{document.get('protocol_violation_count')}"
        )
    if document.get("contains_target_labels") is not False:
        errors.append(f"contains_target_labels must be false, observed {document.get('contains_target_labels')}")

    selection_files = document.get("selection_files")
    expected_selection_files = sorted(
        f"selections/{task.id}/{selector}.json"
        for task in TASKS
        for selector in selectors
    )
    observed_selection_files = sorted(selection_files) if isinstance(selection_files, dict) else []
    if not isinstance(selection_files, dict):
        errors.append("selection_files must be a dictionary of packaged selector-file hashes")
    else:
        missing_selection_files = sorted(set(expected_selection_files) - set(observed_selection_files))
        unexpected_selection_files = sorted(set(observed_selection_files) - set(expected_selection_files))
        if missing_selection_files:
            errors.append(f"missing packaged selection files: {missing_selection_files[:10]}")
        if unexpected_selection_files:
            errors.append(f"unexpected packaged selection files: {unexpected_selection_files[:10]}")
        for relative in expected_selection_files:
            expected_hash = selection_files.get(relative)
            if not isinstance(expected_hash, str):
                errors.append(f"{relative} missing sha256 entry")
                continue
            selection_path = path.parent / relative
            if not selection_path.is_file():
                errors.append(f"missing packaged selection file: {selection_path}")
                continue
            actual_hash = sha256_file(selection_path)
            if actual_hash != expected_hash:
                errors.append(
                    f"selection file hash mismatch for {relative}: "
                    f"manifest={expected_hash}, actual={actual_hash}"
                )

    fields = {
        "path": str(path),
        "expected_task_count": len(TASKS),
        "expected_candidate_count": expected_candidate_count,
        "min_selector_count": min_selector_count,
        "required_selectors": required_selectors,
        "observed_selector_count": document.get("selector_count"),
        "observed_selectors": selectors,
        "expected_selection_file_count": len(expected_selection_files),
        "observed_selection_file_count": len(observed_selection_files),
    }
    if errors:
        return _failure_report(name, errors, **fields)
    return _ok_report(name, **fields)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--open-dev-truth",
        type=Path,
        default=Path("results/gda_select/open_dev/gate1_candidate_truth.json"),
    )
    parser.add_argument(
        "--open-dev-selection-root",
        type=Path,
        action="append",
        default=[],
        help="Root containing open-development task/selector.json files; may be repeated.",
    )
    parser.add_argument(
        "--final-submission-manifest",
        type=Path,
        default=Path("results/gda_select/submissions/final_multi_selector/submission_manifest.json"),
    )
    parser.add_argument("--expected-open-dev-candidates", type=int, default=675)
    parser.add_argument("--expected-final-candidates", type=int, default=675)
    parser.add_argument(
        "--required-open-dev-selector",
        action="append",
        default=None,
    )
    parser.add_argument(
        "--required-final-selector",
        action="append",
        default=None,
    )
    parser.add_argument("--min-final-selector-count", type=int, default=2)
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    required_open_dev_selectors = args.required_open_dev_selector or [
        "spectra_trust",
        "transfer_score",
    ]
    required_final_selectors = args.required_final_selector or ["transfer_score"]
    checks = [
        audit_open_dev_truth(
            args.open_dev_truth.resolve(),
            args.expected_open_dev_candidates,
        ),
    ]
    if args.open_dev_selection_root:
        checks.append(
            audit_selector_roots(
                [root.resolve() for root in args.open_dev_selection_root],
                tasks=OPEN_DEVELOPMENT_TASKS,
                expected_candidate_count=args.expected_open_dev_candidates,
                required_selectors=required_open_dev_selectors,
                name="open_dev_selector_roots",
            )
        )
    else:
        checks.append(
            _missing_report(
                "open_dev_selector_roots",
                Path("<pass --open-dev-selection-root>"),
                required_selectors=required_open_dev_selectors,
            )
        )
    checks.append(
        audit_final_submission_manifest(
            args.final_submission_manifest.resolve(),
            expected_candidate_count=args.expected_final_candidates,
            min_selector_count=args.min_final_selector_count,
            required_selectors=required_final_selectors,
        )
    )
    failures = [check for check in checks if not check["ok"]]
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "v3 open-development and sealed-package readiness audit",
        "ok": not failures,
        "failure_count": len(failures),
        "checks": checks,
        "label_access_count": 0,
        "protocol_violation_count": 0,
    }


def main() -> None:
    result = run(parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
