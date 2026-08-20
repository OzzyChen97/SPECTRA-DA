#!/usr/bin/env python3
"""Generate label-free baseline selections for every task in a candidate bank."""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.trajectory_export.schema import (  # noqa: E402
    atomic_json,
    candidate_bank_hash,
    discover_candidate_records,
)
from selector.baselines import (  # noqa: E402
    SELECTORS,
    _agreement_on_the_line,
    build_selection_result,
    choose,
)


def is_valid_cached_selection(
    path: Path,
    *,
    task: str,
    selector: str,
    direction: str,
    candidate_ids: set[str],
    bank_hash: str,
) -> bool:
    """Return whether an existing result is safe to reuse verbatim."""

    if not path.is_file():
        return False
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
        scores = result["candidate_scores"]
        if not isinstance(scores, dict) or set(scores) != candidate_ids:
            return False
        numeric_scores = {identifier: float(value) for identifier, value in scores.items()}
        if not all(math.isfinite(value) for value in numeric_scores.values()):
            return False
        expected_semantics = "estimated_error" if selector in {"dev", "gde"} else "ranking"
        return bool(
            result.get("schema_version") == 1
            and result.get("task") == task
            and result.get("selector") == selector
            and result.get("candidate_bank_sha256") == bank_hash
            and result.get("candidate_count") == len(candidate_ids)
            and result.get("score_direction") == direction
            and result.get("score_semantics") == expected_semantics
            and result.get("selected_candidate_id") == choose(numeric_scores, direction)
            and result.get("label_access_count") == 0
            and result.get("protocol_violation_count") == 0
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError, OSError):
        return False


def discover_tasks(candidate_root: Path) -> list[str]:
    tasks = sorted(
        path.name
        for path in candidate_root.iterdir()
        if path.is_dir() and not path.name.startswith("_")
    )
    if not tasks:
        raise FileNotFoundError(f"no task directories under {candidate_root}")
    return tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--task", action="append", dest="tasks")
    parser.add_argument(
        "--selector",
        action="append",
        dest="selectors",
        choices=sorted(SELECTORS),
        help="repeat to run a subset; the default runs every registered baseline",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidate_root = args.candidate_root.resolve()
    output_root = args.output_root.resolve()
    tasks = args.tasks or discover_tasks(candidate_root)
    selectors = args.selectors or sorted(SELECTORS)
    for task in tasks:
        lock_root = output_root.parent / f".{output_root.name}.locks"
        lock_root.mkdir(parents=True, exist_ok=True)
        lock_path = lock_root / f"{task}.lock"
        with lock_path.open("a+", encoding="utf-8") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            records = discover_candidate_records(candidate_root, task)
            bank_hash = candidate_bank_hash(records)
            candidate_ids = {record["metadata"]["candidate_id"] for record in records}
            shared_scores: dict[str, dict[str, float]] = {}
            for selector in selectors:
                function, direction = SELECTORS[selector]
                output = output_root / task / f"{selector}.json"
                if is_valid_cached_selection(
                    output,
                    task=task,
                    selector=selector,
                    direction=direction,
                    candidate_ids=candidate_ids,
                    bank_hash=bank_hash,
                ):
                    print(
                        f"[baseline] task={task} selector={selector} reuse=validated",
                        flush=True,
                    )
                    continue
                print(f"[baseline] task={task} selector={selector}", flush=True)
                scores = shared_scores.get(selector)
                if scores is None and selector in {"aol_s", "aol_d"}:
                    aline_s, aline_d = _agreement_on_the_line(records)
                    shared_scores.update({"aol_s": aline_s, "aol_d": aline_d})
                    scores = shared_scores[selector]
                if scores is None:
                    scores = function(records)
                result = build_selection_result(
                    records,
                    task,
                    selector,
                    scores,
                    direction,
                    bank_hash=bank_hash,
                )
                atomic_json(result, output)
                print(
                    json.dumps(
                        {
                            "task": task,
                            "selector": selector,
                            "candidate_count": result["candidate_count"],
                            "selected_candidate_id": result["selected_candidate_id"],
                            "output": str(output),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    main()
