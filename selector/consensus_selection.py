#!/usr/bin/env python3
"""Build explicit label-free shortlist/consensus selectors.

This script is a small semantic wrapper around the v3 reliable-selection
primitives.  It is intended for the compact Stage-B method set:

* Transfer Score top-k -> Agreement Reference rerank;
* Transfer Score top-k -> SPECTRA/Agreement midrank consensus.

It reads only precomputed selector JSON files for the same candidate bank and
does not read candidate artifacts, target labels, ``sealed_eval``, or final
label paths.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from selector.reliable_selection import (  # noqa: E402
    SHORTLIST_EXCLUSION_SCORE,
    choose,
    load_selection,
    percentile_ranks,
    top_fraction_candidates,
)
from selector.run_reliable_suite import discover_paired_tasks  # noqa: E402

FORBIDDEN_PATH_PARTS = {".sealed", "sealed_eval", "final_12_labels"}


def reject_forbidden_path(path: Path) -> None:
    forbidden = sorted(set(path.resolve().parts) & FORBIDDEN_PATH_PARTS)
    if forbidden:
        raise RuntimeError(f"forbidden path components: {forbidden}")


def atomic_json(document: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(document, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def validate_aligned(selections: list[dict[str, Any]]) -> set[str]:
    if not selections:
        raise ValueError("at least one selection is required")
    tasks = {selection.get("task") for selection in selections}
    if len(tasks) != 1:
        raise ValueError(f"selection tasks do not align: {tasks}")
    hashes = {selection.get("candidate_bank_sha256") for selection in selections}
    if len(hashes) != 1:
        raise ValueError(f"candidate banks do not align: {hashes}")
    counts = {selection.get("candidate_count") for selection in selections}
    if len(counts) != 1:
        raise ValueError(f"candidate counts do not align: {counts}")
    candidate_sets = [set(selection["candidate_scores"]) for selection in selections]
    if len({frozenset(candidates) for candidates in candidate_sets}) != 1:
        raise ValueError("selection candidate coverage does not align")
    return candidate_sets[0]


def build_consensus(
    *,
    task: str,
    shortlist_owner: dict[str, Any],
    rerankers: list[dict[str, Any]],
    selector_name: str,
    shortlist_fraction: float,
) -> dict[str, Any]:
    if not 0.0 < shortlist_fraction <= 1.0:
        raise ValueError("shortlist_fraction must lie in (0, 1]")
    candidates = validate_aligned([shortlist_owner, *rerankers])
    owner_rank = percentile_ranks(
        shortlist_owner["candidate_scores"],
        shortlist_owner["score_direction"],
    )
    shortlist = top_fraction_candidates(owner_rank, fraction=shortlist_fraction)
    reranker_ranks = [
        percentile_ranks(reranker["candidate_scores"], reranker["score_direction"])
        for reranker in rerankers
    ]
    fused_scores = {}
    for candidate in candidates:
        if candidate in shortlist:
            fused_scores[candidate] = float(
                sum(ranks[candidate] for ranks in reranker_ranks) / len(reranker_ranks)
            )
        else:
            fused_scores[candidate] = float(SHORTLIST_EXCLUSION_SCORE + owner_rank[candidate])
    selected = choose(fused_scores, "minimize")
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task": task,
        "selector": selector_name,
        "candidate_bank_sha256": shortlist_owner["candidate_bank_sha256"],
        "candidate_count": shortlist_owner["candidate_count"],
        "candidate_scores": fused_scores,
        "score_direction": "minimize",
        "score_semantics": "ranking",
        "selected_candidate_id": selected,
        "label_access_count": 0,
        "protocol_violation_count": 0,
        "fusion_config": {
            "fusion_mode": "transfer_shortlist_consensus_rerank",
            "shortlist_owner": shortlist_owner["selector"],
            "shortlist_fraction": float(shortlist_fraction),
            "shortlist_size": len(shortlist),
            "shortlist_exclusion_score": SHORTLIST_EXCLUSION_SCORE,
            "rerank_selectors": [reranker["selector"] for reranker in rerankers],
            "rerank_rule": "tie-aware percentile midrank mean",
        },
        "component_selected_candidate_id": {
            "shortlist_owner": shortlist_owner["selected_candidate_id"],
            **{
                reranker["selector"]: reranker["selected_candidate_id"]
                for reranker in rerankers
            },
        },
    }


def discover_tasks_from_owner(
    owner_root: Path,
    first_reranker_root: Path,
    owner_selector: str,
    first_reranker_selector: str,
) -> list[str]:
    return discover_paired_tasks(
        first_reranker_root,
        owner_root,
        first_reranker_selector,
        owner_selector,
    )


def load_task_selection(root: Path, task: str, selector: str) -> dict[str, Any]:
    reject_forbidden_path(root)
    path = root / task / f"{selector}.json"
    return load_selection(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shortlist-root", type=Path, required=True)
    parser.add_argument("--shortlist-selector", default="transfer_score")
    parser.add_argument("--rerank-root", type=Path, action="append", required=True)
    parser.add_argument("--rerank-selector", action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-selector", required=True)
    parser.add_argument("--shortlist-fraction", type=float, default=0.20)
    parser.add_argument("--task", action="append", dest="tasks")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.rerank_root) != len(args.rerank_selector):
        raise ValueError("--rerank-root and --rerank-selector must have the same count")
    shortlist_root = args.shortlist_root.resolve()
    rerank_roots = [path.resolve() for path in args.rerank_root]
    output_root = args.output_root.resolve()
    tasks = args.tasks or discover_tasks_from_owner(
        shortlist_root,
        rerank_roots[0],
        args.shortlist_selector,
        args.rerank_selector[0],
    )
    manifest = {
        "schema_version": 1,
        "selector": args.output_selector,
        "tasks": tasks,
        "shortlist_root": str(shortlist_root),
        "shortlist_selector": args.shortlist_selector,
        "shortlist_fraction": float(args.shortlist_fraction),
        "rerank_roots": [str(path) for path in rerank_roots],
        "rerank_selectors": args.rerank_selector,
        "fusion_mode": "transfer_shortlist_consensus_rerank",
        "label_access_count": 0,
        "protocol_violation_count": 0,
        "contains_target_labels": False,
    }
    for task in tasks:
        shortlist_owner = load_task_selection(shortlist_root, task, args.shortlist_selector)
        rerankers = [
            load_task_selection(root, task, selector)
            for root, selector in zip(rerank_roots, args.rerank_selector)
        ]
        result = build_consensus(
            task=task,
            shortlist_owner=shortlist_owner,
            rerankers=rerankers,
            selector_name=args.output_selector,
            shortlist_fraction=args.shortlist_fraction,
        )
        atomic_json(result, output_root / task / f"{args.output_selector}.json")
    atomic_json(manifest, output_root / f"{args.output_selector}_manifest.json")
    print(
        json.dumps(
            {
                "task_count": len(tasks),
                "selector": args.output_selector,
                "output_root": str(output_root),
                "label_access_count": 0,
                "protocol_violation_count": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
