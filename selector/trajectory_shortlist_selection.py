#!/usr/bin/env python3
"""Build trajectory-level shortlist selectors from label-free rankings.

Candidates are grouped by ``(method, config, seed)``.  Each shortlist owner
scores a trajectory by the mean of its best-q checkpoint percentile ranks.
When multiple owners are supplied, their trajectory percentile ranks are
averaged.  The top trajectory fraction is expanded back to checkpoints and a
label-free reranker selects the final candidate.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from selector.consensus_selection import reject_forbidden_path, validate_aligned  # noqa: E402
from selector.reliable_selection import (  # noqa: E402
    SHORTLIST_EXCLUSION_SCORE,
    choose,
    load_selection,
    percentile_ranks,
    top_fraction_candidates,
)
from selector.run_reliable_suite import discover_paired_tasks  # noqa: E402


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


def trajectory_key(candidate_id: str) -> str:
    parts = str(candidate_id).split("__")
    if len(parts) < 4:
        raise ValueError(f"candidate id has no method/config fields: {candidate_id}")
    seed = next((part for part in parts if part.startswith("seed-")), None)
    if seed is None:
        raise ValueError(f"candidate id has no seed field: {candidate_id}")
    return "__".join((parts[1], parts[2], seed))


def group_candidates(candidates: set[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for candidate in sorted(candidates):
        groups.setdefault(trajectory_key(candidate), []).append(candidate)
    return groups


def owner_trajectory_ranks(
    owner: dict[str, Any],
    groups: dict[str, list[str]],
    *,
    best_q: int,
) -> tuple[dict[str, float], dict[str, float]]:
    if best_q <= 0:
        raise ValueError("best_q must be positive")
    candidate_ranks = percentile_ranks(
        owner["candidate_scores"],
        owner["score_direction"],
    )
    raw_scores = {
        trajectory: float(
            sum(sorted(candidate_ranks[candidate] for candidate in candidates)[:best_q])
            / min(best_q, len(candidates))
        )
        for trajectory, candidates in groups.items()
    }
    return raw_scores, percentile_ranks(raw_scores, "minimize")


def build_trajectory_shortlist(
    *,
    task: str,
    shortlist_owners: list[dict[str, Any]],
    reranker: dict[str, Any],
    selector_name: str,
    trajectory_fraction: float,
    best_q: int,
) -> dict[str, Any]:
    if not 0.0 < trajectory_fraction <= 1.0:
        raise ValueError("trajectory_fraction must lie in (0, 1]")
    if not shortlist_owners:
        raise ValueError("at least one shortlist owner is required")
    candidates = validate_aligned([*shortlist_owners, reranker])
    groups = group_candidates(candidates)
    owner_scores_and_ranks = [
        owner_trajectory_ranks(owner, groups, best_q=best_q)
        for owner in shortlist_owners
    ]
    consensus_trajectory_ranks = {
        trajectory: float(
            sum(ranks[trajectory] for _, ranks in owner_scores_and_ranks)
            / len(owner_scores_and_ranks)
        )
        for trajectory in groups
    }
    shortlist_trajectories = top_fraction_candidates(
        consensus_trajectory_ranks,
        fraction=trajectory_fraction,
    )
    shortlist_candidates = {
        candidate
        for trajectory in shortlist_trajectories
        for candidate in groups[trajectory]
    }
    reranker_ranks = percentile_ranks(
        reranker["candidate_scores"],
        reranker["score_direction"],
    )
    fused_scores = {
        candidate: float(
            reranker_ranks[candidate]
            if candidate in shortlist_candidates
            else SHORTLIST_EXCLUSION_SCORE
            + consensus_trajectory_ranks[trajectory_key(candidate)]
        )
        for candidate in candidates
    }
    selected = choose(fused_scores, "minimize")
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task": task,
        "selector": selector_name,
        "candidate_bank_sha256": reranker["candidate_bank_sha256"],
        "candidate_count": reranker["candidate_count"],
        "candidate_scores": fused_scores,
        "score_direction": "minimize",
        "score_semantics": "ranking",
        "selected_candidate_id": selected,
        "label_access_count": 0,
        "protocol_violation_count": 0,
        "fusion_config": {
            "fusion_mode": "trajectory_shortlist_consensus_rerank",
            "shortlist_owners": [owner["selector"] for owner in shortlist_owners],
            "trajectory_score": "mean best-q checkpoint percentile ranks",
            "best_q": int(best_q),
            "trajectory_fraction": float(trajectory_fraction),
            "trajectory_count": len(groups),
            "shortlist_trajectory_count": len(shortlist_trajectories),
            "shortlist_candidate_count": len(shortlist_candidates),
            "shortlist_trajectories": sorted(shortlist_trajectories),
            "rerank_selector": reranker["selector"],
            "rerank_rule": "tie-aware candidate percentile rank",
            "shortlist_exclusion_score": SHORTLIST_EXCLUSION_SCORE,
        },
        "trajectory_scores": {
            trajectory: {
                "consensus_percentile_rank": consensus_trajectory_ranks[trajectory],
                "owner_best_q_scores": {
                    owner["selector"]: owner_scores_and_ranks[index][0][trajectory]
                    for index, owner in enumerate(shortlist_owners)
                },
                "owner_percentile_ranks": {
                    owner["selector"]: owner_scores_and_ranks[index][1][trajectory]
                    for index, owner in enumerate(shortlist_owners)
                },
            }
            for trajectory in sorted(groups)
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shortlist-root", type=Path, action="append", required=True)
    parser.add_argument("--shortlist-selector", action="append", required=True)
    parser.add_argument("--rerank-root", type=Path, required=True)
    parser.add_argument("--rerank-selector", default="transfer_score")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-selector", required=True)
    parser.add_argument("--trajectory-fraction", type=float, default=0.20)
    parser.add_argument("--best-q", type=int, default=3)
    parser.add_argument("--task", action="append", dest="tasks")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.shortlist_root) != len(args.shortlist_selector):
        raise ValueError("--shortlist-root and --shortlist-selector must have the same count")
    shortlist_roots = [root.resolve() for root in args.shortlist_root]
    rerank_root = args.rerank_root.resolve()
    output_root = args.output_root.resolve()
    for root in [*shortlist_roots, rerank_root, output_root]:
        reject_forbidden_path(root)
    tasks = args.tasks or discover_paired_tasks(
        rerank_root,
        shortlist_roots[0],
        args.rerank_selector,
        args.shortlist_selector[0],
    )
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selector": args.output_selector,
        "tasks": tasks,
        "shortlist_roots": [str(root) for root in shortlist_roots],
        "shortlist_selectors": args.shortlist_selector,
        "rerank_root": str(rerank_root),
        "rerank_selector": args.rerank_selector,
        "trajectory_fraction": float(args.trajectory_fraction),
        "best_q": int(args.best_q),
        "label_access_count": 0,
        "protocol_violation_count": 0,
        "contains_target_labels": False,
    }
    for task in tasks:
        owners = [
            load_selection(root / task / f"{selector}.json")
            for root, selector in zip(shortlist_roots, args.shortlist_selector)
        ]
        reranker = load_selection(
            rerank_root / task / f"{args.rerank_selector}.json"
        )
        result = build_trajectory_shortlist(
            task=task,
            shortlist_owners=owners,
            reranker=reranker,
            selector_name=args.output_selector,
            trajectory_fraction=args.trajectory_fraction,
            best_q=args.best_q,
        )
        atomic_json(result, output_root / task / f"{args.output_selector}.json")
    atomic_json(manifest, output_root / f"{args.output_selector}_manifest.json")
    print(
        json.dumps(
            {
                "task_count": len(tasks),
                "selector": args.output_selector,
                "label_access_count": 0,
                "protocol_violation_count": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
