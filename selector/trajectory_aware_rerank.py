#!/usr/bin/env python3
"""Rerank a candidate shortlist with trajectory-level Transfer Score evidence.

The shortlist remains candidate-level (for example Agreement Reference top
20%).  Within that fixed shortlist, each method/config/seed trajectory is
scored by the mean of its best-k Transfer Score percentile ranks.  Trajectories
with fewer than k shortlisted checkpoints receive worst-rank padding.  The
selector first chooses the best trajectory and then its best Transfer Score
checkpoint.  It reads selector JSON files only and never reads target labels.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from selector.consensus_selection import (
    atomic_json,
    load_task_selection,
    reject_forbidden_path,
    validate_aligned,
)
from selector.reliable_selection import (
    SHORTLIST_EXCLUSION_SCORE,
    choose,
    percentile_ranks,
    top_fraction_candidates,
)
from selector.run_reliable_suite import discover_paired_tasks
from selector.trajectory_shortlist_selection import group_candidates


def trajectory_topk_score(
    candidates: list[str],
    reranker_ranks: dict[str, float],
    *,
    top_k: int,
) -> float:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    values = sorted(float(reranker_ranks[candidate]) for candidate in candidates)
    padded = values[:top_k] + [1.0] * max(0, top_k - len(values))
    return float(sum(padded) / top_k)


def build_trajectory_aware_rerank(
    *,
    task: str,
    shortlist_owner: dict[str, Any],
    reranker: dict[str, Any],
    selector_name: str,
    shortlist_fraction: float,
    top_k: int,
) -> dict[str, Any]:
    if not 0.0 < shortlist_fraction <= 1.0:
        raise ValueError("shortlist_fraction must lie in (0, 1]")
    candidates = validate_aligned([shortlist_owner, reranker])
    owner_ranks = percentile_ranks(
        shortlist_owner["candidate_scores"], shortlist_owner["score_direction"]
    )
    shortlist = top_fraction_candidates(owner_ranks, fraction=shortlist_fraction)
    shortlist_groups = group_candidates(set(shortlist))

    shortlist_reranker_scores = {
        candidate: float(reranker["candidate_scores"][candidate])
        for candidate in shortlist
    }
    reranker_ranks = percentile_ranks(
        shortlist_reranker_scores, reranker["score_direction"]
    )
    trajectory_scores = {
        trajectory: trajectory_topk_score(
            members, reranker_ranks, top_k=top_k
        )
        for trajectory, members in shortlist_groups.items()
    }
    candidate_trajectory = {
        candidate: trajectory
        for trajectory, members in shortlist_groups.items()
        for candidate in members
    }
    selected_trajectory = min(
        trajectory_scores,
        key=lambda trajectory: (trajectory_scores[trajectory], trajectory),
    )
    selected_members = shortlist_groups[selected_trajectory]
    selected = choose(
        {candidate: reranker_ranks[candidate] for candidate in selected_members},
        "minimize",
    )

    fused_scores: dict[str, float] = {}
    for candidate in candidates:
        if candidate not in shortlist:
            fused_scores[candidate] = float(
                SHORTLIST_EXCLUSION_SCORE + owner_ranks[candidate]
            )
        elif candidate in selected_members:
            fused_scores[candidate] = float(reranker_ranks[candidate])
        else:
            fused_scores[candidate] = float(
                2.0 + trajectory_scores[candidate_trajectory[candidate]]
            )
    if choose(fused_scores, "minimize") != selected:
        raise AssertionError("encoded ranking does not preserve hierarchical choice")

    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task": task,
        "selector": selector_name,
        "candidate_bank_sha256": reranker["candidate_bank_sha256"],
        "candidate_count": reranker["candidate_count"],
        "candidate_scores": fused_scores,
        "score_direction": "minimize",
        "score_semantics": "hierarchical_trajectory_then_checkpoint_ranking",
        "selected_candidate_id": selected,
        "label_access_count": 0,
        "protocol_violation_count": 0,
        "fusion_config": {
            "fusion_mode": "trajectory_aware_shortlist_rerank",
            "shortlist_owner": shortlist_owner["selector"],
            "shortlist_fraction": float(shortlist_fraction),
            "shortlist_size": len(shortlist),
            "shortlist_exclusion_score": SHORTLIST_EXCLUSION_SCORE,
            "represented_trajectory_count": len(shortlist_groups),
            "trajectory_score": "mean best-k shortlist reranker percentile ranks",
            "missing_checkpoint_rank": 1.0,
            "top_k": int(top_k),
            "selected_trajectory": selected_trajectory,
            "rerank_selector": reranker["selector"],
            "checkpoint_rule": "best shortlist reranker percentile rank",
        },
        "trajectory_scores": trajectory_scores,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shortlist-root", type=Path, required=True)
    parser.add_argument("--shortlist-selector", required=True)
    parser.add_argument("--rerank-root", type=Path, required=True)
    parser.add_argument("--rerank-selector", default="transfer_score")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-selector", required=True)
    parser.add_argument("--shortlist-fraction", type=float, default=0.20)
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument("--task", action="append", dest="tasks")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    shortlist_root = args.shortlist_root.resolve()
    rerank_root = args.rerank_root.resolve()
    output_root = args.output_root.resolve()
    for root in (shortlist_root, rerank_root, output_root):
        reject_forbidden_path(root)
    tasks = args.tasks or discover_paired_tasks(
        rerank_root,
        shortlist_root,
        args.rerank_selector,
        args.shortlist_selector,
    )
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selector": args.output_selector,
        "tasks": tasks,
        "shortlist_root": str(shortlist_root),
        "shortlist_selector": args.shortlist_selector,
        "shortlist_fraction": float(args.shortlist_fraction),
        "rerank_root": str(rerank_root),
        "rerank_selector": args.rerank_selector,
        "top_k": int(args.top_k),
        "label_access_count": 0,
        "protocol_violation_count": 0,
        "contains_target_labels": False,
    }
    for task in tasks:
        shortlist_owner = load_task_selection(
            shortlist_root, task, args.shortlist_selector
        )
        reranker = load_task_selection(rerank_root, task, args.rerank_selector)
        result = build_trajectory_aware_rerank(
            task=task,
            shortlist_owner=shortlist_owner,
            reranker=reranker,
            selector_name=args.output_selector,
            shortlist_fraction=args.shortlist_fraction,
            top_k=args.top_k,
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
