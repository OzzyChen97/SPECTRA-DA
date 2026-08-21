#!/usr/bin/env python3
"""Build fixed-budget candidate shortlists with group coverage floors.

The selector preserves candidate-level screening while reserving a minimum
number of slots for every method/config/seed trajectory, or an equal quota for
every method. Remaining trajectory-floor slots are filled by the global
shortlist-owner ranking. A separate label-free selector then reranks the fixed
shortlist. Only selector JSON files are read; target labels are never loaded.
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

from selector.consensus_selection import (  # noqa: E402
    atomic_json,
    load_task_selection,
    reject_forbidden_path,
    validate_aligned,
)
from selector.reliable_selection import (  # noqa: E402
    SHORTLIST_EXCLUSION_SCORE,
    choose,
    percentile_ranks,
)
from selector.run_reliable_suite import discover_paired_tasks  # noqa: E402
from selector.trajectory_shortlist_selection import (  # noqa: E402
    group_candidates,
    trajectory_key,
)

MODES = ("trajectory", "method")


def method_key(candidate_id: str) -> str:
    parts = str(candidate_id).split("__")
    if len(parts) < 2:
        raise ValueError(f"candidate id has no method field: {candidate_id}")
    return parts[1]


def ranked_candidates(
    candidates: set[str] | list[str], ranks: dict[str, float]
) -> list[str]:
    return sorted(candidates, key=lambda candidate: (float(ranks[candidate]), candidate))


def build_fixed_budget_shortlist(
    candidates: set[str],
    owner_ranks: dict[str, float],
    *,
    budget: int,
    mode: str,
    group_quota: int,
) -> tuple[set[str], dict[str, list[str]]]:
    if mode not in MODES:
        raise ValueError(f"unknown coverage-floor mode: {mode}")
    if not 0 < budget <= len(candidates):
        raise ValueError("budget must lie in [1, candidate_count]")
    if group_quota <= 0:
        raise ValueError("group_quota must be positive")

    if mode == "trajectory":
        groups = group_candidates(candidates)
    else:
        groups: dict[str, list[str]] = {}
        for candidate in sorted(candidates):
            groups.setdefault(method_key(candidate), []).append(candidate)

    guaranteed: dict[str, list[str]] = {}
    shortlist: set[str] = set()
    for group, members in sorted(groups.items()):
        if len(members) < group_quota:
            raise ValueError(
                f"group {group} has {len(members)} candidates, below quota {group_quota}"
            )
        selected = ranked_candidates(members, owner_ranks)[:group_quota]
        guaranteed[group] = selected
        shortlist.update(selected)
    if len(shortlist) > budget:
        raise ValueError(
            f"coverage floor requires {len(shortlist)} slots, above budget {budget}"
        )

    for candidate in ranked_candidates(candidates, owner_ranks):
        if len(shortlist) >= budget:
            break
        shortlist.add(candidate)
    if len(shortlist) != budget:
        raise AssertionError("fixed-budget shortlist construction failed")
    return shortlist, guaranteed


def build_coverage_floor_selection(
    *,
    task: str,
    shortlist_owner: dict[str, Any],
    reranker: dict[str, Any],
    selector_name: str,
    budget: int,
    mode: str,
    group_quota: int,
) -> dict[str, Any]:
    candidates = validate_aligned([shortlist_owner, reranker])
    owner_ranks = percentile_ranks(
        shortlist_owner["candidate_scores"], shortlist_owner["score_direction"]
    )
    shortlist, guaranteed = build_fixed_budget_shortlist(
        candidates,
        owner_ranks,
        budget=budget,
        mode=mode,
        group_quota=group_quota,
    )
    reranker_ranks = percentile_ranks(
        reranker["candidate_scores"], reranker["score_direction"]
    )
    fused_scores = {
        candidate: float(
            reranker_ranks[candidate]
            if candidate in shortlist
            else SHORTLIST_EXCLUSION_SCORE + owner_ranks[candidate]
        )
        for candidate in candidates
    }
    selected = choose(fused_scores, "minimize")
    if selected not in shortlist:
        raise AssertionError("coverage-floor reranker selected outside shortlist")

    represented_trajectories = {trajectory_key(candidate) for candidate in shortlist}
    represented_methods = {method_key(candidate) for candidate in shortlist}
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task": task,
        "selector": selector_name,
        "candidate_bank_sha256": reranker["candidate_bank_sha256"],
        "candidate_count": reranker["candidate_count"],
        "candidate_scores": fused_scores,
        "score_direction": "minimize",
        "score_semantics": "fixed_budget_group_coverage_then_rerank",
        "selected_candidate_id": selected,
        "label_access_count": 0,
        "protocol_violation_count": 0,
        "fusion_config": {
            "fusion_mode": "coverage_floor_shortlist_rerank",
            "shortlist_owner": shortlist_owner["selector"],
            "shortlist_size": len(shortlist),
            "shortlist_exclusion_score": SHORTLIST_EXCLUSION_SCORE,
            "budget": int(budget),
            "coverage_mode": mode,
            "group_quota": int(group_quota),
            "group_count": len(guaranteed),
            "guaranteed_candidate_count": sum(len(values) for values in guaranteed.values()),
            "global_fill_count": len(shortlist) - sum(
                len(values) for values in guaranteed.values()
            ),
            "represented_trajectory_count": len(represented_trajectories),
            "represented_method_count": len(represented_methods),
            "rerank_selector": reranker["selector"],
            "rerank_rule": "tie-aware candidate percentile rank",
        },
        "guaranteed_candidates": guaranteed,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shortlist-root", type=Path, required=True)
    parser.add_argument("--shortlist-selector", required=True)
    parser.add_argument("--rerank-root", type=Path, required=True)
    parser.add_argument("--rerank-selector", default="transfer_score")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-selector", required=True)
    parser.add_argument("--budget", type=int, default=135)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--group-quota", type=int, required=True)
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
        "rerank_root": str(rerank_root),
        "rerank_selector": args.rerank_selector,
        "budget": int(args.budget),
        "mode": args.mode,
        "group_quota": int(args.group_quota),
        "label_access_count": 0,
        "protocol_violation_count": 0,
        "contains_target_labels": False,
    }
    for task in tasks:
        shortlist_owner = load_task_selection(
            shortlist_root, task, args.shortlist_selector
        )
        reranker = load_task_selection(rerank_root, task, args.rerank_selector)
        result = build_coverage_floor_selection(
            task=task,
            shortlist_owner=shortlist_owner,
            reranker=reranker,
            selector_name=args.output_selector,
            budget=args.budget,
            mode=args.mode,
            group_quota=args.group_quota,
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
