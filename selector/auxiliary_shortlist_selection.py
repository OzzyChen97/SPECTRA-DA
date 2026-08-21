#!/usr/bin/env python3
"""Build fixed-budget shortlists from a registered auxiliary selector signal.

Stage-C uses this tool for exactly three controls: a fixed covariance-gamma
shortlist, an Agreement/gamma midrank shortlist, and a fixed-budget union of
their top-10% candidates. The union is capped or filled by the two-signal mean
percentile midrank so every control exposes the same number of candidates to
the unchanged Transfer Score reranker.
"""

from __future__ import annotations

import argparse
import json
import math
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
from selector.coverage_floor_selection import ranked_candidates  # noqa: E402
from selector.reliable_selection import (  # noqa: E402
    SHORTLIST_EXCLUSION_SCORE,
    choose,
    percentile_ranks,
)
from selector.run_reliable_suite import discover_paired_tasks  # noqa: E402

MODES = ("single", "midrank", "union")


def fixed_auxiliary_shortlist(
    candidates: set[str],
    owner_ranks: list[dict[str, float]],
    *,
    mode: str,
    budget: int,
    owner_fraction: float,
) -> tuple[set[str], dict[str, Any]]:
    if mode not in MODES:
        raise ValueError(f"unknown auxiliary shortlist mode: {mode}")
    if not owner_ranks:
        raise ValueError("at least one owner ranking is required")
    if mode == "single" and len(owner_ranks) != 1:
        raise ValueError("single mode requires exactly one owner")
    if mode in {"midrank", "union"} and len(owner_ranks) != 2:
        raise ValueError(f"{mode} mode requires exactly two owners")
    if not 0 < budget <= len(candidates):
        raise ValueError("budget must lie in [1, candidate_count]")
    if not 0.0 < owner_fraction <= 1.0:
        raise ValueError("owner_fraction must lie in (0, 1]")

    consensus = {
        candidate: float(
            sum(ranks[candidate] for ranks in owner_ranks) / len(owner_ranks)
        )
        for candidate in candidates
    }
    if mode in {"single", "midrank"}:
        shortlist = set(ranked_candidates(candidates, consensus)[:budget])
        return shortlist, {
            "component_shortlist_size": None,
            "raw_union_size": None,
            "budget_adjustment": "fixed consensus top-budget",
        }

    component_count = max(1, math.ceil(owner_fraction * len(candidates)))
    component_sets = [
        set(ranked_candidates(candidates, ranks)[:component_count])
        for ranks in owner_ranks
    ]
    raw_union = set.union(*component_sets)
    shortlist = set(ranked_candidates(raw_union, consensus)[:budget])
    if len(shortlist) < budget:
        for candidate in ranked_candidates(candidates, consensus):
            if len(shortlist) >= budget:
                break
            shortlist.add(candidate)
    if len(shortlist) != budget:
        raise AssertionError("fixed auxiliary shortlist budget was not met")
    adjustment = (
        "midrank_cap" if len(raw_union) > budget else "midrank_fill"
        if len(raw_union) < budget
        else "none"
    )
    return shortlist, {
        "component_shortlist_size": component_count,
        "component_shortlist_sizes": [len(values) for values in component_sets],
        "raw_union_size": len(raw_union),
        "budget_adjustment": adjustment,
    }


def build_auxiliary_shortlist_selection(
    *,
    task: str,
    owners: list[dict[str, Any]],
    reranker: dict[str, Any],
    selector_name: str,
    mode: str,
    budget: int,
    owner_fraction: float,
) -> dict[str, Any]:
    candidates = validate_aligned([*owners, reranker])
    owner_ranks = [
        percentile_ranks(owner["candidate_scores"], owner["score_direction"])
        for owner in owners
    ]
    shortlist, construction = fixed_auxiliary_shortlist(
        candidates,
        owner_ranks,
        mode=mode,
        budget=budget,
        owner_fraction=owner_fraction,
    )
    reranker_ranks = percentile_ranks(
        reranker["candidate_scores"], reranker["score_direction"]
    )
    consensus = {
        candidate: float(
            sum(ranks[candidate] for ranks in owner_ranks) / len(owner_ranks)
        )
        for candidate in candidates
    }
    fused_scores = {
        candidate: float(
            reranker_ranks[candidate]
            if candidate in shortlist
            else SHORTLIST_EXCLUSION_SCORE + consensus[candidate]
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
        "score_semantics": "fixed_budget_auxiliary_shortlist_then_rerank",
        "selected_candidate_id": selected,
        "label_access_count": 0,
        "protocol_violation_count": 0,
        "fusion_config": {
            "fusion_mode": "fixed_auxiliary_shortlist_rerank",
            "auxiliary_mode": mode,
            "shortlist_owners": [owner["selector"] for owner in owners],
            "shortlist_size": len(shortlist),
            "shortlist_exclusion_score": SHORTLIST_EXCLUSION_SCORE,
            "budget": int(budget),
            "owner_fraction": float(owner_fraction),
            "owner_rank_fusion": "tie-aware percentile midrank mean",
            "rerank_selector": reranker["selector"],
            **construction,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner-root", type=Path, action="append", required=True)
    parser.add_argument("--owner-selector", action="append", required=True)
    parser.add_argument("--rerank-root", type=Path, required=True)
    parser.add_argument("--rerank-selector", default="transfer_score")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-selector", required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--budget", type=int, default=135)
    parser.add_argument("--owner-fraction", type=float, default=0.10)
    parser.add_argument("--task", action="append", dest="tasks")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.owner_root) != len(args.owner_selector):
        raise ValueError("--owner-root and --owner-selector must have the same count")
    owner_roots = [root.resolve() for root in args.owner_root]
    rerank_root = args.rerank_root.resolve()
    output_root = args.output_root.resolve()
    for root in (*owner_roots, rerank_root, output_root):
        reject_forbidden_path(root)
    tasks = args.tasks or discover_paired_tasks(
        rerank_root,
        owner_roots[0],
        args.rerank_selector,
        args.owner_selector[0],
    )
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selector": args.output_selector,
        "tasks": tasks,
        "owner_roots": [str(root) for root in owner_roots],
        "owner_selectors": args.owner_selector,
        "rerank_root": str(rerank_root),
        "rerank_selector": args.rerank_selector,
        "mode": args.mode,
        "budget": int(args.budget),
        "owner_fraction": float(args.owner_fraction),
        "label_access_count": 0,
        "protocol_violation_count": 0,
        "contains_target_labels": False,
    }
    for task in tasks:
        owners = [
            load_task_selection(root, task, selector)
            for root, selector in zip(owner_roots, args.owner_selector)
        ]
        reranker = load_task_selection(rerank_root, task, args.rerank_selector)
        result = build_auxiliary_shortlist_selection(
            task=task,
            owners=owners,
            reranker=reranker,
            selector_name=args.output_selector,
            mode=args.mode,
            budget=args.budget,
            owner_fraction=args.owner_fraction,
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
