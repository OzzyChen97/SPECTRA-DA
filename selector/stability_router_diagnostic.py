#!/usr/bin/env python3
"""Audit the pre-registered parameter-free stability router on open development.

The rule chooses Agreement@20% -> Transfer Score only when its bootstrapped
trajectory-set shortlist Jaccard is strictly higher than Transfer Score's;
otherwise it chooses Transfer Score. Because the rule has no fitted parameter,
its per-task held-out evaluation is equivalent to leave-one-task-out use. The
tool consumes only selector/stability JSON and the exported open-development
truth artifact, never raw target labels.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from metrics import normalized_regret  # noqa: E402
from selector.consensus_selection import atomic_json, reject_forbidden_path  # noqa: E402
from selector.objective_v2 import (  # noqa: E402
    OPEN_DEVELOPMENT_TASKS,
    load_open_dev_truth,
    load_selection,
)

STABILITY_KEY = "mean_trajectory_shortlist_jaccard"


def _load_json(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if int(document.get("label_access_count", 0)) != 0:
        raise ValueError(f"diagnostic reports label access: {path}")
    if int(document.get("protocol_violation_count", 0)) != 0:
        raise ValueError(f"diagnostic reports protocol violation: {path}")
    return document


def audit_task(
    *,
    task: str,
    agreement_stability: dict[str, Any],
    transfer_stability: dict[str, Any],
    agreement_selection: dict[str, Any],
    transfer_selection: dict[str, Any],
    truth: dict[str, Any],
) -> dict[str, Any]:
    agreement_value = float(
        agreement_stability["bootstrap_diagnostics"][STABILITY_KEY]
    )
    transfer_value = float(
        transfer_stability["bootstrap_diagnostics"][STABILITY_KEY]
    )
    chosen_expert = (
        "agreement20_transfer_rerank"
        if agreement_value > transfer_value
        else "transfer_score"
    )
    selections = {
        "agreement20_transfer_rerank": agreement_selection,
        "transfer_score": transfer_selection,
    }
    risks = {candidate: float(value) for candidate, value in truth["risks"].items()}
    selected_candidates = {
        expert: str(selection["selected_candidate_id"])
        for expert, selection in selections.items()
    }
    expert_risks = {
        expert: risks[candidate]
        for expert, candidate in selected_candidates.items()
    }
    chosen_candidate = selected_candidates[chosen_expert]
    chosen_risk = expert_risks[chosen_expert]
    other_expert = (
        "transfer_score"
        if chosen_expert == "agreement20_transfer_rerank"
        else "agreement20_transfer_rerank"
    )
    correct = chosen_risk <= expert_risks[other_expert] + 1.0e-12
    return {
        "task": task,
        "agreement_trajectory_stability": agreement_value,
        "transfer_trajectory_stability": transfer_value,
        "chosen_expert": chosen_expert,
        "chosen_candidate_id": chosen_candidate,
        "chosen_target_error": chosen_risk,
        "normalized_regret": normalized_regret(chosen_risk, np.asarray(list(risks.values()))),
        "agreement_selected_candidate_id": selected_candidates[
            "agreement20_transfer_rerank"
        ],
        "agreement_selected_target_error": expert_risks[
            "agreement20_transfer_rerank"
        ],
        "transfer_selected_candidate_id": selected_candidates["transfer_score"],
        "transfer_selected_target_error": expert_risks["transfer_score"],
        "chose_noninferior_expert": bool(correct),
    }


def aggregate_task_audits(reports: list[dict[str, Any]]) -> dict[str, Any]:
    correct_count = sum(bool(report["chose_noninferior_expert"]) for report in reports)
    regrets = [float(report["normalized_regret"]) for report in reports]
    qualification = correct_count >= 3
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "router_rule": (
            "choose Agreement@20%->TS iff its mean bootstrapped trajectory-set "
            "shortlist Jaccard is strictly greater than Transfer Score's"
        ),
        "stability_metric": STABILITY_KEY,
        "parameter_count": 0,
        "loto_semantics": (
            "parameter-free rule; each task decision is independent of the other "
            "three tasks, so held-out evaluation is LOTO-equivalent"
        ),
        "task_count": len(reports),
        "heldout_noninferior_expert_count": correct_count,
        "heldout_noninferior_expert_rate": correct_count / len(reports),
        "router_qualification_min_count": 3,
        "router_qualification_pass": qualification,
        "router_mean_normalized_regret": float(np.mean(regrets)),
        "router_worst_normalized_regret": float(max(regrets)),
        "router_promoted": False,
        "promotion_status": (
            "eligible_for_objective_evaluation" if qualification else "rejected_before_use"
        ),
        "tasks": reports,
        "label_access_count": 0,
        "protocol_violation_count": 0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-truth-report", type=Path, required=True)
    parser.add_argument("--agreement-stability-root", type=Path, required=True)
    parser.add_argument("--transfer-stability-root", type=Path, required=True)
    parser.add_argument("--agreement-selection-root", type=Path, required=True)
    parser.add_argument("--agreement-selector", default="agreement20_transfer_rerank")
    parser.add_argument("--transfer-selection-root", type=Path, required=True)
    parser.add_argument("--transfer-selector", default="transfer_score")
    parser.add_argument("--task", action="append", dest="tasks")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    roots = (
        args.dev_truth_report,
        args.agreement_stability_root,
        args.transfer_stability_root,
        args.agreement_selection_root,
        args.transfer_selection_root,
        args.output,
    )
    for root in roots:
        reject_forbidden_path(root)
    tasks = args.tasks or list(OPEN_DEVELOPMENT_TASKS)
    truth = load_open_dev_truth(args.dev_truth_report.resolve())
    reports = []
    for task in tasks:
        agreement_stability = _load_json(
            args.agreement_stability_root
            / task
            / "stable_agreement20_transfer_rerank.json"
        )
        transfer_stability = _load_json(
            args.transfer_stability_root
            / task
            / "transfer_score_bootstrap_stability.json"
        )
        agreement_selection = load_selection(
            args.agreement_selection_root
            / task
            / f"{args.agreement_selector}.json"
        )
        transfer_selection = load_selection(
            args.transfer_selection_root
            / task
            / f"{args.transfer_selector}.json"
        )
        reports.append(
            audit_task(
                task=task,
                agreement_stability=agreement_stability,
                transfer_stability=transfer_stability,
                agreement_selection=agreement_selection,
                transfer_selection=transfer_selection,
                truth=truth[task],
            )
        )
    result = aggregate_task_audits(reports)
    atomic_json(result, args.output.resolve())
    print(
        json.dumps(
            {
                "router_qualification_pass": result["router_qualification_pass"],
                "heldout_noninferior_expert_count": result[
                    "heldout_noninferior_expert_count"
                ],
                "label_access_count": 0,
                "protocol_violation_count": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
