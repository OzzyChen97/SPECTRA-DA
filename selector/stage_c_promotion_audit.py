#!/usr/bin/env python3
"""Consolidate Stage-C controls and enforce the registered freeze decision.

The audit is intentionally conservative: missing source-simulated family-out
CVaR evidence fails that guardrail, and any failed guard prevents freezing or
sealed-final evaluation. Inputs are existing open-development reports only.
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

from selector.consensus_selection import atomic_json, reject_forbidden_path  # noqa: E402


def load_protocol_clean(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if int(document.get("label_access_count", 0)) != 0:
        raise ValueError(f"report records label access: {path}")
    if int(document.get("protocol_violation_count", 0)) != 0:
        raise ValueError(f"report records protocol violation: {path}")
    return document


def collect_selectors(reports: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    selectors: dict[str, dict[str, Any]] = {}
    for report in reports:
        for selector, entry in report.get("selectors", {}).items():
            existing = selectors.get(selector)
            if existing is None or len(entry) > len(existing):
                selectors[selector] = entry
    return selectors


def promotion_audit(
    *,
    selectors: dict[str, dict[str, Any]],
    candidate_selector: str,
    transfer_selector: str,
    loto_report: dict[str, Any],
    router_report: dict[str, Any],
    runtime_seconds: float,
    source_sim_cvar_pass: bool | None,
) -> dict[str, Any]:
    candidate = selectors[candidate_selector]
    transfer = selectors[transfer_selector]
    transfer_by_task = {entry["task"]: entry for entry in transfer["tasks"]}
    noninferior = [
        float(entry["normalized_regret"])
        <= float(transfer_by_task[entry["task"]]["normalized_regret"]) + 1.0e-12
        for entry in candidate["tasks"]
    ]
    noninferior_count = sum(noninferior)
    loto_mean = float(loto_report["aggregate"]["mean_validation_normalized_regret"])
    trajectory_recall = float(candidate["mean_oracle_trajectory_recall_at_20pct"])
    checks = {
        "mean_normalized_regret_le_0_10": float(candidate["mean_normalized_regret"])
        <= 0.10,
        "worst_normalized_regret_le_0_20": float(candidate["worst_normalized_regret"])
        <= 0.20,
        "task_noninferiority_at_least_3_of_4": noninferior_count >= 3,
        "loto_mean_normalized_regret_le_0_15": loto_mean <= 0.15,
        "oracle_trajectory_recall_at_20pct_ge_0_75": trajectory_recall >= 0.75,
        "source_sim_family_out_cvar_degradation_le_5pct": source_sim_cvar_pass
        is True,
        "runtime_lt_480_seconds": runtime_seconds < 480.0,
        "label_access_count_zero": True,
        "protocol_violation_count_zero": True,
    }
    freeze_allowed = all(checks.values())
    controls = {
        selector: {
            "mean_normalized_regret": entry.get("mean_normalized_regret"),
            "worst_normalized_regret": entry.get("worst_normalized_regret"),
            "mean_selected_micro_f1": entry.get("mean_selected_micro_f1"),
            "mean_oracle_trajectory_recall_at_20pct": entry.get(
                "mean_oracle_trajectory_recall_at_20pct"
            ),
        }
        for selector, entry in sorted(
            selectors.items(),
            key=lambda item: float(item[1].get("mean_normalized_regret", float("inf"))),
        )
    }
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "candidate_selector": candidate_selector,
        "transfer_selector": transfer_selector,
        "candidate_metrics": {
            "mean_normalized_regret": candidate["mean_normalized_regret"],
            "worst_normalized_regret": candidate["worst_normalized_regret"],
            "mean_selected_micro_f1": candidate["mean_selected_micro_f1"],
            "task_noninferiority_count_vs_transfer": noninferior_count,
            "task_count": len(noninferior),
            "loto_mean_normalized_regret": loto_mean,
            "oracle_trajectory_recall_at_20pct": trajectory_recall,
            "runtime_evidence_seconds": runtime_seconds,
            "source_sim_family_out_cvar_status": (
                "pass" if source_sim_cvar_pass is True else "fail"
                if source_sim_cvar_pass is False
                else "not_evaluated"
            ),
        },
        "promotion_checks": checks,
        "failed_checks": sorted(check for check, passed in checks.items() if not passed),
        "freeze_allowed": freeze_allowed,
        "sealed_final_evaluation_allowed": freeze_allowed,
        "decision": (
            "freeze_one_selector" if freeze_allowed else "stop_no_freeze_no_sealed_evaluation"
        ),
        "router_qualification_pass": bool(
            router_report.get("router_qualification_pass", False)
        ),
        "router_heldout_noninferior_expert_count": router_report.get(
            "heldout_noninferior_expert_count"
        ),
        "controls": controls,
        "label_access_count": 0,
        "protocol_violation_count": 0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objective-report", type=Path, action="append", required=True)
    parser.add_argument("--candidate-selector", default="agreement20_transfer_rerank")
    parser.add_argument("--transfer-selector", default="transfer_score")
    parser.add_argument("--loto-report", type=Path, required=True)
    parser.add_argument("--router-report", type=Path, required=True)
    parser.add_argument("--runtime-seconds", type=float, required=True)
    parser.add_argument("--source-sim-cvar-pass", choices=("pass", "fail"))
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in [
        *args.objective_report,
        args.loto_report,
        args.router_report,
        args.output,
    ]:
        reject_forbidden_path(path)
    objective_reports = [load_protocol_clean(path) for path in args.objective_report]
    loto_report = load_protocol_clean(args.loto_report)
    router_report = load_protocol_clean(args.router_report)
    source_sim_pass = (
        args.source_sim_cvar_pass == "pass"
        if args.source_sim_cvar_pass is not None
        else None
    )
    result = promotion_audit(
        selectors=collect_selectors(objective_reports),
        candidate_selector=args.candidate_selector,
        transfer_selector=args.transfer_selector,
        loto_report=loto_report,
        router_report=router_report,
        runtime_seconds=args.runtime_seconds,
        source_sim_cvar_pass=source_sim_pass,
    )
    atomic_json(result, args.output.resolve())
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "freeze_allowed": result["freeze_allowed"],
                "failed_checks": result["failed_checks"],
                "label_access_count": 0,
                "protocol_violation_count": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
