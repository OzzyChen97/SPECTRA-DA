from __future__ import annotations

from selector.stage_c_promotion_audit import promotion_audit


def _entry(mean: float, worst: float, task_values: list[float], trajectory_recall: float) -> dict:
    return {
        "mean_normalized_regret": mean,
        "worst_normalized_regret": worst,
        "mean_selected_micro_f1": 0.7,
        "mean_oracle_trajectory_recall_at_20pct": trajectory_recall,
        "tasks": [
            {"task": f"t{index}", "normalized_regret": value}
            for index, value in enumerate(task_values)
        ],
    }


def test_missing_source_sim_evidence_prevents_freeze() -> None:
    selectors = {
        "candidate": _entry(0.08, 0.19, [0.1, 0.1, 0.1, 0.0], 1.0),
        "transfer": _entry(0.2, 0.4, [0.2, 0.2, 0.2, 0.2], 0.5),
    }
    result = promotion_audit(
        selectors=selectors,
        candidate_selector="candidate",
        transfer_selector="transfer",
        loto_report={"aggregate": {"mean_validation_normalized_regret": 0.1}},
        router_report={"router_qualification_pass": False},
        runtime_seconds=100.0,
        source_sim_cvar_pass=None,
    )
    assert result["freeze_allowed"] is False
    assert "source_sim_family_out_cvar_degradation_le_5pct" in result["failed_checks"]


def test_all_registered_checks_are_required() -> None:
    selectors = {
        "candidate": _entry(0.08, 0.19, [0.1, 0.1, 0.1, 0.0], 1.0),
        "transfer": _entry(0.2, 0.4, [0.2, 0.2, 0.2, 0.2], 0.5),
    }
    result = promotion_audit(
        selectors=selectors,
        candidate_selector="candidate",
        transfer_selector="transfer",
        loto_report={"aggregate": {"mean_validation_normalized_regret": 0.1}},
        router_report={"router_qualification_pass": True},
        runtime_seconds=100.0,
        source_sim_cvar_pass=True,
    )
    assert result["freeze_allowed"] is True
    assert result["sealed_final_evaluation_allowed"] is True
