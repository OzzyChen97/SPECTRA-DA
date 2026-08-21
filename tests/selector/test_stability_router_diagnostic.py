from __future__ import annotations

from selector.stability_router_diagnostic import aggregate_task_audits


def test_router_requires_three_of_four_noninferior_expert_choices() -> None:
    reports = [
        {
            "normalized_regret": value,
            "chose_noninferior_expert": correct,
        }
        for value, correct in (
            (0.1, True),
            (0.2, True),
            (0.3, False),
            (0.0, True),
        )
    ]

    result = aggregate_task_audits(reports)

    assert result["heldout_noninferior_expert_count"] == 3
    assert result["router_qualification_pass"] is True
    assert result["router_promoted"] is False


def test_router_rejects_two_of_four() -> None:
    reports = [
        {"normalized_regret": 0.1, "chose_noninferior_expert": index < 2}
        for index in range(4)
    ]
    result = aggregate_task_audits(reports)
    assert result["router_qualification_pass"] is False
    assert result["promotion_status"] == "rejected_before_use"
