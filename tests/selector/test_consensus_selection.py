from __future__ import annotations

import json
from pathlib import Path

from selector.consensus_selection import build_consensus, load_task_selection


def _selection(
    selector: str,
    scores: dict[str, float],
    direction: str = "minimize",
) -> dict:
    selected = min(scores, key=scores.get) if direction == "minimize" else max(scores, key=scores.get)
    return {
        "schema_version": 1,
        "created_at": "2026-08-20T00:00:00+00:00",
        "task": "A_to_B",
        "selector": selector,
        "candidate_bank_sha256": "bank",
        "candidate_count": len(scores),
        "candidate_scores": scores,
        "score_direction": direction,
        "score_semantics": "ranking",
        "selected_candidate_id": selected,
        "label_access_count": 0,
        "protocol_violation_count": 0,
    }


def test_consensus_uses_transfer_shortlist_and_multiple_rerankers() -> None:
    owner = _selection(
        "transfer_score",
        {"transfer_best": 0.9, "inside": 0.8, "outside": 0.1, "bad": 0.0},
        "maximize",
    )
    spectra = _selection(
        "spectra_cal",
        {"transfer_best": 0.8, "inside": 0.1, "outside": 0.0, "bad": 0.9},
    )
    agreement = _selection(
        "agreement_reference",
        {"transfer_best": 0.7, "inside": 0.2, "outside": 0.0, "bad": 0.9},
    )

    result = build_consensus(
        task="A_to_B",
        shortlist_owner=owner,
        rerankers=[spectra, agreement],
        selector_name="ts20_spectra_agreement_consensus",
        shortlist_fraction=0.5,
    )

    assert result["selected_candidate_id"] == "inside"
    assert result["candidate_scores"]["outside"] > 1.0e11
    assert result["fusion_config"]["shortlist_owner"] == "transfer_score"
    assert result["fusion_config"]["rerank_selectors"] == [
        "spectra_cal",
        "agreement_reference",
    ]
    assert result["label_access_count"] == 0
    assert result["protocol_violation_count"] == 0


def test_consensus_single_reranker_matches_shortlist_rerank() -> None:
    owner = _selection(
        "transfer_score",
        {"transfer_best": 0.9, "inside": 0.8, "outside": 0.7, "bad": 0.0},
        "maximize",
    )
    agreement = _selection(
        "agreement_reference",
        {"transfer_best": 0.5, "inside": 0.1, "outside": 0.0, "bad": 0.9},
    )

    result = build_consensus(
        task="A_to_B",
        shortlist_owner=owner,
        rerankers=[agreement],
        selector_name="ts20_agreement_rerank",
        shortlist_fraction=0.5,
    )

    assert result["selected_candidate_id"] == "inside"
    assert result["candidate_scores"]["outside"] > 1.0e11


def test_load_task_selection_rejects_bad_protocol(tmp_path: Path) -> None:
    task_dir = tmp_path / "A_to_B"
    task_dir.mkdir()
    document = _selection("bad", {"a": 0.0})
    document["label_access_count"] = 1
    (task_dir / "bad.json").write_text(json.dumps(document), encoding="utf-8")

    try:
        load_task_selection(tmp_path, "A_to_B", "bad")
    except ValueError as exc:
        assert "target-label access" in str(exc)
    else:
        raise AssertionError("expected label-access rejection")
