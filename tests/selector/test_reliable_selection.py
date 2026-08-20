from __future__ import annotations

import json

import pytest

from selector.check_reliable_inputs import main as check_reliable_inputs_main
from selector.freeze_reliable_selector import main as freeze_reliable_selector_main
from selector.run_reliable_grid import main as run_reliable_grid_main
from selector.reliable_selection import (
    covariance_confidence,
    load_selection,
    percentile_ranks,
    reliable_rank_fusion,
)


def _selection(
    selector: str,
    scores: dict[str, float],
    direction: str,
    *,
    uncertainty: dict[str, float] | None = None,
) -> dict:
    document = {
        "schema_version": 1,
        "created_at": "2026-08-20T00:00:00+00:00",
        "task": "A_to_B",
        "selector": selector,
        "candidate_bank_sha256": "bank",
        "candidate_count": len(scores),
        "candidate_scores": scores,
        "score_direction": direction,
        "score_semantics": "ranking",
        "selected_candidate_id": min(scores, key=scores.get)
        if direction == "minimize"
        else max(scores, key=scores.get),
        "label_access_count": 0,
        "protocol_violation_count": 0,
    }
    if uncertainty is not None:
        document["candidate_uncertainty"] = uncertainty
    return document


def test_percentile_ranks_make_lower_always_better() -> None:
    assert percentile_ranks({"a": 3.0, "b": 1.0, "c": 2.0}, "minimize") == {
        "b": 0.0,
        "c": 0.5,
        "a": 1.0,
    }
    assert percentile_ranks({"a": 3.0, "b": 1.0, "c": 2.0}, "maximize") == {
        "a": 0.0,
        "c": 0.5,
        "b": 1.0,
    }


def test_percentile_ranks_use_midrank_for_ties() -> None:
    assert percentile_ranks({"a": 1.0, "b": 1.0, "c": 3.0, "d": 4.0}, "minimize") == {
        "a": pytest.approx(1.0 / 6.0),
        "b": pytest.approx(1.0 / 6.0),
        "c": pytest.approx(2.0 / 3.0),
        "d": pytest.approx(1.0),
    }
    assert percentile_ranks({"a": 4.0, "b": 4.0, "c": 2.0, "d": 1.0}, "maximize") == {
        "a": pytest.approx(1.0 / 6.0),
        "b": pytest.approx(1.0 / 6.0),
        "c": pytest.approx(2.0 / 3.0),
        "d": pytest.approx(1.0),
    }


def test_reliable_rank_fusion_can_follow_transfer_score_prior() -> None:
    spectra = _selection(
        "spectra_robust",
        {"risk_best": 0.01, "transfer_best": 0.02, "bad": 0.50},
        "minimize",
    )
    transfer = _selection(
        "transfer_score",
        {"risk_best": 0.10, "transfer_best": 0.90, "bad": 0.20},
        "maximize",
    )

    result = reliable_rank_fusion(
        spectra,
        transfer,
        uncertainty_weight=0.0,
        transfer_score_weight=2.0,
        covariance_shrinkage=0.75,
        calibration_temperature=1.0,
        selector_name="spectra_reliable_rank_fusion",
    )

    assert result["selected_candidate_id"] == "transfer_best"
    assert result["score_direction"] == "minimize"
    assert result["score_semantics"] == "ranking"
    assert result["label_access_count"] == 0
    assert result["protocol_violation_count"] == 0
    assert result["fusion_config"]["transfer_selector"] == "transfer_score"


def test_reliable_rank_fusion_penalizes_uncertain_top_candidate() -> None:
    spectra = _selection(
        "spectra_robust",
        {"uncertain": 0.01, "stable": 0.02, "bad": 0.50},
        "minimize",
        uncertainty={"uncertain": 9.0, "stable": 0.1, "bad": 0.2},
    )
    transfer = _selection(
        "transfer_score",
        {"uncertain": 0.5, "stable": 0.5, "bad": 0.1},
        "maximize",
    )

    result = reliable_rank_fusion(
        spectra,
        transfer,
        uncertainty_weight=2.0,
        transfer_score_weight=0.0,
        covariance_shrinkage=0.0,
        calibration_temperature=1.0,
        selector_name="spectra_reliable_rank_fusion",
    )

    assert result["selected_candidate_id"] == "stable"
    assert result["fusion_config"]["uncertainty_source"] == "candidate_uncertainty"


def test_transfer_shortlist_spectra_rerank_uses_transfer_for_top_region() -> None:
    spectra = _selection(
        "spectra_robust",
        {"spectra_best": 0.01, "both_good": 0.02, "transfer_best": 0.40, "bad": 0.90},
        "minimize",
    )
    transfer = _selection(
        "transfer_score",
        {"spectra_best": 0.10, "both_good": 0.80, "transfer_best": 0.90, "bad": 0.20},
        "maximize",
    )

    result = reliable_rank_fusion(
        spectra,
        transfer,
        uncertainty_weight=0.0,
        transfer_score_weight=0.0,
        covariance_shrinkage=0.0,
        calibration_temperature=1.0,
        selector_name="spectra_reliable_tsr",
        fusion_mode="transfer_shortlist_spectra_rerank",
        shortlist_fraction=0.5,
    )

    assert result["selected_candidate_id"] == "both_good"
    assert result["fusion_config"]["shortlist_owner"] == "transfer_score"
    assert result["fusion_config"]["shortlist_size"] == 2


def test_transfer_shortlist_is_a_hard_exclusion_region() -> None:
    spectra = _selection(
        "spectra_robust",
        {
            "outside_spectra_best": 0.01,
            "inside_second": 0.02,
            "inside_first": 0.03,
            "outside_bad": 0.90,
        },
        "minimize",
        uncertainty={
            "outside_spectra_best": 0.0,
            "inside_second": 100.0,
            "inside_first": 200.0,
            "outside_bad": 0.0,
        },
    )
    transfer = _selection(
        "transfer_score",
        {
            "outside_spectra_best": 0.10,
            "inside_second": 0.90,
            "inside_first": 0.80,
            "outside_bad": 0.20,
        },
        "maximize",
    )

    result = reliable_rank_fusion(
        spectra,
        transfer,
        uncertainty_weight=10.0,
        transfer_score_weight=0.0,
        covariance_shrinkage=0.0,
        calibration_temperature=1.0,
        selector_name="spectra_reliable_tsr",
        fusion_mode="transfer_shortlist_spectra_rerank",
        shortlist_fraction=0.5,
    )

    assert result["selected_candidate_id"] in {"inside_first", "inside_second"}
    assert result["candidate_scores"]["outside_spectra_best"] > 1.0e11


def test_spectra_shortlist_transfer_rerank_is_the_directional_control() -> None:
    spectra = _selection(
        "spectra_robust",
        {"spectra_best": 0.01, "both_good": 0.02, "transfer_best": 0.40, "bad": 0.90},
        "minimize",
    )
    transfer = _selection(
        "transfer_score",
        {"spectra_best": 0.10, "both_good": 0.80, "transfer_best": 0.90, "bad": 0.20},
        "maximize",
    )

    result = reliable_rank_fusion(
        spectra,
        transfer,
        uncertainty_weight=0.0,
        transfer_score_weight=0.0,
        covariance_shrinkage=0.0,
        calibration_temperature=1.0,
        selector_name="spectra_reliable_str",
        fusion_mode="spectra_shortlist_transfer_rerank",
        shortlist_fraction=0.5,
    )

    assert result["selected_candidate_id"] == "both_good"
    assert result["fusion_config"]["shortlist_owner"] == "spectra"
    assert result["fusion_config"]["shortlist_size"] == 2


def test_support_adaptive_fusion_uses_spectra_covariance_gamma() -> None:
    spectra = _selection(
        "spectra_robust",
        {"spectra_best": 0.01, "transfer_best": 0.02, "bad": 0.50},
        "minimize",
    )
    spectra["transport_diagnostics"] = {
        "covariance_shrinkage": {
            "mode": "pair_consistency",
            "gamma": 0.0,
        }
    }
    transfer = _selection(
        "transfer_score",
        {"spectra_best": 0.10, "transfer_best": 0.90, "bad": 0.20},
        "maximize",
    )

    result = reliable_rank_fusion(
        spectra,
        transfer,
        uncertainty_weight=0.0,
        transfer_score_weight=1.0,
        covariance_shrinkage=0.75,
        calibration_temperature=1.0,
        selector_name="spectra_reliable_sa",
        fusion_mode="support_adaptive",
    )

    assert covariance_confidence(spectra, fallback_shrinkage=0.75) == (
        0.0,
        "transport_diagnostics.covariance_shrinkage.gamma",
    )
    assert result["selected_candidate_id"] == "transfer_best"
    assert result["fusion_config"]["covariance_gamma"] == 0.0


def test_support_adaptive_transfer_weight_is_not_clamped_to_one() -> None:
    spectra = _selection(
        "spectra_robust",
        {"spectra_best": 0.01, "transfer_best": 0.02, "bad": 0.50},
        "minimize",
    )
    spectra["transport_diagnostics"] = {
        "covariance_shrinkage": {"mode": "pair_consistency", "gamma": 0.5}
    }
    transfer = _selection(
        "transfer_score",
        {"spectra_best": 0.10, "transfer_best": 0.90, "bad": 0.20},
        "maximize",
    )

    no_transfer = reliable_rank_fusion(
        spectra,
        transfer,
        uncertainty_weight=0.0,
        transfer_score_weight=0.0,
        covariance_shrinkage=0.0,
        calibration_temperature=1.0,
        selector_name="spectra_reliable_sa",
        fusion_mode="support_adaptive",
    )
    with_transfer = reliable_rank_fusion(
        spectra,
        transfer,
        uncertainty_weight=0.0,
        transfer_score_weight=1.0,
        covariance_shrinkage=0.0,
        calibration_temperature=1.0,
        selector_name="spectra_reliable_sa",
        fusion_mode="support_adaptive",
    )

    assert no_transfer["candidate_scores"] != with_transfer["candidate_scores"]
    assert no_transfer["selected_candidate_id"] == "spectra_best"
    assert with_transfer["selected_candidate_id"] == "transfer_best"


def test_reliable_rank_fusion_rejects_misaligned_selection_files() -> None:
    spectra = _selection("spectra_robust", {"a": 0.1, "b": 0.2}, "minimize")
    transfer = _selection("transfer_score", {"a": 0.8, "c": 0.7}, "maximize")

    with pytest.raises(ValueError, match="coverage mismatch"):
        reliable_rank_fusion(
            spectra,
            transfer,
            uncertainty_weight=0.0,
            transfer_score_weight=1.0,
            covariance_shrinkage=0.0,
            calibration_temperature=1.0,
            selector_name="spectra_reliable_rank_fusion",
        )


def test_load_selection_rejects_protocol_violations(tmp_path) -> None:
    path = tmp_path / "selection.json"
    document = _selection("spectra_robust", {"a": 0.1}, "minimize")
    document["label_access_count"] = 1
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="target-label access"):
        load_selection(path)


def test_run_reliable_grid_writes_manifest_and_selector_outputs(tmp_path, monkeypatch) -> None:
    selection_root = tmp_path / "inputs"
    task_dir = selection_root / "A_to_B"
    task_dir.mkdir(parents=True)
    spectra = _selection(
        "spectra_robust",
        {"risk_best": 0.01, "transfer_best": 0.02, "bad": 0.50},
        "minimize",
        uncertainty={"risk_best": 0.9, "transfer_best": 0.1, "bad": 0.2},
    )
    transfer = _selection(
        "transfer_score",
        {"risk_best": 0.10, "transfer_best": 0.90, "bad": 0.20},
        "maximize",
    )
    (task_dir / "spectra_robust.json").write_text(json.dumps(spectra), encoding="utf-8")
    (task_dir / "transfer_score.json").write_text(json.dumps(transfer), encoding="utf-8")

    output_root = tmp_path / "outputs"
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_reliable_grid.py",
            "--selection-root",
            str(selection_root),
            "--output-root",
            str(output_root),
            "--uncertainty-weights",
            "0,1",
            "--transfer-score-weights",
            "0,1",
            "--covariance-shrinkages",
            "0.5",
            "--calibration-temperatures",
            "1",
        ],
    )

    run_reliable_grid_main()

    manifest = json.loads((output_root / "reliable_grid_manifest.json").read_text(encoding="utf-8"))
    selector_files = sorted((output_root / "A_to_B").glob("*.json"))
    assert manifest["selector_count"] == 4
    assert len(selector_files) == 4
    assert manifest["protocol"]["label_free_post_selector"] is True
    assert set(manifest["protocol"]["allowed_knobs"]) == {
        "uncertainty_weight",
        "transfer_score_weight",
        "covariance_shrinkage",
        "calibration_temperature",
        "fusion_mode",
        "shortlist_fraction",
    }
    for path in selector_files:
        document = json.loads(path.read_text(encoding="utf-8"))
        assert document["label_access_count"] == 0
        assert document["protocol_violation_count"] == 0
        assert document["score_direction"] == "minimize"


def test_run_reliable_grid_accepts_separate_spectra_and_transfer_roots(tmp_path, monkeypatch) -> None:
    spectra_root = tmp_path / "spectra"
    transfer_root = tmp_path / "transfer"
    (spectra_root / "A_to_B").mkdir(parents=True)
    (transfer_root / "A_to_B").mkdir(parents=True)
    spectra = _selection(
        "spectra_robust",
        {"risk_best": 0.01, "transfer_best": 0.02, "bad": 0.50},
        "minimize",
        uncertainty={"risk_best": 0.9, "transfer_best": 0.1, "bad": 0.2},
    )
    transfer = _selection(
        "transfer_score",
        {"risk_best": 0.10, "transfer_best": 0.90, "bad": 0.20},
        "maximize",
    )
    (spectra_root / "A_to_B" / "spectra_robust.json").write_text(json.dumps(spectra), encoding="utf-8")
    (transfer_root / "A_to_B" / "transfer_score.json").write_text(json.dumps(transfer), encoding="utf-8")
    output_root = tmp_path / "outputs"
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_reliable_grid.py",
            "--selection-root",
            str(spectra_root),
            "--spectra-root",
            str(spectra_root),
            "--transfer-root",
            str(transfer_root),
            "--output-root",
            str(output_root),
            "--uncertainty-weights",
            "0",
            "--transfer-score-weights",
            "1",
            "--covariance-shrinkages",
            "0.5",
            "--calibration-temperatures",
            "1",
        ],
    )

    run_reliable_grid_main()

    manifest = json.loads((output_root / "reliable_grid_manifest.json").read_text(encoding="utf-8"))
    assert manifest["spectra_root"] == str(spectra_root.resolve())
    assert manifest["transfer_root"] == str(transfer_root.resolve())
    assert len(list((output_root / "A_to_B").glob("*.json"))) == 1


def test_run_reliable_grid_can_generate_shortlist_modes(tmp_path, monkeypatch) -> None:
    selection_root = tmp_path / "inputs"
    task_dir = selection_root / "A_to_B"
    task_dir.mkdir(parents=True)
    spectra = _selection(
        "spectra_robust",
        {"spectra_best": 0.01, "both_good": 0.02, "transfer_best": 0.40, "bad": 0.90},
        "minimize",
    )
    transfer = _selection(
        "transfer_score",
        {"spectra_best": 0.10, "both_good": 0.80, "transfer_best": 0.90, "bad": 0.20},
        "maximize",
    )
    (task_dir / "spectra_robust.json").write_text(json.dumps(spectra), encoding="utf-8")
    (task_dir / "transfer_score.json").write_text(json.dumps(transfer), encoding="utf-8")
    output_root = tmp_path / "outputs"
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_reliable_grid.py",
            "--selection-root",
            str(selection_root),
            "--output-root",
            str(output_root),
            "--uncertainty-weights",
            "0",
            "--transfer-score-weights",
            "0",
            "--covariance-shrinkages",
            "0",
            "--calibration-temperatures",
            "1",
            "--fusion-modes",
            "transfer_shortlist_spectra_rerank,spectra_shortlist_transfer_rerank",
            "--shortlist-fractions",
            "0.5",
        ],
    )

    run_reliable_grid_main()

    outputs = sorted(path.name for path in (output_root / "A_to_B").glob("*.json"))
    assert outputs == [
        "spectra_reliable_uw000_tw000_cs000_ct100_str_sf050.json",
        "spectra_reliable_uw000_tw000_cs000_ct100_tsr_sf050.json",
    ]


def test_check_reliable_inputs_rejects_missing_transfer_score(tmp_path, monkeypatch) -> None:
    selection_root = tmp_path / "inputs"
    task_dir = selection_root / "A_to_B"
    task_dir.mkdir(parents=True)
    spectra = _selection("spectra_robust", {"a": 0.1, "b": 0.2}, "minimize")
    (task_dir / "spectra_robust.json").write_text(json.dumps(spectra), encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        [
            "check_reliable_inputs.py",
            "--selection-root",
            str(selection_root),
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        check_reliable_inputs_main()
    assert exc_info.value.code == 1


def test_check_reliable_inputs_accepts_matching_label_free_inputs(tmp_path, monkeypatch) -> None:
    selection_root = tmp_path / "inputs"
    task_dir = selection_root / "A_to_B"
    task_dir.mkdir(parents=True)
    spectra = _selection(
        "spectra_robust",
        {"a": 0.1, "b": 0.2},
        "minimize",
        uncertainty={"a": 0.3, "b": 0.1},
    )
    transfer = _selection("transfer_score", {"a": 0.7, "b": 0.9}, "maximize")
    (task_dir / "spectra_robust.json").write_text(json.dumps(spectra), encoding="utf-8")
    (task_dir / "transfer_score.json").write_text(json.dumps(transfer), encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        [
            "check_reliable_inputs.py",
            "--selection-root",
            str(selection_root),
            "--require-uncertainty",
        ],
    )

    check_reliable_inputs_main()


def test_freeze_reliable_selector_copies_one_registered_selector(tmp_path, monkeypatch) -> None:
    grid_root = tmp_path / "grid"
    task_dir = grid_root / "A_to_B"
    task_dir.mkdir(parents=True)
    selector = "spectra_reliable_uw000_tw100_cs050_ct100"
    manifest = {
        "schema_version": 1,
        "tasks": ["A_to_B"],
        "selectors": [
            {
                "selector": selector,
                "uncertainty_weight": 0.0,
                "transfer_score_weight": 1.0,
                "covariance_shrinkage": 0.5,
                "calibration_temperature": 1.0,
            }
        ],
    }
    (grid_root / "reliable_grid_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    document = _selection(selector, {"a": 0.2, "b": 0.1}, "minimize")
    (task_dir / f"{selector}.json").write_text(json.dumps(document), encoding="utf-8")

    output_root = tmp_path / "frozen"
    monkeypatch.setattr(
        "sys.argv",
        [
            "freeze_reliable_selector.py",
            "--grid-root",
            str(grid_root),
            "--selector",
            selector,
            "--output-root",
            str(output_root),
        ],
    )

    freeze_reliable_selector_main()

    assert (output_root / "A_to_B" / f"{selector}.json").is_file()
    freeze_manifest = json.loads((output_root / "reliable_freeze_manifest.json").read_text(encoding="utf-8"))
    assert freeze_manifest["selector"] == selector
    assert freeze_manifest["task_count"] == 1
    assert freeze_manifest["label_access_count"] == 0
    assert freeze_manifest["protocol_violation_count"] == 0
