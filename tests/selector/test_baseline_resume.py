from __future__ import annotations

import json
import sys

from selector.baselines import build_selection_result
import selector.run_baseline_suite as baseline_suite
from selector.run_baseline_suite import is_valid_cached_selection


def _records() -> list[dict]:
    return [
        {
            "metadata": {
                "artifact_sha256": {
                    "source_val.npz": f"source-{index}",
                    "target_public.npz": f"target-{index}",
                    "model_state.pt": f"state-{index}",
                },
                "candidate_id": f"task__Method__config__seed-1__epoch-{index:04d}",
                "config_id": "config",
                "epoch": index,
                "method": "Method",
                "seed": 1,
                "source_graph_sha256": "source-graph",
                "source_split_sha256": "source-split",
                "target_graph_sha256": "target-graph",
                "task": "A_to_B",
            }
        }
        for index in range(1, 4)
    ]


def test_valid_cached_selection_is_reused(tmp_path) -> None:
    records = _records()
    scores = {
        record["metadata"]["candidate_id"]: float(index)
        for index, record in enumerate(records)
    }
    result = build_selection_result(
        records,
        "task",
        "entropy",
        scores,
        "minimize",
        bank_hash="frozen-bank",
    )
    path = tmp_path / "entropy.json"
    path.write_text(json.dumps(result), encoding="utf-8")

    assert is_valid_cached_selection(
        path,
        task="task",
        selector="entropy",
        direction="minimize",
        candidate_ids=set(scores),
        bank_hash="frozen-bank",
    )


def test_cached_selection_must_match_bank_coverage_and_optimum(tmp_path) -> None:
    records = _records()
    scores = {
        record["metadata"]["candidate_id"]: float(index)
        for index, record in enumerate(records)
    }
    result = build_selection_result(
        records,
        "task",
        "entropy",
        scores,
        "minimize",
        bank_hash="frozen-bank",
    )
    path = tmp_path / "entropy.json"

    wrong_bank = dict(result, candidate_bank_sha256="stale-bank")
    path.write_text(json.dumps(wrong_bank), encoding="utf-8")
    assert not is_valid_cached_selection(
        path,
        task="task",
        selector="entropy",
        direction="minimize",
        candidate_ids=set(scores),
        bank_hash="frozen-bank",
    )

    incomplete = dict(result, candidate_scores=dict(list(scores.items())[:-1]))
    path.write_text(json.dumps(incomplete), encoding="utf-8")
    assert not is_valid_cached_selection(
        path,
        task="task",
        selector="entropy",
        direction="minimize",
        candidate_ids=set(scores),
        bank_hash="frozen-bank",
    )

    non_optimal = dict(result, selected_candidate_id=max(scores))
    path.write_text(json.dumps(non_optimal), encoding="utf-8")
    assert not is_valid_cached_selection(
        path,
        task="task",
        selector="entropy",
        direction="minimize",
        candidate_ids=set(scores),
        bank_hash="frozen-bank",
    )


def test_baseline_suite_writes_manifest_and_records_cache_reuse(tmp_path, monkeypatch) -> None:
    records = _records()

    def toy_selector(candidate_records: list[dict]) -> dict[str, float]:
        return {
            record["metadata"]["candidate_id"]: float(index)
            for index, record in enumerate(candidate_records)
        }

    monkeypatch.setitem(baseline_suite.SELECTORS, "toy_baseline", (toy_selector, "minimize"))
    monkeypatch.setattr(
        baseline_suite,
        "discover_candidate_records",
        lambda candidate_root, task: records,
    )

    output_root = tmp_path / "baselines"
    argv = [
        "run_baseline_suite.py",
        "--candidate-root",
        str(tmp_path / "candidates"),
        "--output-root",
        str(output_root),
        "--task",
        "A_to_B",
        "--selector",
        "toy_baseline",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    baseline_suite.main()

    manifest_path = baseline_suite.baseline_manifest_path(output_root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["selector_count"] == 1
    assert manifest["selectors"] == ["toy_baseline"]
    assert manifest["label_access_count"] == 0
    assert manifest["protocol_violation_count"] == 0
    selector_report = manifest["task_results"]["A_to_B"]["selectors"]["toy_baseline"]
    assert selector_report["reused_cached_selection"] is False
    assert selector_report["selection_file"] == "A_to_B/toy_baseline.json"
    assert len(selector_report["selection_sha256"]) == 64

    baseline_suite.main()
    reused_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    reused_report = reused_manifest["task_results"]["A_to_B"]["selectors"]["toy_baseline"]
    assert reused_report["reused_cached_selection"] is True
    assert reused_report["selection_sha256"] == selector_report["selection_sha256"]
