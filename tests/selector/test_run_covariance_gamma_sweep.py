from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from selector import run_covariance_gamma_sweep as sweep


def _write_config(path: Path) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "active_parameters": {
                    "transport_regularization": 0.1,
                    "descriptor_floor": 0.05,
                    "risk_ridge": 1.0e-6,
                    "pair_weight_power": 1.0,
                    "bootstrap_samples": 16,
                    "uncertainty_beta": 0.25,
                }
            }
        ),
        encoding="utf-8",
    )


def _make_bank(tmp_path: Path) -> tuple[Path, Path]:
    candidate_root = tmp_path / "candidates"
    calibration_root = tmp_path / "calibration"
    for task in ("A_to_B", "C_to_D"):
        (candidate_root / task).mkdir(parents=True)
        calibration_dir = calibration_root / task / "abc123"
        calibration_dir.mkdir(parents=True)
        (calibration_dir / "metadata.json").write_text("{}", encoding="utf-8")
        (calibration_dir / "calibration.npz").write_bytes(b"not used")
    return candidate_root, calibration_root


def test_build_selector_specs_names_are_stable() -> None:
    specs = sweep.build_selector_specs(
        output_prefix="spectra_cov",
        fixed_gammas=[0.0, 0.5, 1.0],
        include_pair_consistency=True,
        include_support_gate=True,
    )

    assert [spec["selector"] for spec in specs] == [
        "spectra_cov_gamma000",
        "spectra_cov_gamma050",
        "spectra_cov_gamma100",
        "spectra_cov_pair_consistency",
        "spectra_cov_support_gate",
    ]


def test_run_sweep_writes_distinct_outputs_and_manifest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    candidate_root, calibration_root = _make_bank(tmp_path)
    config = tmp_path / "search_space.yaml"
    _write_config(config)
    output_root = tmp_path / "outputs"
    calls = []

    def fake_select(namespace: argparse.Namespace) -> dict:
        calls.append(namespace)
        return {
            "schema_version": 1,
            "task": namespace.task,
            "selector": namespace.output_selector,
            "candidate_bank_sha256": "bank",
            "candidate_count": 2,
            "candidate_scores": {"a": 0.0, "b": 1.0},
            "score_direction": "minimize",
            "score_semantics": "estimated_error",
            "selected_candidate_id": "a",
            "label_access_count": 0,
            "protocol_violation_count": 0,
            "transport_config": {
                "covariance_shrinkage_mode": namespace.covariance_shrinkage_mode,
                "fixed_covariance_gamma": namespace.fixed_covariance_gamma,
            },
            "selector_runtime_seconds": 0.01,
        }

    monkeypatch.setattr(sweep, "select_calibrated", fake_select)
    args = argparse.Namespace(
        candidate_root=candidate_root,
        calibration_root=calibration_root,
        output_root=output_root,
        config=config,
        sidecar_manifest=None,
        tasks=["A_to_B"],
        device="cpu",
        output_prefix="spectra_cov",
        fixed_gammas=[0.0, 0.5],
        include_pair_consistency=True,
        include_support_gate=False,
        support_rmse_threshold=None,
    )

    manifest = sweep.run_sweep(args)

    assert [call.output_selector for call in calls] == [
        "spectra_cov_gamma000",
        "spectra_cov_gamma050",
        "spectra_cov_pair_consistency",
    ]
    assert [call.covariance_shrinkage_mode for call in calls] == [
        "fixed",
        "fixed",
        "pair_consistency",
    ]
    assert (output_root / "A_to_B" / "spectra_cov_gamma000.json").is_file()
    assert (output_root / "A_to_B" / "spectra_cov_gamma050.json").is_file()
    assert (output_root / "A_to_B" / "spectra_cov_pair_consistency.json").is_file()
    manifest_path = output_root / "covariance_gamma_sweep_manifest.json"
    assert manifest_path.is_file()
    written = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert written["label_access_count"] == 0
    assert written["protocol_violation_count"] == 0
    assert len(written["outputs"]) == 3
    assert manifest["outputs"][1]["fixed_covariance_gamma"] == 0.5
