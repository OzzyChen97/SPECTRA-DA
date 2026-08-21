#!/usr/bin/env python3
"""Lightweight public-release audit for SPECTRA-DA.

This script intentionally avoids importing torch, trajectory loaders, or sealed
evaluation code. It checks repository text and Python syntax only, so it can be
run before pushing a public release or handing the project to an external
evaluator.
"""

from __future__ import annotations

import json
import py_compile
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]

REQUIRED_FILES = [
    "README.md",
    "docs/PROJECT_STATUS.md",
    "docs/RESULTS.md",
    "docs/GITHUB_PUBLISHING.md",
    "docs/REPRODUCIBILITY.md",
    "docs/EXTERNAL_EVALUATION.md",
    "docs/AUTOSOTA_RELIABLE_SELECTION_STEERING.md",
    "configs/v3_search_space.yaml",
    "arxiv/main.tex",
    "arxiv/sections/abstract.tex",
    "arxiv/sections/introduction.tex",
    "arxiv/sections/method.tex",
    "arxiv/sections/experiments.tex",
    "arxiv/sections/conclusion.tex",
    "arxiv/figures/figure1-gpt-image-2.png",
    "arxiv/notes/claim_evidence.md",
    "arxiv/notes/citation_audit.md",
    "selector/reliable_selection.py",
    "selector/objective_v2.py",
    "selector/loto_open_dev_selection.py",
    "selector/selector_complementarity.py",
    "selector/consensus_selection.py",
    "selector/baselines.py",
    "selector/transfer_score.py",
    "selector/run_baseline_suite.py",
    "selector/run_reliable_suite.py",
    "selector/run_reliable_grid.py",
    "selector/run_covariance_gamma_sweep.py",
    "selector/selection_error_decomposition.py",
    "selector/trajectory_shortlist_selection.py",
    "selector/check_reliable_inputs.py",
    "selector/freeze_reliable_selector.py",
    "sealed_eval/export_open_dev_truth.py",
    "shift_simulator/committee_descriptors.py",
    "shift_simulator/descriptor_metric.py",
    "scripts/package_external_evaluation.py",
    "scripts/check_v3_readiness.py",
]

PYTHON_SYNTAX_FILES = [
    "selector/reliable_selection.py",
    "selector/objective_v2.py",
    "selector/baselines.py",
    "selector/transfer_score.py",
    "selector/run_baseline_suite.py",
    "selector/run_reliable_suite.py",
    "selector/run_reliable_grid.py",
    "selector/run_covariance_gamma_sweep.py",
    "selector/selection_error_decomposition.py",
    "selector/trajectory_shortlist_selection.py",
    "selector/check_reliable_inputs.py",
    "selector/freeze_reliable_selector.py",
    "sealed_eval/export_open_dev_truth.py",
    "shift_simulator/committee_descriptors.py",
    "shift_simulator/descriptor_metric.py",
    "tests/selector/test_reliable_selection.py",
    "tests/selector/test_objective_v2.py",
    "tests/selector/test_loto_open_dev_selection.py",
    "tests/selector/test_selector_complementarity.py",
    "tests/selector/test_consensus_selection.py",
    "tests/selector/test_selection_error_decomposition.py",
    "tests/selector/test_trajectory_shortlist_selection.py",
    "tests/sealed_eval/test_export_open_dev_truth.py",
    "tests/test_committee_descriptors.py",
    "tests/test_descriptor_metric.py",
    "tests/test_label_free_baselines.py",
    "tests/selector/test_baseline_resume.py",
    "tests/test_external_evaluation_package.py",
    "tests/test_v3_readiness.py",
]

CLAIM_BOUNDARY_PATTERNS = {
    "no_sota_claim": r"no state-of-the-art claim|not a state-of-the-art",
    "transfer_score_stronger": r"Transfer Score remains stronger",
    "gate1_transfer_score_value": r"0\.1467",
    "gate1_spectra_value": r"0\.2560",
    "source_simulated_gain_start": r"0\.02278",
    "source_simulated_gain_end": r"0\.01415",
    "reliable_not_completed_result": r"not (?:yet )?completed|not an achieved result|not report this as a completed",
}

FORBIDDEN_SECRET_PATTERNS = {
    "github_classic_token": re.compile(r"ghp_[A-Za-z0-9_]{20,}"),
    "github_fine_grained_token": re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
}


def read_text(path: str) -> str:
    return (REPO / path).read_text(encoding="utf-8")


def check_required_files() -> list[dict[str, str]]:
    failures = []
    for relative in REQUIRED_FILES:
        if not (REPO / relative).is_file():
            failures.append({"check": "required_file", "path": relative})
    return failures


def check_python_syntax() -> list[dict[str, str]]:
    failures = []
    for relative in PYTHON_SYNTAX_FILES:
        path = REPO / relative
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as exc:
            failures.append({"check": "python_syntax", "path": relative, "error": str(exc)})
    return failures


def check_claim_boundaries() -> list[dict[str, str]]:
    text = "\n".join(
        read_text(path)
        for path in [
            "README.md",
            "docs/PROJECT_STATUS.md",
            "docs/RESULTS.md",
            "arxiv/sections/abstract.tex",
            "arxiv/sections/introduction.tex",
            "arxiv/sections/method.tex",
            "arxiv/sections/experiments.tex",
            "arxiv/sections/conclusion.tex",
            "arxiv/notes/claim_evidence.md",
        ]
    )
    failures = []
    for name, pattern in CLAIM_BOUNDARY_PATTERNS.items():
        if re.search(pattern, text, flags=re.IGNORECASE) is None:
            failures.append({"check": "claim_boundary", "missing": name})
    return failures


def check_table_highlights() -> list[dict[str, str]]:
    main = read_text("arxiv/main.tex")
    experiments = read_text("arxiv/sections/experiments.tex")
    expectations = {
        "best_macro": r"\\newcommand\{\\best\}",
        "ours_macro": r"\\newcommand\{\\ours\}",
        "best_background": r"\\colorbox\{spectrabluebg\}",
        "ours_background": r"\\colorbox\{spectragreenbg\}",
        "oursbest_background": r"\\colorbox\{spectraorangebg\}",
        "highlight_bold": r"\\textbf\{\\textcolor",
        "ours_values": r"\\ours\{0\.2560\}.*\\ours\{12\.16\}.*\\ours\{0\.695\}",
    }
    text = main + "\n" + experiments.replace("\n", " ")
    failures = []
    for name, pattern in expectations.items():
        if re.search(pattern, text) is None:
            failures.append({"check": "table_highlight", "missing": name})
    return failures


def check_arxiv_pdf_freshness() -> list[dict[str, str]]:
    pdf = REPO / "arxiv/main.pdf"
    if not pdf.is_file():
        return []
    source_paths = [
        REPO / "arxiv/main.tex",
        *sorted((REPO / "arxiv/sections").glob("*.tex")),
        *sorted((REPO / "arxiv/figures").glob("*")),
    ]
    newest_source = max(
        (path for path in source_paths if path.is_file()),
        key=lambda path: path.stat().st_mtime,
    )
    if newest_source.stat().st_mtime > pdf.stat().st_mtime:
        return [
            {
                "check": "arxiv_pdf_freshness",
                "pdf": "arxiv/main.pdf",
                "newer_source": str(newest_source.relative_to(REPO)),
                "detail": "Rebuild the PDF; rendered tables may not reflect updated best/ours highlights.",
            }
        ]
    return []


def check_secret_patterns() -> list[dict[str, str]]:
    roots = [
        ".github",
        "README.md",
        "docs",
        "selector",
        "tests",
        "arxiv",
        "configs",
        "results",
        "scripts",
    ]
    failures = []
    for root in roots:
        path = REPO / root
        if not path.exists():
            continue
        paths = [path] if path.is_file() else [child for child in path.rglob("*") if child.is_file()]
        for child in paths:
            if any(part in {".git", "__pycache__"} for part in child.parts):
                continue
            try:
                text = child.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            for name, pattern in FORBIDDEN_SECRET_PATTERNS.items():
                if pattern.search(text):
                    failures.append(
                        {
                            "check": "secret_pattern",
                            "pattern": name,
                            "path": str(child.relative_to(REPO)),
                        }
                    )
    return failures


def _write_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document), encoding="utf-8")


def check_reliable_workflow_smoke() -> list[dict[str, str]]:
    failures = []
    with tempfile.TemporaryDirectory(prefix="spectra_release_audit_") as temporary:
        root = Path(temporary)
        spectra_root = root / "spectra"
        transfer_root = root / "transfer"
        task = "A_to_B"
        spectra_doc = {
            "schema_version": 1,
            "task": task,
            "selector": "spectra_robust",
            "candidate_bank_sha256": "bank",
            "candidate_count": 3,
            "candidate_scores": {"risk_best": 0.01, "transfer_best": 0.02, "bad": 0.50},
            "candidate_transport_uncertainty": {"risk_best": 0.9, "transfer_best": 0.1, "bad": 0.2},
            "score_direction": "minimize",
            "score_semantics": "ranking",
            "selected_candidate_id": "risk_best",
            "label_access_count": 0,
            "protocol_violation_count": 0,
        }
        transfer_doc = {
            **spectra_doc,
            "selector": "transfer_score",
            "candidate_scores": {"risk_best": 0.10, "transfer_best": 0.90, "bad": 0.20},
            "score_direction": "maximize",
            "selected_candidate_id": "transfer_best",
        }
        _write_json(spectra_root / task / "spectra_robust.json", spectra_doc)

        missing = subprocess.run(
            [
                sys.executable,
                "selector/check_reliable_inputs.py",
                "--selection-root",
                str(spectra_root),
                "--spectra-root",
                str(spectra_root),
                "--transfer-root",
                str(transfer_root),
            ],
            cwd=REPO,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if missing.returncode == 0:
            failures.append({"check": "reliable_smoke", "error": "missing Transfer Score input was accepted"})

        _write_json(transfer_root / task / "transfer_score.json", transfer_doc)
        preflight = subprocess.run(
            [
                sys.executable,
                "selector/check_reliable_inputs.py",
                "--selection-root",
                str(spectra_root),
                "--spectra-root",
                str(spectra_root),
                "--transfer-root",
                str(transfer_root),
                "--require-uncertainty",
            ],
            cwd=REPO,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if preflight.returncode != 0:
            failures.append({"check": "reliable_smoke", "step": "preflight", "error": preflight.stderr})

        grid_root = root / "grid"
        grid = subprocess.run(
            [
                sys.executable,
                "selector/run_reliable_grid.py",
                "--selection-root",
                str(spectra_root),
                "--spectra-root",
                str(spectra_root),
                "--transfer-root",
                str(transfer_root),
                "--output-root",
                str(grid_root),
                "--uncertainty-weights",
                "0",
                "--transfer-score-weights",
                "1",
                "--covariance-shrinkages",
                "0.5",
                "--calibration-temperatures",
                "1",
            ],
            cwd=REPO,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if grid.returncode != 0:
            failures.append({"check": "reliable_smoke", "step": "grid", "error": grid.stderr})

        manifest_path = grid_root / "reliable_grid_manifest.json"
        selector = "spectra_reliable_uw000_tw100_cs050_ct100"
        if not manifest_path.is_file():
            failures.append({"check": "reliable_smoke", "step": "grid", "error": "missing manifest"})
        elif not (grid_root / task / f"{selector}.json").is_file():
            failures.append({"check": "reliable_smoke", "step": "grid", "error": "missing selector output"})

        frozen_root = root / "frozen"
        freeze = subprocess.run(
            [
                sys.executable,
                "selector/freeze_reliable_selector.py",
                "--grid-root",
                str(grid_root),
                "--selector",
                selector,
                "--output-root",
                str(frozen_root),
            ],
            cwd=REPO,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if freeze.returncode != 0:
            failures.append({"check": "reliable_smoke", "step": "freeze", "error": freeze.stderr})
        elif not (frozen_root / "reliable_freeze_manifest.json").is_file():
            failures.append({"check": "reliable_smoke", "step": "freeze", "error": "missing freeze manifest"})
    return failures


def check_objective_v2_smoke() -> list[dict[str, str]]:
    failures = []
    with tempfile.TemporaryDirectory(prefix="spectra_objective_v2_audit_") as temporary:
        root = Path(temporary)
        tasks = ["ACMv9_to_Citationv1", "USA_to_BRAZIL"]
        truth = {
            "schema_version": 1,
            "tasks": [
                {
                    "task": task,
                    "candidate_bank_sha256": f"bank-{task}",
                    "candidate_truth": {
                        "best": {"target_error": 0.10},
                        "middle": {"target_error": 0.20},
                        "bad": {"target_error": 0.50},
                    },
                }
                for task in tasks
            ],
        }
        truth_path = root / "open_dev_truth.json"
        _write_json(truth_path, truth)
        spectra_root = root / "spectra"
        transfer_root = root / "transfer"
        for task in tasks:
            base = {
                "schema_version": 1,
                "task": task,
                "candidate_bank_sha256": f"bank-{task}",
                "candidate_count": 3,
                "score_semantics": "ranking",
                "label_access_count": 0,
                "protocol_violation_count": 0,
                "selector_runtime_seconds": 1.0,
            }
            _write_json(
                spectra_root / task / "spectra_trust.json",
                {
                    **base,
                    "selector": "spectra_trust",
                    "candidate_scores": {"best": 0.0, "middle": 0.5, "bad": 1.0},
                    "score_direction": "minimize",
                    "selected_candidate_id": "best",
                },
            )
            _write_json(
                transfer_root / task / "transfer_score.json",
                {
                    **base,
                    "selector": "transfer_score",
                    "candidate_scores": {"best": 0.2, "middle": 0.9, "bad": 0.1},
                    "score_direction": "maximize",
                    "selected_candidate_id": "middle",
                },
            )
        output = root / "objective_v2.json"
        run = subprocess.run(
            [
                sys.executable,
                "selector/objective_v2.py",
                "--dev-truth-report",
                str(truth_path),
                "--selection-root",
                str(spectra_root),
                "--selection-root",
                str(transfer_root),
                "--tasks",
                *tasks,
                "--objective-selector",
                "spectra_trust",
                "--expected-candidate-count",
                "3",
                "--output",
                str(output),
            ],
            cwd=REPO,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if run.returncode != 0:
            failures.append({"check": "objective_v2_smoke", "error": run.stderr})
        elif not output.is_file():
            failures.append({"check": "objective_v2_smoke", "error": "missing output"})
        else:
            document = json.loads(output.read_text(encoding="utf-8"))
            if document.get("label_access_count") != 0:
                failures.append({"check": "objective_v2_smoke", "error": "label access reported"})
            if not document.get("guardrails", {}).get("gate_b_pass"):
                failures.append({"check": "objective_v2_smoke", "error": "expected Gate B pass"})
    return failures


def collect_warnings() -> list[dict[str, str]]:
    warnings = []
    latex_engines = [
        shutil.which(name) for name in ("tectonic", "pdflatex", "xelatex", "latexmk")
    ]
    latex_engines.append(str(REPO / "arxiv/tools/tectonic"))
    if not any(engine and Path(engine).is_file() for engine in latex_engines):
        warnings.append(
            {
                "check": "latex_engine",
                "status": "missing",
                "detail": "PDF rebuild cannot be verified in this environment",
            }
        )
    return warnings


def main() -> None:
    failures = []
    failures.extend(check_required_files())
    failures.extend(check_python_syntax())
    failures.extend(check_claim_boundaries())
    failures.extend(check_table_highlights())
    failures.extend(check_arxiv_pdf_freshness())
    failures.extend(check_secret_patterns())
    failures.extend(check_reliable_workflow_smoke())
    failures.extend(check_objective_v2_smoke())
    warnings = collect_warnings()
    result = {
        "schema_version": 1,
        "ok": not failures,
        "failure_count": len(failures),
        "warning_count": len(warnings),
        "failures": failures,
        "warnings": warnings,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
