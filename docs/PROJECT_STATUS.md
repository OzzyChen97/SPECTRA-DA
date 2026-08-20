# Project status and handoff audit

This file records the current evidence boundary for the public SPECTRA-DA
release. It is intentionally conservative: an item is marked complete only
when the repository contains verifiable artifacts or commands that prove it.

## Completed in the repository

- The paper is reframed away from a state-of-the-art claim and toward
  target-label-free evaluation, covariance-aware risk recovery, and reliability
  of covariance correction under shift.
- The main evidence boundary is explicit: Transfer Score is stronger in the
  four-task Gate-1 real-target top-1 comparison (`0.1467` mean normalized
  regret versus `0.2560` for SPECTRA-DA), while the positive SPECTRA result is
  source-simulated (`0.02278 -> 0.01415`).
- The LaTeX tables mark best values and the SPECTRA-DA rows explicitly in
  source. Best cells use colored bold `\best{...}`; our entries use colored
  bold `\ours{...}` or `\oursbest{...}`.
- Figure 1 is generated from the workspace AutoSOTA image API path using
  `gpt-image-2`; the selected asset is
  `arxiv/figures/figure1-gpt-image-2.png`.
- Citation and claim audits are stored under `arxiv/notes/`, including the
  claim-evidence map and citation audit.
- Reliability-aware selection has executable scaffolding:
  - `selector/reliable_selection.py` for rank fusion;
  - `selector/run_reliable_suite.py` for one frozen configuration;
  - `selector/run_reliable_grid.py` for the pre-registered allowed grid;
  - `selector/check_reliable_inputs.py` for label-free preflight checks;
  - `selector/freeze_reliable_selector.py` for freezing one grid configuration
    before sealed evaluation.
- The reliable fusion workflow supports separate SPECTRA and Transfer Score
  output roots, so baseline JSON files do not need to be copied into the frozen
  SPECTRA root.
- The reliable fusion workflow now includes the v3 top-1 repair modes:
  fixed rank fusion, Transfer Score shortlist -> SPECTRA rerank, SPECTRA
  shortlist -> Transfer Score rerank, and support-adaptive mixture using
  SPECTRA's label-free covariance gamma when present.
- The v3 covariance-trust scaffold is implemented behind explicit
  `spectra_cal.py` switches. The default `--covariance-shrinkage-mode none`
  preserves the frozen v2 selector, while `support_gate` applies the
  descriptor-support gate directly to transported covariance, `fixed` exposes
  gamma sweeps for shrinkage ablations, and `pair_consistency` chooses gamma
  from `{0, .25, .5, .75, 1}` by minimizing a label-free pair-sum consistency
  residual. This is an implementation scaffold only; no real-target
  improvement is claimed from it yet.
- The v3 AutoSOTA objective scaffold is available at
  `selector/objective_v2.py` with configuration in
  `configs/v3_search_space.yaml`. It consumes an exported open-development
  candidate-truth report for the four Gate-1 tasks and evaluates selector JSON
  files on top-1 metrics, Transfer Score guardrails, CVaR, worst-task regret,
  NDCG@10, top-weighted Kendall, and oracle/top-5 shortlist recall at
  5/10/20/30/50% candidate cutoffs. These shortlist diagnostics determine
  whether Transfer Score or SPECTRA should own the first-stage candidate
  shortlist before reranking. Promotion now requires either Gate A or Gate B
  plus all guardrails. The guardrails require the candidate selector's worst
  open-development task regret not to exceed Transfer Score's worst task
  regret, preventing mean-only improvements from hiding a tail regression. When
  supplied with a source-simulated leave-one-shift-family selector report, it
  also rejects candidates whose source-sim CVaR degrades by more than the
  registered tolerance relative to the source-sim reference. It rejects
  `.sealed`, `sealed_eval`, and final-label paths and does not import the
  sealed evaluator.
- The trusted evaluator-side exporter for that open-development report is
  available at `sealed_eval/export_open_dev_truth.py`. It requires
  `--trusted-evaluator`, reads labels only for the four Gate-1 tasks, refuses
  non-Gate-1 tasks, and records `final_sealed_tasks_exposed: 0` in its output.
- The 675-candidate open-development truth artifact now exists at
  `results/gda_select/open_dev/gate1_candidate_truth.json`. It covers exactly
  the four Gate-1 tasks, reports 675 candidates per task, records
  `evaluator_target_label_read_count: 4`, and records
  `final_sealed_tasks_exposed: 0`. It was generated from the frozen
  `gda_select_v1` candidate bank and is open-development only; it is not a
  sealed-final result.
- The 675-candidate open-development comparison can now be reproduced with
  `selector/objective_v2.py`. The current evidence is mixed: `spectra_cal`
  remains below Transfer Score on mean normalized regret (`0.3201` versus
  `0.2168`), while `agreement_reference` reaches lower mean normalized regret
  than Transfer Score (`0.1630` versus `0.2168`) and higher selected Micro-F1,
  but still fails promotion because its top-10 shortlist recall is worse than
  Transfer Score's and the gain is localized. This is evidence for continued
  top-of-ranking calibration work, not a SOTA claim.
- Descriptor v2 has a label-free committee-behavior extractor at
  `shift_simulator/committee_descriptors.py`. It summarizes candidate entropy,
  prediction margins, class-prior concentration, pairwise disagreement,
  prediction-kernel effective rank/eigenvalues, within/between-method
  disagreement, same-trajectory disagreement, and checkpoint drift from
  `target_public.npz` plus metadata. Transfer Score component summaries are
  available behind an explicit optional flag.
- Descriptor attribution now has an executable correction-distance diagnostic
  at `shift_simulator/descriptor_metric.py`. It computes source-simulated
  recovery-relevant correction vectors from band risks and covariances, reports
  raw descriptor-distance versus correction-distance Spearman correlation,
  learns a nonnegative diagonal descriptor metric, and emits leave-one-shift-
  family diagnostics without reading real target labels. The diagnostic now
  audits every descriptor dimension, applies protocol-independent median/MAD
  scaling with clipping by default, can optionally use signed-log transforms
  for heavy-tailed descriptors, and reports target-to-bank nearest-distance
  contribution shares when a public target descriptor is present.
- External evaluation packaging exists in
  `scripts/package_external_evaluation.py`, supports repeated
  `--selection-root` arguments for one-shot multi-selector submissions, and
  records hashes of available baseline/reliable provenance manifests in the
  final submission manifest. Packaging rejects provenance manifests that do not
  explicitly report zero label access and zero protocol violations. The
  packager also supports `--require-selector` and `--min-selector-count` so a
  final comparison package can fail early if Transfer Score or multi-selector
  coverage is missing. The documentation requires freezing exactly one
  reliable-grid selector before sealed submission.
- A full-16 multi-selector package has been generated at
  `results/gda_select/submissions/final_multi_selector/` with archive
  `results/gda_select/submissions/final_multi_selector.tar.gz`. It covers all
  16 tasks, 675 candidates per task, and 40 label-free selectors including
  Transfer Score, source/confidence/geometric baselines, SPECTRA variants,
  rank-fusion variants, support-adaptive variants, and both shortlist/rerank
  directions. Its manifest records `target_label_access_count: 0`,
  `protocol_violation_count: 0`, and `contains_target_labels: false`.
- V3 readiness auditing is available at `scripts/check_v3_readiness.py`. It
  checks that the open-development truth report is restricted to the four
  Gate-1 tasks with the expected 675-candidate bank, that open-development
  selector roots cover required SPECTRA/Transfer Score selectors, and that the
  final submission manifest is a 16-task multi-selector package rather than a
  single-selector SPECTRA-only package. It also verifies packaged selection-file
  coverage and SHA-256 hashes for every declared selector on every task. It
  refuses `.sealed`, `sealed_eval`, and final-label paths.
- A GitHub Actions workflow is present at
  `.github/workflows/release-audit.yml` to run the lightweight public audit
  after the local commits are pushed.

## Verified local checks

The following code-level checks were run successfully in the current
environment:

```bash
python -m py_compile \
  selector/spectra_cal.py \
  selector/objective_v2.py \
  scripts/check_v3_readiness.py \
  sealed_eval/export_open_dev_truth.py \
  shift_simulator/committee_descriptors.py \
  selector/freeze_reliable_selector.py \
  selector/check_reliable_inputs.py \
  selector/reliable_selection.py \
  selector/run_reliable_suite.py \
  selector/run_reliable_grid.py \
  tests/test_spectra_prior.py \
  tests/selector/test_objective_v2.py \
  tests/sealed_eval/test_export_open_dev_truth.py \
  tests/test_committee_descriptors.py \
  tests/test_descriptor_metric.py \
  tests/test_v3_readiness.py \
  tests/selector/test_reliable_selection.py
```

Additional synthetic smoke tests were run for:

- reliable-grid generation;
- separate SPECTRA/Transfer Score roots;
- preflight rejection of missing Transfer Score input;
- reliable selector freezing;
- secret-pattern scanning for GitHub token formats.

These workflow checks are now encoded in `scripts/release_audit.py` so they can
be rerun without private artifacts.
The full release audit now also checks whether `arxiv/main.pdf` is newer than
the LaTeX sources and figure assets. The repository-pinned
`arxiv/tools/tectonic` binary rebuilds the paper locally; stale rendered tables
now fail the audit instead of being mistaken for current source.
The same audit is wired into `.github/workflows/release-audit.yml` for
push/pull-request CI after the repository is uploaded.

The focused SPECTRA prior/shrinkage tests were run under Python 3.12 with
third-party pytest plugin autoload disabled:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /usr/bin/python3.12 -m pytest \
  tests/test_spectra_prior.py -q

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /usr/bin/python3.12 -m pytest \
  tests/selector/test_objective_v2.py -q

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /usr/bin/python3.12 -m pytest \
  tests/sealed_eval/test_export_open_dev_truth.py -q

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /usr/bin/python3.12 -m pytest \
  tests/test_committee_descriptors.py -q

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /usr/bin/python3.12 -m pytest \
  tests/test_descriptor_metric.py -q

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /usr/bin/python3.12 -m pytest \
  tests/test_v3_readiness.py -q
```

The default Python 3.13 environment has a mismatched torch binary, so broader
torch-importing tests should use Python 3.12 in this container.

## Not yet completed

- The repository has local commits that have not been pushed to GitHub from
  this environment because GitHub write credentials are unavailable. Safe
  publishing instructions and a Git-bundle fallback are in
  `docs/GITHUB_PUBLISHING.md`.
- The reliability-aware grid has not been generated on the real 16-task
  candidate bank in this checkout because the matching frozen artifacts are not
  present here.
- `selector/objective_v2.py` has not yet been run on a real 675-candidate
  Gate-1 development bank because this checkout contains only aggregate Gate-1
  comparison results, not an exported candidate-level open-development truth
  report.
- The committee descriptor extractor is not yet wired into the default
  calibration bank. The descriptor metric diagnostic can score existing
  calibration descriptor keys, but the graph+committee+Transfer Score descriptor
  matrix still has to be generated on real calibration artifacts before it can
  drive covariance transport or shift-bank expansion.
- No new real-target result is reported for the reliable fusion direction. It
  is a pre-registered next method iteration, not an achieved result.
- The final one-time 16-transfer sealed evaluation is still pending. Until that
  evaluator runs once on a frozen submission bundle, the paper must not claim
  state-of-the-art real-target selection.

## Required next commands when artifacts and credentials are available

Generate Transfer Score and the reliable grid:

```bash
python sealed_eval/export_open_dev_truth.py \
  --candidate-root trajectory_bank/candidates/gda_select_v1 \
  --output results/gda_select/open_dev/gate1_candidate_truth.json \
  --expected-candidates-per-task 675 \
  --trusted-evaluator

python selector/run_baseline_suite.py \
  --candidate-root trajectory_bank/candidates/gda_select_v1 \
  --output-root results/gda_select/selections/baselines \
  --selector transfer_score

# Verify results/gda_select/selections/baselines/baseline_suite_manifest.json
# before using baseline outputs in objective_v2.py or sealed packaging.

python selector/check_reliable_inputs.py \
  --selection-root results/gda_select/selections/spectra_frozen_v2 \
  --spectra-root results/gda_select/selections/spectra_frozen_v2 \
  --transfer-root results/gda_select/selections/baselines \
  --spectra-selector spectra_robust \
  --transfer-selector transfer_score \
  --require-uncertainty

python selector/run_reliable_grid.py \
  --selection-root results/gda_select/selections/spectra_frozen_v2 \
  --spectra-root results/gda_select/selections/spectra_frozen_v2 \
  --transfer-root results/gda_select/selections/baselines \
  --output-root results/gda_select/selections/reliable_grid \
  --spectra-selector spectra_robust \
  --transfer-selector transfer_score \
  --uncertainty-weights 0,0.1,0.5,1 \
  --transfer-score-weights 0,0.25,0.5,0.75,1 \
  --covariance-shrinkages 0,0.25,0.5,0.75 \
  --calibration-temperatures 1

python selector/objective_v2.py \
  --dev-truth-report results/gda_select/open_dev/gate1_candidate_truth.json \
  --selection-root results/gda_select/selections/reliable_grid \
  --selection-root results/gda_select/selections/baselines \
  --objective-selector SELECTED_OR_CANDIDATE_SELECTOR_NAME \
  --transfer-selector transfer_score \
  --expected-candidate-count 675 \
  --source-sim-report results/gda_select/source_sim/leave_one_shift_family_selectors.json \
  --source-sim-reference-selector spectra_v2 \
  --source-sim-cvar-degradation-max 0.05 \
  --output results/gda_select/objective_v2.json
```

Check v3 readiness before freezing or submitting:

```bash
python scripts/check_v3_readiness.py \
  --open-dev-truth results/gda_select/open_dev/gate1_candidate_truth.json \
  --open-dev-selection-root results/gda_select/selections/reliable_grid \
  --open-dev-selection-root results/gda_select/selections/baselines \
  --final-submission-manifest results/gda_select/submissions/final_multi_selector/submission_manifest.json \
  --required-open-dev-selector SELECTED_RELIABLE_SELECTOR_NAME \
  --required-open-dev-selector transfer_score \
  --required-final-selector transfer_score \
  --min-final-selector-count 2
```

Freeze one pre-registered selector before sealed evaluation:

```bash
python selector/freeze_reliable_selector.py \
  --grid-root results/gda_select/selections/reliable_grid \
  --selector SELECTED_RELIABLE_SELECTOR_NAME \
  --output-root results/gda_select/selections/reliable_frozen
```

Package a frozen selector root for the external evaluator:

```bash
python scripts/package_external_evaluation.py \
  --candidate-root trajectory_bank/candidates/gda_select_v1 \
  --selection-root results/gda_select/selections/baselines \
  --selection-root results/gda_select/selections/reliable_frozen \
  --output-root results/gda_select/submissions/final_multi_selector \
  --archive results/gda_select/submissions/final_multi_selector.tar.gz \
  --require-selector transfer_score \
  --min-selector-count 2
```

Rebuild the paper PDF with the repository-pinned Tectonic binary:

```bash
cd arxiv
./tools/tectonic --keep-logs main.tex
```

or use an equivalent external `pdflatex`/BibTeX pipeline.
