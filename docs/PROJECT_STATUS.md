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
- Leave-one-task-out open-development selector-choice auditing is available at
  `selector/loto_open_dev_selection.py`. It chooses reliable-selector
  configurations from three open-dev tasks and validates on the held-out task,
  so a four-task average cannot silently select a configuration using the same
  task on which it is reported.
- Selector complementarity auditing is available at
  `selector/selector_complementarity.py`. It compares full candidate-score
  rankings, top-set overlaps, selected-candidate cross-ranks, tie statistics,
  and selected method/seed/epoch metadata across label-free selectors. The
  current open-development artifact is
  `results/gda_select/open_dev/selector_complementarity_diagnostic.json`; it
  confirms that Transfer Score and Agreement Reference/repaired SPECTRA-like
  selectors use substantially different top-of-ranking signals, but simple
  overlap statistics do not yet provide a reliable trust rule.
- Explicit Stage-B consensus controls are available at
  `selector/consensus_selection.py`. The current open-development artifact is
  `results/gda_select/open_dev/stage_b_consensus_objective_v2.json`, with
  leave-one-task-out validation in
  `results/gda_select/open_dev/stage_b_consensus_loto_objective_v2.json`.
  Transfer Score top-20% -> SPECTRA/Agreement consensus slightly improves the
  four-task mean over Agreement Reference (`0.1589` versus `0.1630`) but fails
  worst-task, task-non-inferiority, oracle-recall, localized-gain, and LOTO
  promotion checks.
- Shortlist attribution has now been corrected with explicit semantic
  controls. The legacy best-named
  `spectra_reliable_uw000_tw100_cs000_ct100_str_sf020` file actually uses
  `agreement_reference` as its shortlist owner. Its score maps and selected
  candidates are exactly reproduced by `agreement20_transfer_rerank` on all
  four open-development tasks. The corrected result remains `0.087507` mean
  normalized regret and `0.623427` mean selected Micro-F1. A genuine
  `spectra_cal` top-20% shortlist followed by Transfer Score is substantially
  worse (`0.261692` mean regret), and the Agreement/SPECTRA union is also worse
  (`0.246890`). The result artifact is
  `results/gda_select/open_dev/shortlist_attribution_objective_v2.json`.
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
  improvement is claimed from it yet. `spectra_cal.py --output-selector` now
  allows fixed-gamma and pair-consistency runs to be written under distinct
  selector names without overwriting the default `spectra_cal.json`.
- The full four-task covariance-gamma diagnostic is complete. Fixed gamma
  `.25` is the best global fixed control (`0.140279` mean NRegret), but it
  fails worst-task and non-inferiority guardrails. Candidate-level pair
  consistency chooses overly large gamma values and reaches `0.339996` mean
  regret. A trajectory-balanced variant is invariant to checkpoint
  duplication and improves this only to `0.320986`; it still does not track
  task-wise oracle gamma. Covariance consistency is therefore a no-go as a
  deployment trust signal. The authoritative reports are
  `results/gda_select/open_dev/covariance_gamma_sweep_objective_v2.json` and
  `results/gda_select/open_dev/covariance_trajectory_pair_consistency_objective_v2.json`.
- The covariance sweep now reuses immutable task-level candidate,
  calibration, and spectral-disagreement context. Its complete-pair projection
  uses the exact analytic inverse of `(M-2)I + 11^T` instead of dense least
  squares. Cached and direct gamma-zero scientific outputs are identical, and
  the complete 28-selector CPU sweep finishes in `449.26` seconds.
- Regret factorization is executable at
  `selector/selection_error_decomposition.py`. For the best candidate-level
  Agreement shortlist selector, `95.5%` of remaining normalized regret is
  trajectory gap and only `4.5%` is checkpoint gap. The first q=3
  trajectory-level shortlist control was nevertheless unsuccessful:
  Agreement-owned trajectory screening reaches `0.247994`, while
  Agreement/SPECTRA trajectory midrank reaches `0.164165`, both worse than the
  candidate-level `0.087507`. This control is frozen as a no-go result rather
  than tuned further.
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
  but this is direct-selector evidence rather than a frozen final method. After
  repairing rank-fusion ties and hard shortlist behavior, the best small-grid
  near-miss is Agreement-shortlist -> Transfer-Score rerank at 20%
  (`0.0875` mean normalized regret, `0.6234` selected Micro-F1). It still fails
  promotion because task non-inferiority is only `2/4` and oracle recall@20% is
  `0.50`, below the registered `0.75` shortlist guardrail. The leave-one-task-
  out diagnostic reaches the same conclusion: validation mean normalized regret
  is `0.0875`, but held-out task non-inferiority remains `0.50`. This is
  evidence for continued top-of-ranking calibration work, not a SOTA claim.
  It is not evidence that SPECTRA owns the successful shortlist.
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
- A GitHub Actions workflow is present locally at
  `.github/workflows/release-audit.yml` to run the lightweight public audit.
  It has not been pushed to the public `main` branch because the currently
  available GitHub token lacks the `workflow` scope required by GitHub for
  creating or updating Actions workflow files.

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
  selector/loto_open_dev_selection.py \
  selector/selector_complementarity.py \
  selector/consensus_selection.py \
  selector/check_reliable_inputs.py \
  selector/reliable_selection.py \
  selector/run_reliable_suite.py \
  selector/run_reliable_grid.py \
  tests/test_spectra_prior.py \
  tests/selector/test_objective_v2.py \
  tests/selector/test_loto_open_dev_selection.py \
  tests/selector/test_selector_complementarity.py \
  tests/selector/test_consensus_selection.py \
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
The same audit is wired into `.github/workflows/release-audit.yml` locally.
Pushing that workflow requires a GitHub token with `workflow` scope.

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

## Open items and current constraints

- The public GitHub `main` branch contains the current code, paper,
  open-development artifacts, final multi-selector package, repaired reliable
  selector code, and leave-one-task-out audit. The local `main` branch
  additionally contains `.github/workflows/release-audit.yml`; that one file is
  intentionally not on the public branch because pushing or updating GitHub
  Actions workflow files requires a token with `workflow` scope.
- The full 16-task reliable grid and final multi-selector package are generated
  packaging artifacts for external evaluation. They are not evidence that an
  "ours" grid may be selected after seeing final sealed scores.
- `selector/objective_v2.py` has been run on the real 675-candidate Gate-1
  open-development bank. The best current open-development single baseline
  signal is `agreement_reference` (`0.1630` mean normalized regret), and the
  best repaired reliable-selector near-miss is the misleadingly named
  `spectra_reliable_uw000_tw100_cs000_ct100_str_sf020` (`0.0875` mean
  normalized regret), whose actual shortlist owner is Agreement Reference.
  The semantic alias is `agreement20_transfer_rerank`. This remains
  open-development evidence only.
- The committee descriptor extractor is not yet wired into the default
  calibration bank. The descriptor metric diagnostic can score existing
  calibration descriptor keys, but the graph+committee+Transfer Score descriptor
  matrix still has to be generated on real calibration artifacts before it can
  drive covariance transport or shift-bank expansion.
- The reliable fusion code has been revised to fix known top-of-ranking
  calibration bugs, and a small repaired open-development grid has been
  regenerated/evaluated. The best repaired configuration is a near-miss, not a
  frozen final selector.
- The selector complementarity diagnostic has identified real ranking
  diversity between Transfer Score, Agreement Reference, `spectra_cal`, and the
  repaired reliable selector. A deployable label-free owner/consensus rule is
  still open: top-set overlap and rank correlation alone do not separate the
  Citation failures from the Airport wins.
- The first explicit Stage-B consensus attempt is also not ready to freeze.
  Its four-task mean is competitive, but LOTO validation shows that the
  apparent gain is not stable enough for sealed evaluation.
- Actual covariance-shrinkage sweeps are complete on the four open-development
  tasks. The manifest-backed run covers fixed gamma
  `{0, .25, .5, .75, 1}`, support gating, and pair consistency, with a separate
  trajectory-balanced follow-up. The full run has zero label accesses and
  protocol violations and remains open-development evidence only. Neither
  consistency rule is eligible for promotion; see the completed-result bullets
  above and `docs/RESULTS.md`.
- The final one-time 12 held-out transfer sealed evaluation is still pending.
  It should not be run until exactly one "ours" configuration is frozen from
  open-development evidence. Until that evaluator runs once on a frozen
  submission bundle, the paper must not claim state-of-the-art real-target
  selection.

## Required next commands before sealed evaluation

Re-run only the repaired open-development/reliable-selection path, then freeze
one "ours" selector before any sealed-final call:

```bash
python selector/run_reliable_grid.py \
  --selection-root /mnt/workspace/Wilson/AutoSOTA/baseline/ADAlign/results/gda_select/selections/gda_select_v1 \
  --spectra-root /mnt/workspace/Wilson/AutoSOTA/baseline/ADAlign/results/gda_select/selections/gda_select_v1 \
  --transfer-root /mnt/workspace/Wilson/AutoSOTA/baseline/ADAlign/results/gda_select/selections/gda_select_v1 \
  --output-root results/gda_select/selections/reliable_grid_repaired \
  --spectra-selector agreement_reference \
  --transfer-selector transfer_score \
  --uncertainty-weights 0 \
  --transfer-score-weights 0,0.5,1 \
  --covariance-shrinkages 0 \
  --calibration-temperatures 1 \
  --fusion-modes rank_fusion,transfer_shortlist_spectra_rerank,spectra_shortlist_transfer_rerank \
  --shortlist-fractions 0.2

python selector/objective_v2.py \
  --dev-truth-report results/gda_select/open_dev/gate1_candidate_truth.json \
  --selection-root /mnt/workspace/Wilson/AutoSOTA/baseline/ADAlign/results/gda_select/selections/gda_select_v1 \
  --selection-root results/gda_select/selections/reliable_grid_repaired \
  --selector transfer_score \
  --selector agreement_reference \
  --selector <candidate_selector_from_repaired_grid> \
  --objective-selector <candidate_selector_from_repaired_grid> \
  --transfer-selector transfer_score \
  --expected-candidate-count 675 \
  --runtime-budget-seconds 480 \
  --output results/gda_select/open_dev/repaired_reliable_objective_v2.json
```

Check v3 readiness before freezing or submitting:

```bash
python scripts/check_v3_readiness.py \
  --open-dev-truth results/gda_select/open_dev/gate1_candidate_truth.json \
  --open-dev-selection-root /mnt/workspace/Wilson/AutoSOTA/baseline/ADAlign/results/gda_select/selections/gda_select_v1 \
  --open-dev-selection-root results/gda_select/selections/reliable_grid_repaired \
  --final-submission-manifest results/gda_select/submissions/final_multi_selector/submission_manifest.json \
  --required-open-dev-selector transfer_score \
  --required-open-dev-selector <one_preselected_open_dev_winner> \
  --required-final-selector transfer_score \
  --min-final-selector-count 2
```

Freeze one pre-registered selector before sealed evaluation:

```bash
python selector/freeze_reliable_selector.py \
  --grid-root results/gda_select/selections/reliable_grid_repaired \
  --selector <one_preselected_open_dev_winner> \
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
