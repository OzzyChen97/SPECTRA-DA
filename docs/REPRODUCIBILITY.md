# Reproducibility guide

## Artifact boundary

Candidate checkpoints are immutable directories containing:

```text
metadata.json
source_val.npz
target_public.npz
model_state.pt
```

`source_val.npz` may contain source-validation labels. `target_public.npz`
must contain only logits, probabilities, hard predictions, and embeddings.
The schema validator rejects label-like target keys and binds artifacts to
SHA-256 hashes.

## Sidecar calibration

The final controlled variant augments the base source-simulated calibration
bank with pre-frozen source-labelled sidecars. These artifacts are not included
because they are large and machine-specific. Generate them locally, copy
`configs/refinement_sidecars.example.json` to
`configs/refinement_sidecars_v1.json`, then replace each path and hash.

The four-task refinement manifest must:

- cover all four frozen development tasks;
- report zero target-label access and zero protocol violations;
- match the candidate-bank ordering and spectral configuration;
- pass the source-node rank gate for feature-mask grids.

For frozen deployment, combine the development sidecars with the remaining
task sidecars and validate them against all 16 base calibration artifacts:

```bash
python scripts/build_sidecar_manifest.py \
  --calibration-root trajectory_bank/calibration/gda_select_v1 \
  --base-manifest configs/refinement_sidecars_v1.json \
  --sidecar-root /path/to/remaining/combined_sidecars \
  --sidecar-root /path/to/remaining/lifted_sidecars \
  --output configs/spectra_sidecars_16.json
```

The builder rejects incomplete task coverage, stale candidate ordering,
spectral-config or parent-calibration mismatches, artifact-hash mismatches,
target-label access, and protocol violations.

## Frozen refinement command

```bash
CUDA_VISIBLE_DEVICES=7 python selector/refinement_objective.py \
  --calibration-root trajectory_bank/calibration/formal \
  --sidecar-manifest configs/refinement_sidecars_v1.json \
  --reference-results results/reference/final24.json \
  --bootstrap-workers 4 \
  --output results/autosota_refinement.json
```

The reference result contains development-fold regrets only; it is used to
compute localized-gain diagnostics and must itself report zero label access.

## Expected aggregate output

The exact aggregate values are stored in `results/final_metrics.json`. Runtime
varies by host; the frozen rerun was `327.394s`, with a hard guardrail of
`480s`. All non-runtime values should match the release metrics when the same
candidate bank, calibration arrays, sidecars, package versions, and random
seeds are used.

## Frozen 16-task selector generation

After freezing the sidecar manifest and selector configuration, generate a new
immutable selection root. This step uses public target predictions and
source-only calibration artifacts, but never target labels:

```bash
CUDA_VISIBLE_DEVICES=7 python selector/run_spectra_suite.py \
  --candidate-root trajectory_bank/candidates/gda_select_v1 \
  --calibration-root trajectory_bank/calibration/gda_select_v1 \
  --sidecar-manifest configs/spectra_sidecars_16.json \
  --config configs/search_space.yaml \
  --output-root results/gda_select/selections/spectra_frozen_v2 \
  --device cuda:0
```

If public graph artifacts live outside the release checkout, set
`SPECTRA_PUBLIC_ROOT=/absolute/path/to/trajectory_bank/public`. The loader still
rejects any public graph containing a label field.

Do not overwrite earlier selector outputs. Hash and submit the complete frozen
selection root to a separately operated evaluator; the local development
boundary is not sufficient evidence for a final sealed result.

## Reliability-aware selector grid

The current real-target development comparison shows that Transfer Score is a
stronger top-1 selector than the frozen SPECTRA selector. The next
pre-registered direction is therefore a conservative post-selector that fuses
SPECTRA risk ranks, Transfer Score ranks, and transport uncertainty ranks
without reading candidate artifacts or target labels.

Before AutoSOTA v3 can optimize real-target top-1 behavior, the trusted
evaluator must explicitly convert the four already-inspected Gate-1 tasks into
an open development set. This exporter is evaluator-only, reads sealed labels
only for the Gate-1 tasks, and refuses non-Gate-1 tasks:

```bash
python sealed_eval/export_open_dev_truth.py \
  --candidate-root trajectory_bank/candidates/gda_select_v1 \
  --output results/gda_select/open_dev/gate1_candidate_truth.json \
  --expected-candidates-per-task 675 \
  --trusted-evaluator
```

Do not run this exporter for the remaining 12 final transfers. Those transfers
stay hidden until the one-time sealed evaluation.

Generate label-free committee descriptors for each candidate bank before
learning any descriptor metric. These descriptors use only target predictions,
probabilities, embeddings, and candidate metadata:

```bash
python shift_simulator/committee_descriptors.py \
  --candidate-root trajectory_bank/candidates/gda_select_v1 \
  --task ACMv9_to_Citationv1 \
  --output results/gda_select/descriptors/ACMv9_to_Citationv1_committee.json
```

Add `--include-transfer-components` only when classifier-head checkpoints are
available and Transfer Score component summaries are part of the registered
descriptor experiment. Do not expand the shift bank until committee descriptor
distance is shown to correlate with recovery-relevant correction distance.

Audit whether a calibration descriptor is actually useful for covariance
transport by measuring distance to the recovery-relevant correction vector,
not by graph-descriptor reconstruction error:

```bash
python shift_simulator/descriptor_metric.py \
  --calibration-dir trajectory_bank/calibration/gda_select_v1/ACMv9_to_Citationv1/CALIBRATION_ID \
  --descriptor-key shift_deltas \
  --target-descriptor-key target_delta \
  --output results/gda_select/descriptors/ACMv9_to_Citationv1_descriptor_metric.json
```

The report includes raw descriptor/correction Spearman correlation, learned
nonnegative diagonal metric weights, per-dimension descriptor audits, optional
target-to-bank distance contribution shares, and leave-one-shift-family
diagnostics. Median/MAD scaling with clipping is enabled by default; add
`--descriptor-transform signed_log1p` for signed heavy-tailed descriptors, or
`--no-robust-scale` only when reproducing the unnormalized diagnostic. Treat
this as a go/no-go check: do not let descriptors drive covariance transport or
active compositional shift expansion unless the learned metric shows meaningful
correction-distance correlation under family holdout.

First generate Transfer Score on the same immutable candidate bank used by the
frozen SPECTRA selector:

```bash
python selector/run_baseline_suite.py \
  --candidate-root trajectory_bank/candidates/gda_select_v1 \
  --output-root results/gda_select/selections/baselines \
  --selector transfer_score
```

The baseline root contains one selector JSON per task plus
`baseline_suite_manifest.json`, which records candidate-bank hashes, selected
candidates, selector coverage, cache reuse, and file hashes.

Then audit and generate the full allowed grid:

```bash
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
  --calibration-temperatures 1 \
  --fusion-modes rank_fusion,transfer_shortlist_spectra_rerank,spectra_shortlist_transfer_rerank,support_adaptive \
  --shortlist-fractions 0.1,0.2,0.3
```

The command writes one label-free selector JSON per task and configuration plus
`reliable_grid_manifest.json`. The manifest is the audit artifact proving that
only the allowed knobs changed: `uncertainty_weight`,
`transfer_score_weight`, `covariance_shrinkage`, and
`calibration_temperature`, `fusion_mode`, and `shortlist_fraction`.

Score the allowed grid on the open real-target development report:

```bash
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

`objective_v2.py` rejects `.sealed`, `sealed_eval`, and final-label paths. It
is the AutoSOTA-facing objective; the trusted exporter is not. In addition to
top-1 regret, F1 gap, NDCG@10, top-weighted Kendall, CVaR, and worst-task
metrics, the report includes oracle recall and oracle-top-5% recall at
5/10/20/30/50% predicted shortlist cutoffs. These fields are the diagnostic for
choosing between Transfer Score shortlist -> SPECTRA rerank and SPECTRA
shortlist -> Transfer Score rerank modes. A selector is promotion-ready only
when either Gate A or Gate B passes and all guardrails pass. The guardrails also
compare the candidate selector's worst open-development task regret against
Transfer Score's worst task regret; a selector with better average regret is
not promotion-ready if it creates a worse tail case. When `--source-sim-report`
is provided, the same guardrail block checks that source-simulated
leave-one-shift-family CVaR does not degrade by more than 5% relative to the
registered source-sim reference selector. If the report is omitted, this
guardrail is explicitly marked `not_evaluated` rather than silently passed.

Before treating the run as ready for freezing, audit that all required v3
artifacts exist and have the expected scope:

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

This readiness audit is label-safe. It refuses sealed/final-label paths and
checks that open-development truth is restricted to the four Gate-1 tasks, that
selector JSON files cover the required selectors on the 675-candidate bank, and
that the final submission manifest is a full-16 multi-selector package rather
than a SPECTRA-only package. The final-package check also verifies that every
declared selector has a packaged `selections/<task>/<selector>.json` file for
every one of the 16 tasks and that each file's SHA-256 digest matches the
manifest.

After source-family-holdout validation chooses one configuration, freeze only
that selector before sealed evaluation:

```bash
python selector/freeze_reliable_selector.py \
  --grid-root results/gda_select/selections/reliable_grid \
  --selector spectra_reliable_uw000_tw100_cs050_ct100 \
  --output-root results/gda_select/selections/reliable_frozen
```

Do not submit the entire grid to the sealed evaluator. The frozen root contains
one selector JSON per task plus `reliable_freeze_manifest.json`, which records
the chosen pre-registered configuration and file hashes.

For the final one-shot sealed comparison, package the frozen reliable selector
together with the baseline selector root. Repeating `--selection-root` lets the
packager merge baselines and our frozen selector while still enforcing identical
task coverage, duplicate-selector rejection, arg-opt consistency, and zero
label access:

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

The resulting `submission_manifest.json` also records hashes for any
`baseline_suite_manifest.json` or `reliable_freeze_manifest.json` found at the
supplied selection roots, binding the final package to the baseline-generation
and selector-freezing provenance. Packaging rejects any provenance manifest that
does not explicitly report zero label access and zero protocol violations.

## Hidden evaluation

Do not run hidden evaluation inside the optimization process. The sealed
evaluator should run once after the selector and protocol are frozen, ideally
under a separate account or service that exposes neither target labels nor
candidate-level target scores.
