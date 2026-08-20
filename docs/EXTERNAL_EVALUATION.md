# External sealed evaluation contract

The final 16-transfer result must not be queried from the development
workspace. After code, calibration sidecars, and selector outputs are frozen,
build a submission bundle:

```bash
python scripts/package_external_evaluation.py \
  --candidate-root trajectory_bank/candidates/gda_select_v1 \
  --selection-root results/gda_select/selections/spectra_frozen_v2 \
  --output-root results/gda_select/submissions/spectra_frozen_v2 \
  --archive results/gda_select/submissions/spectra_frozen_v2.tar.gz \
  --selector spectra_robust
```

The packager verifies all 16 task banks, exact candidate-score coverage,
arg-opt consistency, identical selector coverage across tasks, zero label
access, zero protocol violations, absence of target-truth fields, and every
file hash. It intentionally does not include candidate artifacts or labels, and
redacts machine-local absolute paths from submitted selector diagnostics.

For fast packaging after a separate trajectory export audit, add:

```bash
  --metadata-only-candidate-check
```

This mode checks candidate metadata, artifact presence, candidate-bank hashes,
and selector score coverage without reopening every large checkpoint and NumPy
artifact. The submission manifest records the candidate validation mode, so the
external evaluator can decide whether to require a stricter local rehash before
opening the sealed labels.

For the reliability-aware direction, first freeze exactly one pre-registered
grid configuration:

```bash
python selector/freeze_reliable_selector.py \
  --grid-root results/gda_select/selections/reliable_grid \
  --selector spectra_reliable_uw000_tw100_cs050_ct100 \
  --output-root results/gda_select/selections/reliable_frozen

python scripts/package_external_evaluation.py \
  --candidate-root trajectory_bank/candidates/gda_select_v1 \
  --selection-root results/gda_select/selections/reliable_frozen \
  --output-root results/gda_select/submissions/reliable_frozen \
  --archive results/gda_select/submissions/reliable_frozen.tar.gz \
  --selector spectra_reliable_uw000_tw100_cs050_ct100
```

The full reliable grid is a development artifact and must not be submitted for
repeated sealed scoring.

For a fair final comparison, package one frozen selection root that contains all
pre-registered selectors to be evaluated on the same immutable candidate bank.
At minimum this root should contain Transfer Score and the frozen SPECTRA
variant; include additional baselines only if their selector JSON files were
generated before submission lock:

```bash
python selector/run_baseline_suite.py \
  --candidate-root trajectory_bank/candidates/gda_select_v1 \
  --output-root results/gda_select/selections/baselines \
  --selector transfer_score \
  --selector entropy \
  --selector infomax \
  --selector source_val \
  --selector last_source_val \
  --selector snd \
  --selector dev \
  --selector gde \
  --selector aol_s \
  --selector aol_d
```

The baseline suite writes
`results/gda_select/selections/baselines/baseline_suite_manifest.json`; verify
this manifest before freezing the final multi-selector submission.

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

When `--selector` is omitted, the packager includes every selector JSON present
under all supplied selection roots for each task and rejects duplicate selector
names or inconsistent selector coverage across tasks. This is the required mode
for a one-time sealed comparison against Transfer Score; a package containing
only `spectra_robust` cannot support a SOTA claim. The final comparison command
therefore requires `transfer_score` and at least two selectors at packaging
time, before the readiness audit runs. If a supplied selection root
contains `baseline_suite_manifest.json`, `reliable_freeze_manifest.json`, or
`reliable_grid_manifest.json`, the submission manifest records that provenance
file and its SHA-256 hash. Provenance manifests must explicitly report
`label_access_count=0` and `protocol_violation_count=0`; otherwise packaging is
rejected.

Before submitting the package, run the v3 readiness audit:

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

The readiness audit must report `"ok": true`. It is intentionally label-safe:
it checks exported open-development scope, selector coverage, candidate counts,
arg-opt consistency, and whether the final manifest is a full-16
multi-selector submission, but it refuses `.sealed`, `sealed_eval`, and
final-label paths.

The receiving evaluator must independently possess the frozen candidate bank
and hidden labels. It must:

1. verify the candidate-bank and selection-file hashes;
2. permit a single evaluation after submission lock;
3. retain labels and candidate-level truth privately;
4. return only pre-registered aggregate and task-level reports;
5. record the evaluator code/version and immutable submission hash.

Development code must not receive repeated feedback from this service. A local
filesystem permission boundary is useful for auditing but is not accepted as
the final isolation mechanism.
