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
