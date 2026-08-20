# Sealed evaluator

This directory is frozen and is the only repository code permitted to read
target labels. A selector must submit a complete score for every candidate and
declare zero label access and zero protocol violations. The evaluator verifies
the candidate-bank hash, recomputes target metrics, stores candidate-level truth
under the workspace-level `.sealed/spectra_da/evaluator_reports`, and writes
only aggregate metrics to the public report path.

The current filesystem separation is an audited development boundary, not an
operating-system security boundary. Final hidden-label evaluation must run in a
separate container, account, or service that exposes neither labels nor
candidate-level scores to AutoSOTA.

`export_open_dev_truth.py` is the one exception for the already-inspected
Gate-1 tasks. It is still evaluator-only and requires `--trusted-evaluator`,
but it intentionally exports candidate-level truth for the four open
development tasks so that `selector/objective_v2.py` can tune top-1 behavior
without querying the final 12 sealed transfers. It refuses non-Gate-1 tasks and
must not be used for final evaluation.
