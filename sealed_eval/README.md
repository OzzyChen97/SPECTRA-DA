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
