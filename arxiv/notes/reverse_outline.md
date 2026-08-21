# Reverse Outline

## Abstract

- Opening: deployment requires target-label-free model selection in addition
  to UGDA training.
- Benchmark: GDA-Select seals labels and audits candidate trajectories.
- Theory: spectral disagreement recovers risk only through an unknown
  covariance term.
- Source-simulated evidence: covariance correction reduces mean regret from
  0.05057 to 0.01415; spectral increment is inconclusive.
- Real-target evidence: Agreement@20% to Transfer Score reaches 0.087507 but is
  not SPECTRA and fails three promotion requirements.
- Final message: 79.5% of residual regret is reranking; no selector is frozen
  and twelve transfers remain sealed.

## Introduction

1. UGDA training advances do not answer deployable model selection.
2. Target-oracle checkpoint reporting creates an evaluation mismatch.
3. Correlated trajectory errors and top-of-ranking calibration make the task
   technically difficult.
4. GDA-Select supplies the protocol; SPECTRA-DA supplies a recovery analysis.
5. The pre-registered evidence supports benchmark/theory but not a promoted
   selector.
6. Contributions are benchmark, covariance-aware theory, and audited failure
   localization.

## Method

1. Define the frozen candidate bank and sealed-label information boundary.
2. Construct a tight target-graph spectral frame.
3. Derive exact hard-error decomposition.
4. Recover individual band risks from disagreement up to covariance.
5. Bound risk-recovery error under covariance misspecification.
6. Estimate covariance from labeled source simulations.
7. State practical recovery and uncertainty accounting.
8. State the global-equivalence attribution boundary.

## Experiments

1. Define four open-development tasks and 675-candidate trajectories.
2. Compare 15 label-free selectors under one protocol.
3. Attribute source-simulated recovery to covariance correction.
4. Show spectral covariance heterogeneity but inconclusive incremental top-1
   gain.
5. Demonstrate simulation-to-real transport failure.
6. Report Stage A/B/C structural no-go controls.
7. Decompose the strongest diagnostic's regret into 20.5% checkpoint coverage
   and 79.5% reranking.
8. Apply all registered promotion guards and stop without querying final
   labels.

## Conclusion

1. Restate protocol and risk-recovery contributions.
2. Separate source-simulated success from real-target deployment failure.
3. Identify shortlist-internal calibration as the unresolved problem.
4. Preserve the final twelve sealed transfers and require a new pre-registered
   evidence base for future work.
