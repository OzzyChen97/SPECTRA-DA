# GDA-Select Paper Outline

## One-sentence story

Target-label-free UGDA selection requires both a mathematical account of what
unlabeled model disagreement can reveal and an evaluation protocol that stops
when the evidence is not robust; GDA-Select provides both, and the resulting
audit localizes the unresolved error to shortlist-internal reranking.

## Introduction

- Separate adaptation training from algorithm/configuration/seed/checkpoint
  selection.
- Audit target-oracle checkpoint reporting as a deployment mismatch.
- Explain the core obstacle: disagreement determines risk only up to
  cross-model error covariance.
- Present GDA-Select as the protocol and SPECTRA-DA as the recovery research
  module.
- State the evidence boundary: source-simulated recovery succeeds, real-target
  covariance transport does not.
- State three contributions: benchmark, theory, and pre-registered negative
  audit with failure localization.

## Method

- Formal target-label-free selection and sealed information boundary.
- Tight graph spectral frame and exact hard-error decomposition.
- Pairwise disagreement identity and covariance misspecification bounds.
- Source-simulated covariance transport.
- Practical weighted recovery and uncertainty accounting.
- Attribution boundary: decomposition alone is equivalent to global recovery.

## Experiments

- Four open-development transfers, 675 candidates per task, twelve final
  transfers still sealed.
- Equal-protocol comparison against strong label-free selectors.
- Source-simulated covariance attribution and spectral heterogeneity.
- Simulation-to-real transport failure.
- Stage A/B/C no-go audit.
- Shortlist decomposition: 20.5% checkpoint coverage, 79.5% reranking.
- Promotion audit and no-freeze/no-query decision.

## Conclusion

- The benchmark and recovery theory are supported.
- A deployable SPECTRA selector is not supported.
- The remaining scientific problem is shortlist-internal top-ranking
  calibration under cross-family shift.
- Further work requires a pre-registered change to the evidence base, not more
  tuning on the same four tasks.
