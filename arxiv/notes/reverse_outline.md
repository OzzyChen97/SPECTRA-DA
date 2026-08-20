# Reverse outline

## Abstract

- Problem: deployable UGDA requires target-label-free selection.
- Challenge: pair disagreement is confounded by shared error covariance.
- Method: sealed trajectory protocol plus tight-frame risk recovery and source-only covariance transport.
- Evidence: four tasks, 44 simulated folds, 37.9% versus the original frozen selector, zero access violations.
- Evidence boundary: covariance drives the simulated-fold gain, spectral conditioning is modest, and Transfer Score wins the four-task real-target comparison.
- Final boundary: the externally operated 16-transfer comparison remains pending.

## Introduction

1. UGDA training advances do not answer deployable model selection.
2. Public target-oracle checkpoint reporting exposes the evaluation gap.
3. Shared classifier errors make agreement insufficient.
4. SPECTRA-DA decomposes and covariance-corrects graph-structured disagreement.
5. Contributions combine protocol, theory, development evidence, and an explicit limitation.

## Related work

1. UGDA progresses from adversarial alignment to structure-, propagation-, attribute-, homophily-, and spectrum-aware training; this paper selects frozen candidates.
2. DEV, SND, Transfer Score, GDE, and ALine motivate strong label-free baselines but omit target adjacency and the joint candidate-bank protocol.
3. Spectral meta-learning and moment estimators expose the restrictive error structure behind unlabeled accuracy estimation.
4. Graph signal processing, spectral wavelets, tight frames, and Chebyshev filtering support the construction; band-conditioned calibration creates the selection signal.

## Method

1. Define frozen-bank selection and sealed labels.
2. Build a tight target-graph spectral frame.
3. Derive exact hard-error decomposition.
4. Recover individual risks from observable pair disagreement up to covariance.
5. Bound recovery error by covariance misspecification.
6. Transport covariance from source-simulated shifts.
7. State practical NNLS recovery and optional robust/bootstrap refinements.
8. State global-equivalence attribution boundary and complexity.

## Experiments

1. Separate the four-task, 140-candidate real-target baseline comparison from the 675-candidate source-simulated refinement study.
2. Report that Transfer Score beats SPECTRA-Cal in Gate-1 top-1 selection while SPECTRA-Cal has stronger global Kendall.
3. Reproduce the 37.9% source-simulated refinement gain and isolate covariance as its dominant component.
4. Show exact global/spectral linear equivalence, modest spectral-conditioned selection gain, and band-wise covariance heterogeneity.
5. Demote robust fitting and uncertainty using strict ablations.
6. Expose SARC/cross-family transport failure and the pending external 16-task benchmark scope.

## Conclusion

1. Separate adaptation training from deployable selection.
2. Summarize exact risk decomposition and development evidence.
3. Identify cross-family transportability as the next scientific problem.
