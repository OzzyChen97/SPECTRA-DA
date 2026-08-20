# SPECTRA-DA Paper Outline

## One-sentence story

Unsupervised graph domain adaptation is not deployable when algorithms,
hyperparameters, and checkpoints are chosen with target labels; SPECTRA-DA
instead estimates candidate target risk from graph-spectral prediction
agreement and source-simulated error covariance under a sealed-label protocol.

## Introduction

- Define target-label-free model selection over a frozen UGDA candidate bank.
- Show why target-label oracle checkpoint selection is a distinct evaluation
  problem rather than a training-label leak.
- Identify the technical obstacle: pairwise disagreement determines individual
  risk only up to cross-model error covariance.
- Present the SPECTRA-DA pipeline: tight spectral decomposition, covariance
  transport from source-simulated shifts, robust risk recovery, and sealed
  evaluation.
- State only development-scope evidence: four tasks, 44 held-out simulated
  shifts, 675 candidates per task, and zero label/protocol violations.

## Related work

- Unsupervised graph domain adaptation and benchmark protocols.
- Label-free model selection in domain adaptation.
- Agreement-based classifier accuracy estimation.
- Graph spectral frames and graph signal decomposition.

## Method

- Problem definition and GDA-Select sealed-label protocol.
- Tight graph spectral frame and exact hard-error decomposition.
- Pairwise disagreement identity and covariance-controlled recovery bound.
- Shift-conditioned covariance transport.
- Robust weighted nonnegative recovery and uncertainty-aware selection.
- Complexity and attribution boundary versus global calibration.

## Experiments

- Development evaluation scope and frozen candidate bank.
- Baselines and metrics.
- Main development result and runtime.
- Source-only covariance-by-band diagnostic.
- Controlled refinement study and rejected variants.
- Failure analysis: cross-family covariance transport and SARC.
- Limitations and remaining 16-task sealed evaluation.

## Conclusion

- Reiterate model selection as a missing deployment layer for UGDA.
- Summarize the exact algebraic contribution and the current empirical evidence.
- State the generalization boundary and the required next evaluation.
