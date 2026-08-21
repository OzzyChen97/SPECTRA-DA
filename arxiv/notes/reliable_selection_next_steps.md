# Reliable selection next steps

> **Historical note, superseded by the completed Stage A/B/C audit.** The
> reliability, fusion, covariance-shrinkage, trajectory, coverage, and
> stability experiments proposed below were subsequently executed or replaced
> by stricter structural controls. The authoritative outcome is
> stop-no-freeze-no-sealed-evaluation.

This note records the next experimental direction after the current paper
repositioning. The current evidence does not support a SPECTRA-DA
state-of-the-art claim on real target selection: Transfer Score is stronger in
the Gate-1 real-target comparison (0.1467 versus 0.2560 mean normalized
regret). The supported finding is narrower and more useful: covariance-aware
risk recovery works on source-simulated folds, but covariance correction is not
reliably transported to real target shifts.

## Scientific hypothesis

The next method should not make the covariance model more expressive first.
Instead, it should decide when covariance correction is trustworthy and fall
back to conservative evidence when transport uncertainty is high.

Candidate score:

\[
\operatorname{score}_m
= \widehat R_m+\lambda U_m^{\mathrm{tr}}-\gamma T_m,
\]

where:

- \(\widehat R_m\): covariance-aware SPECTRA risk estimate;
- \(U_m^{\mathrm{tr}}\): transport uncertainty from donor/source-simulation
  disagreement in covariance or risk-correction space;
- \(T_m\): independent transferability prior, initially Transfer Score;
- \(\lambda,\gamma\): selected only by source leave-one-shift-family-out
  validation.

## Allowed search space

- `uncertainty_weight`
- `transfer_score_weight`
- `covariance_shrinkage`
- `calibration_temperature`
- rank-level fusion between SPECTRA risk and Transfer Score
- source-family-holdout validation objective

## Forbidden in the next iteration

- new graph spectral filters;
- new shift types;
- new recovery solver;
- neural descriptor metric learning;
- target-transfer-specific thresholds;
- any repeated query of hidden target labels.

## Required experiments

1. **Rank fusion with Transfer Score.** Test SPECTRA risk, Transfer Score, and
   rank fusion under a common frozen protocol. Primary question: does the real
   target top-1 regret move below 0.2560 and toward 0.1467 without hurting
   Kendall catastrophically?
2. **Transport uncertainty penalty.** Estimate donor disagreement in
   covariance or risk-correction space and add a conservative penalty. Primary
   question: does worst/top regret improve when aggressive corrections become
   unstable?
3. **Global covariance ablation.** Remove spectral conditioning and compare
   global covariance plus reliability against spectral covariance plus
   reliability. Primary question: is spectral still useful once reliability is
   handled?
4. **Leave-one-shift-family-out validation.** Tune all weights by holding out
   complete shift families, not random source-simulated folds. Primary question:
   does the method interpolate within known shifts only, or can it transfer to
   unseen shift families?

## Promotion rule

A reliability-aware variant should be promoted only if it satisfies at least
one of:

- real-target development mean normalized regret below the current SPECTRA-DA
  value 0.2560 without protocol violations;
- worst/CVaR reduction of at least 10% with no meaningful mean regression;
- source-family-holdout regret improvement with no degradation against
  Transfer Score rank fusion.

It must not be described as state of the art. The completed audit did not
freeze a selector, and the remaining twelve transfers were deliberately left
sealed.
