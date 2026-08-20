# Claim--Evidence Map

| Claim | Evidence | Status / wording boundary |
|---|---|---|
| Target-label-free algorithm, hyperparameter, and checkpoint selection is a distinct UGDA problem. | Formal selector protocol and audited candidate schema. | Supported as a problem formulation; do not claim first without literature audit. |
| Tight graph spectral frames exactly decompose hard classification error. | Lemma 1 and unit tests in `tests/test_spectral_risk_recovery.py` and `tests/selector/test_spectra_theory_guarantees.py`. | Supported analytically for hard one-hot predictions. |
| Pairwise spectral disagreement determines individual band risks up to cross-model error covariance. | Exact identity and Theorem 1. | Supported analytically. |
| Covariance misspecification controls recovery error. | Theorems 1--2 and theory tests. | Supported for the stated unconstrained least-squares core; practical NNLS/Tukey behavior is empirical. |
| Spectral bands expose heterogeneous covariance regimes. | 176 source-simulated leave-one-shift-out comparisons: absolute error correlation 0.381/0.573/0.617 in low/mid/high bands versus 0.487 global. | Supported on the source-simulated calibration bank; do not claim every band decorrelates errors. |
| The frozen refinement selector improves normalized regret over its original frozen selector baseline. | Mean 0.014145866 versus 0.022784678, a 37.915% relative reduction, on four development tasks and 44 folds. | Supported only in this development scope; this is not yet the complete comparison against all label-free selectors. |
| Covariance correction is the main empirical source of the source-simulated gain. | Strict core ablation: full 0.01415 versus 0.05057 without covariance; paired mean-difference bootstrap interval [0.01583, 0.06026]. | Supported on four tasks and 44 source-simulated folds. |
| Band-conditioned covariance improves selection over an equal-information global covariance control. | Full 0.01415 versus global 0.01471 (3.83% relative), but only 4/44 selections differ, paired interval [-0.00073, 0.00265], Wilcoxon p=1.0. | Directionally positive but statistically inconclusive; claim only modest incremental evidence. |
| Robust refitting or bootstrap uncertainty drives the result. | No-robust+UCB obtains 0.01397; no-UCB obtains 0.01452; each changes at most four selections versus full. | Not supported as a contribution; retain only as frozen implementation details. |
| The selector ranks candidates accurately. | Median Kendall 0.9331, mean Spearman 0.9820, top-weighted Kendall 0.9946, top-5% hit 95.45%. | Supported on the same development scope. |
| SPECTRA-DA beats strong label-free selectors on real targets. | Gate-1 four-task comparison: Transfer Score 0.1467 mean normalized regret versus SPECTRA-Cal 0.2560; SPECTRA-Cal has higher median Kendall (0.695 versus 0.356). | Not supported for top-1 selection; report Transfer Score as stronger and localize the failure to top-of-ranking calibration. |
| The deployment selector does not access target labels. | `label_access_count=0`, `protocol_violation_count=0`. | Supported by the frozen audit for the reported development run. |
| Runtime optimization preserves scientific output. | Iterations 2 and 8 preserve mean regret exactly while reducing runtime from 452.0 s to 320.7 s; frozen rerun is 327.4 s. | Supported. |
| Descriptor-conditioned risk-correction transport generalizes across shift families. | SARC family-out mean regret 0.06977 and descriptor/correction Spearman -0.04954. | Not supported; present as a failure/limitation. |
| SPECTRA-DA is SOTA over 16 real transfers. | Requires a frozen comparison against all strong selectors on all 16 transfers. | Not yet supported; must not appear as a factual claim. |
| Target-label oracle selection changes UGDA method rankings broadly. | Requires Gate-1 oracle gap and ranking reversal results across the full task set. | Pending; motivate using the ADAlign implementation audit without generalizing prevalence. |
| The cited literature is authentic and claim-aligned. | `notes/citation_audit.md`: 27 cited entries checked against arXiv where available and official proceedings/DOI pages; one wrong arXiv ID corrected and one unverified inherited title removed. | Supported for the current bibliography. Google Scholar endpoints were attempted but timed out from this environment, which is recorded explicitly rather than treated as a successful check. |

## Reviewer-facing claim policy

- Use “development evaluation” whenever reporting the current 37.915% gain.
- Use “source-simulated” for covariance-by-band and shift-family diagnostics.
- Reserve “final sealed target evaluation” for the future externally operated
  one-time 16-transfer run; call the existing four-task result Gate-1
  development evaluation.
- Attribute the dominant source-simulated gain to covariance correction, not
  spectral decomposition, robust fitting, or uncertainty.
- Describe spectral conditioning as modest and statistically inconclusive for
  top-1 selection on the current folds.
- Describe SARC and delete-one donor variants as rejected diagnostics, not parts
  of the final method.
- Do not restore the removed unverified SpecReg title or the incorrect SPA
  arXiv identifier.
