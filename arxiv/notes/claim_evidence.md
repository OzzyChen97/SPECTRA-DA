# Claim--Evidence Map

| Claim | Evidence | Status / wording boundary |
|---|---|---|
| Target-label-free algorithm, configuration, seed, and checkpoint selection is a distinct UGDA deployment problem. | Formal selector protocol, candidate schema, and ADAlign implementation audit. | Supported as a problem formulation; do not claim universal prevalence or priority without a broader audit. |
| GDA-Select enforces an auditable information boundary. | Candidate-bank hashes, isolated evaluator, public artifact schema, and zero access/violation counters. | Supported for the released protocol and reported runs. |
| Tight graph spectral frames exactly decompose hard classification error. | Lemma 1 and theory tests. | Supported analytically for hard one-hot predictions. |
| Pairwise spectral disagreement determines individual risks up to cross-model error covariance. | Exact identity and recovery theorem. | Supported analytically. |
| Covariance misspecification controls recovery error. | Recovery bounds and theory tests. | Supported for the stated linear core; practical constrained recovery is empirical. |
| Covariance correction is the main source-simulated recovery component. | Full 0.01415 versus 0.05057 without covariance; paired interval [0.01583, 0.06026]. | Supported on four tasks and 44 source-simulated folds. |
| Spectral conditioning consistently improves top-1 selection. | Full 0.01415 versus global-covariance 0.01471; only 4/44 choices differ and interval [-0.00073, 0.00265] crosses zero. | Not supported; describe the increment as modest and statistically inconclusive. |
| Spectral bands expose heterogeneous covariance regimes. | Absolute error correlation 0.381/0.573/0.617 in low/mid/high bands versus 0.487 globally. | Supported on the source-simulated calibration bank. |
| SPECTRA-DA is the best real-target selector. | SPECTRA-Cal mean NRegret 0.320062 on four 675-candidate open-development tasks. | Contradicted; do not make this claim. |
| Agreement@20% followed by Transfer Score is the strongest open-development diagnostic. | Mean NRegret 0.087507 versus Transfer Score 0.216807; selected Micro-F1 0.623427. | Supported only as open-development evidence; it is not SPECTRA and was not promoted. |
| The diagnostic is robust enough for final evaluation. | Worst NRegret 0.223881, non-inferior on 2/4 tasks, missing source-sim family-out CVaR guard. | Not supported; the audit correctly rejects promotion. |
| Candidate or trajectory coverage is the dominant remaining error. | Method and trajectory coverage gaps are both zero; checkpoint coverage is 20.5% of total. | Contradicted. |
| Shortlist-internal reranking is the dominant remaining error. | Mean normalized reranking gap 0.069580, or 79.5% of total 0.087507; USA-to-BRAZIL regret is entirely reranking. | Supported on the four open-development tasks. |
| Cross-fitting, coverage floors, trajectory evidence, gamma auxiliaries, or bootstrap stability repair the failure. | Every registered Stage-C control has worse mean regret than 0.087507; router qualifies on only 2/4 tasks. | Not supported; report as negative controls. |
| No target-label or protocol access occurred. | label_access_count=0 and protocol_violation_count=0 across released Stage-C outputs. | Supported. |
| A final selector was frozen and evaluated on all transfers. | Stage-C audit records freeze_allowed=false and sealed_final_evaluation_allowed=false. | Contradicted; final twelve transfers remain unqueried. |
| The paper establishes real-target SOTA. | No final sealed evaluation and no promoted selector. | Not supported; must not appear. |
| The cited literature is authentic and claim-aligned. | Citation audit checks arXiv and official proceedings/DOI records where available. | Supported for the current bibliography; preserve the audit boundary. |

## Reviewer-facing claim policy

- Use \emph{open-development} for the four real-target tasks.
- Use \emph{source-simulated} for the 44-fold recovery and covariance results.
- Attribute the 0.087507 diagnostic to Agreement screening and Transfer Score
  reranking, never to SPECTRA-DA.
- Treat SPECTRA-DA as a covariance-aware recovery theory and diagnostic module.
- State that spectral top-1 gains over global covariance are inconclusive.
- State the final audit decision explicitly: no freeze and no query of the
  remaining twelve transfers.
- Present 79.5% reranking error as the final failure localization.
- Do not describe additional tuning on the same four tasks as future evidence;
  future work requires a pre-registered protocol change or a new label-free
  signal.
