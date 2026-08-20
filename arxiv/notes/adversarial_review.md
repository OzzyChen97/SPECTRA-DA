# Adversarial paper review

## 1. Contribution

- **Pass:** The paper isolates a meaningful deployment problem that is distinct from improving an adaptation loss.
- **Pass:** The exact tight-frame risk identity, pair-sum system, covariance bound, and global-equivalence boundary provide nontrivial technical insight.
- **Needs new experiment:** The protocol's broader scientific impact requires Gate-1 oracle gaps and method-ranking reversals across the full transfer set.
- **Needs new experiment:** “First” claims require a documented 2025--2026 literature audit and are intentionally absent from the draft.

## 2. Writing clarity

- **Pass:** Each method module has an explicit motivation, forward process, and technical advantage.
- **Pass:** Hard classification error and soft Brier-risk interpretations are separated.
- **Pass:** The practical NNLS/Tukey estimator is not incorrectly assigned the unconstrained theorem's guarantee.
- **Pass:** The original frozen selector baseline, the four-task real-target selector comparison, and the pending 16-task comparison are explicitly separated.
- **Pass:** The related-work section now covers the principal UGDA, UDA model-selection, unlabeled-accuracy, and graph-spectral lines with 27 claim-aligned citations.
- **Pass with audit note:** SPA's incorrect inherited arXiv identifier was fixed and an unverified SpecReg title was removed. Google Scholar was attempted but unreachable from the workspace; official proceedings/DOI pages provide the second verification signal.

## 3. Experimental strength

- **Pass within scope:** The relative mean-regret gain is large and top-of-ranking metrics are strong on 44 source-simulated folds.
- **Pass:** The strict component ablation reproduces the frozen 0.014145866 row and isolates covariance, spectral conditioning, robust refitting, and uncertainty.
- **Limitation exposed:** Covariance is essential, but the spectral increment over global covariance is modest and statistically inconclusive; robust refitting is not a source of gain.
- **Pass:** CVaR, worst-fold behavior, localized gains, runtime, and negative refinements are reported rather than hidden.
- **Needs new experiment:** Current folds are source-simulated and cover only Citation/Airport development tasks; no real-target 16-transfer superiority is established.
- **Needs new experiment:** Paired task-level confidence intervals and seed sensitivity are required for the final benchmark.

## 4. Evaluation completeness

- **Pass:** Strong general UDA selectors are implemented and compared under one protocol on four Gate-1 real-target tasks; Transfer Score is correctly reported as stronger.
- **Needs new experiment:** The full frozen 16-transfer comparison remains incomplete.
- **Pass with negative boundary:** Equal-information global covariance control is complete; it does not provide strong statistical support for a broad spectral top-1 gain.
- **Needs new experiment:** Graph-family holdout, same-method committee stress tests, and complete shift-family holdout remain necessary.

## 5. Method design soundness

- **Pass:** The information boundary is realistic and audited, although final host-level isolation should use a separate account or evaluator service.
- **Pass:** The method explicitly handles correlated candidates instead of assuming independence.
- **Pass:** The attribution proposition prevents crediting a tight-frame reparameterization as an empirical gain by itself.
- **Needs revision/new experiment:** Descriptor-to-covariance transport is the main technical weakness; SARC family-out failure demonstrates that the current descriptor geometry does not generalize.
- **Pass:** The paper frames this as a limitation and avoids a net-benefit or SOTA claim beyond the evidence.

## Submission verdict for the current draft

The draft is suitable as a technically honest working paper, but not yet as a
final ICLR submission. The blocking evidence is the externally operated frozen
16-transfer selector comparison and cross-family transport validation. The
equal-information attribution control is now complete and narrows the current
empirical contribution primarily to covariance-aware recovery.
