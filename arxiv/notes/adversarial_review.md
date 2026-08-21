# Adversarial Paper Review

## 1. Contribution

- **Pass:** The paper defines an operationally meaningful UGDA model-selection
  problem over algorithms, configurations, seeds, and checkpoints.
- **Pass:** The exact tight-frame identity and covariance misspecification
  bounds provide nontrivial mathematical content.
- **Pass:** The pre-registered stopping audit and shortlist-specific
  decomposition provide a distinctive empirical contribution.
- **Risk:** A reviewer may consider the negative real-target result
  insufficient for a main-conference acceptance. The paper must lead with the
  benchmark and new knowledge, not imply a successful SOTA selector.

## 2. Writing clarity

- **Pass:** GDA-Select is consistently the benchmark/protocol; SPECTRA-DA is
  consistently the risk-recovery research module.
- **Pass:** The Agreement-to-Transfer-Score diagnostic is explicitly not
  labeled as ours.
- **Pass:** Source-simulated and real-target open-development evidence are
  separated.
- **Pass:** The final twelve sealed transfers are described as intentionally
  unqueried, not pending.

## 3. Experimental strength

- **Pass:** Fifteen selector rows are compared under the same 675-candidate
  open-development protocol.
- **Pass:** Core source-simulated attribution isolates covariance, spectral
  conditioning, robust refitting, and uncertainty.
- **Pass:** Stage A/B/C records negative controls rather than hiding them.
- **Limitation:** The best diagnostic is non-inferior on only 2/4 tasks and
  exceeds the worst-regret guard. No method-effectiveness claim may rely on its
  pooled mean alone.

## 4. Evaluation completeness

- **Pass:** Mean, worst, Micro-F1, rank quality, trajectory recall, LOTO,
  runtime, and protocol counters are reported.
- **Pass:** The final failure is decomposed into method, trajectory,
  checkpoint, and reranking terms.
- **Limitation:** Open development covers Citation and Airport but not Blog or
  Twitch. This is an explicit scope boundary rather than missing evidence that
  can be inferred away.
- **Pass:** The final evaluation was not run after the promotion audit failed.

## 5. Method design soundness

- **Pass:** The information boundary is realistic and auditable.
- **Pass:** The theory explicitly handles correlated candidates rather than
  assuming independence.
- **Pass:** Global-equivalence analysis prevents attributing gain to spectral
  reparameterization alone.
- **Limitation:** Source-to-real covariance transport is unreliable; current
  descriptors do not predict correction geometry across families.
- **Pass:** The paper treats this as a technical defect of the deployment
  module and does not claim positive net value for SPECTRA-DA as a selector.

## Claim challenges a skeptical reviewer will raise

1. **Why publish a method that is not promoted?**
   The paper's contributions are the auditable benchmark, covariance-aware
   recovery analysis, and a pre-registered negative finding that localizes the
   remaining error. The title and abstract must preserve this framing.
2. **Is 0.087507 simply overfitting four tasks?**
   Possibly; this is why the result is called a diagnostic, fails promotion,
   and is not evaluated on the twelve sealed transfers.
3. **Does the spectral frame itself improve selection?**
   The equal-information control says only modestly and inconclusively. The
   supported role is decomposition and covariance heterogeneity analysis.
4. **Why not keep tuning until the worst task improves?**
   Because the registered search space was exhausted and further feedback from
   the same four labels would be leaderboard overfitting.
5. **What would constitute future evidence?**
   A pre-registered broader development split or a new shortlist-internal
   calibration signal validated without querying the current sealed set.

## Current submission verdict

The draft is coherent as a benchmark/theory/audit paper and no longer makes an
unsupported SOTA claim. Its principal acceptance risk is venue fit: reviewers
must value the protocol and negative finding as contributions despite the lack
of a promoted selector. The paper should therefore keep the Stage-C stopping
decision, failure decomposition, and claim boundary highly visible.
