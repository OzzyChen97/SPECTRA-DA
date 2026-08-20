# Method module map

| Module | Workflow | Why needed | Why it works / evidence |
|---|---|---|---|
| Sealed candidate bank | Freeze method × config × seed × checkpoint outputs; expose no target-label field; select one candidate. | Separates adaptation training from deployable selection and prevents target-oracle checkpoint choice. | Hash-bound artifacts, separate evaluator, and access audit make the boundary testable. |
| Tight spectral frame | Filter candidate predictions with normalized spectral windows implemented by Chebyshev polynomials. | Global disagreement merges smooth community errors and high-frequency boundary errors. | Tightness preserves total signal energy, giving an exact hard-error decomposition. |
| Pair-sum risk recovery | Form all observable pair disagreements per band and solve for nonnegative individual risks. | Individual target errors are hidden. | The pair-incidence design has minimum singular value `sqrt(M-2)` for a complete committee. |
| Covariance transport | Simulate labeled source shifts; measure band error covariance; match unlabeled descriptors to the target. | Related checkpoints share systematic errors, invalidating zero-covariance recovery. | The recovery bound depends directly on covariance estimation error. |
| Robust selection | Apply reliability weights, a support-aware prior, one Huber refit, and bootstrap UCB. | Descriptor matching and individual pair observations can be unreliable. | Downweights correlated/anomalous pairs and penalizes unstable low-risk estimates. |

