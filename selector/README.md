# Label-free selectors

Selectors may read public graphs, source-validation artifacts, and unlabeled
target predictions/embeddings. They must never import `sealed_eval`, read
workspace `.sealed` files, or emit target scores.

The pre-registered baselines are source validation, last checkpoint plus
source validation, target entropy, information maximization, agreement to a
committee reference, global committee disagreement, Soft Neighborhood Density
(SND), Generalization Disagreement Equality (GDE), and both released
Agreement-on-the-Line variants (ALine-S and ALine-D).
SND uses target softmax predictions with the published fixed temperature 0.05;
its exact pairwise entropy is evaluated in row blocks to bound memory. ALine
uses aligned source-validation predictions/labels and unlabeled target hard
predictions, applies the released [0.05, 0.98] pair filter, and fits in probit
space. Each output covers the full candidate bank and is bound to it by a
path-independent SHA-256 hash.

GDE is kept distinct from global committee disagreement: it compares only
independent seeds with the same method, configuration, and checkpoint epoch,
then uses their unlabeled-target disagreement as the candidate risk estimate.

Transfer Score is reconstructed from cached target embeddings/probabilities
and classifier-head weights in each immutable checkpoint. Hopkins sampling is
deterministically seeded by candidate ID; the score retains representation
transferability, normalized information maximization, and classifier geometry.
DEV is an artifact-availability adaptation: schema v1 does not cache
source-training embeddings, so its domain classifier uses source-validation
embeddings both as the source sample and as the weighted-risk sample. It keeps
the published 3000-sample cap, five-strength domain-classifier selection,
density-ratio weighting, and control-variate risk, but must be reported as
``DEV (source-val features)`` rather than a bitwise reproduction of DEV.

`objective.py` is the AutoSOTA-facing entrypoint. It evaluates covariance
transport with leave-one-simulated-shift-out folds using source-derived
calibration artifacts only. It never loads real target labels or calls
`sealed_eval`; the trusted target evaluator remains outside the optimization
loop. Active selector parameters are recorded in `configs/search_space.yaml`.

`spectra_cal.py` uses a support-aware prior-centered recovery rule. The
transported source-simulated risk is weighted at the pair-design curvature
scale `M-2` only when the standardized descriptor matching RMSE is at most 2;
otherwise it falls back to covariance-corrected disagreement recovery. This
two-sigma gate is observable without target labels and prevents unsupported
shift extrapolation from dominating model selection.

For attribution, `spectra_cal.py --spectral-mode global` collapses the same
tight-frame disagreements, simulated band risks, and simulated band
covariances into one global energy channel before transport and recovery.  It
therefore has the same candidate pool and calibration information budget as
the band-conditioned selector.  `run_spectra_suite.py` emits this control as
`spectra_global_cal` (or `spectra_global_robust` when uncertainty is active),
along with both hard- and soft-prediction Static sanity checks.  Claims about
spectral conditioning must compare against this global-calibration control,
not only against uncorrected global disagreement.

`shift_type_diagnostic.py` performs the stricter leave-one-shift-family-out
check over feature masking, feature noise, edge dropout, homophily, label-prior,
and conditional-structure shifts. It compares the original recovery, an
always-on curvature prior, a continuous match-adaptive prior, and the two-sigma
support gate. The diagnostic reads only frozen source-simulated calibration
artifacts and enforces GPU 7 plus zero target-label access.
