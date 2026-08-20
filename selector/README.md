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
scale `M-2` only when the standardized descriptor matching RMSE is at most 2.
By default, the covariance correction remains the frozen v2 behavior. For the
v3 trust experiment, `--covariance-shrinkage-mode support_gate` applies the
same label-free support test directly to transported covariance:
`covariance <- gamma * covariance`, where `gamma=1` inside descriptor support
and `gamma=0` outside it. `--covariance-shrinkage-mode fixed` exposes the
controlled gamma sweep used by shrinkage ablations. The experimental
`pair_consistency` mode chooses `gamma` from `{0, .25, .5, .75, 1}` by minimizing
the label-free residual after projecting `disagreement + 2 * gamma *
covariance` onto the pair-sum risk subspace. These modes make the support gate
act on the hidden correlated-error term, not only on the risk prior, while
preserving the released selector when shrinkage mode is `none`.

The deployment selector optionally accepts a frozen source-only sidecar
manifest. Every sidecar is hash-checked against the base candidate ordering,
spectral configuration, and parent calibration before its simulated risks and
covariances are merged. `scripts/build_sidecar_manifest.py` validates complete
16-task coverage; it does not read target labels.

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

`check_reliable_inputs.py`, `reliable_selection.py`, `run_reliable_suite.py`,
`run_reliable_grid.py`, and `freeze_reliable_selector.py` implement the next
conservative selector direction. They consume already generated SPECTRA and
Transfer Score selection JSON files for the same candidate bank, convert each
score vector to tie-aware percentile midranks, and emit new minimized
rank-fusion scores:

```text
score = (1 - spectra_rank_shrinkage) * rank(SPECTRA)
      + transfer_score_weight * rank(Transfer Score)
      + uncertainty_weight * rank(transport/source uncertainty)
```

The CLI keeps `--covariance-shrinkage` as a backward-compatible argument name,
but this post-hoc rank fusion step does not recompute `d + 2 gamma c`. Actual
covariance shrinkage must be produced inside `spectra_cal.py` and recorded in
the SPECTRA selector diagnostics.

The same post-selector also supports the two directional shortlist controls
from the v3 plan: Transfer Score shortlist followed by SPECTRA reranking, and
SPECTRA shortlist followed by Transfer Score reranking. The `support_adaptive`
mode reads the label-free covariance gamma stored by SPECTRA diagnostics and
uses it as the normalized mixture weight between SPECTRA rank and Transfer
Score rank, falling back to `1 - covariance_shrinkage` for older SPECTRA JSON
files. Shortlist/rerank modes use a hard exclusion barrier outside the selected
shortlist, so an out-of-shortlist candidate cannot win because of score-scale
artifacts.

This post-selector reads no candidate artifacts and no target labels. Its
parameters must be selected by source leave-one-shift-family-out validation or
another pre-registered label-free proxy before a one-time sealed evaluation.
The preflight checker rejects missing inputs, candidate-bank mismatches,
protocol violations, and missing uncertainty when `--require-uncertainty` is
enabled. The reliable runners accept either a single combined `--selection-root`
or separate `--spectra-root` and `--transfer-root` directories, which avoids
copying baseline JSON files into the frozen SPECTRA output root.
The grid runner writes `reliable_grid_manifest.json` so an external evaluator
can verify that only the allowed knobs were varied.
After source-family-holdout validation chooses one configuration,
`freeze_reliable_selector.py` copies only that registered selector into a clean
root with `reliable_freeze_manifest.json`; the full grid is not a sealed
submission artifact.

`core_ablation.py` evaluates the frozen four-task source-simulated folds under
equal-information controls. It verifies global/spectral linear equivalence and
separately removes covariance correction, band conditioning, robust refitting,
and bootstrap uncertainty. The frozen aggregate is stored in
`results/core_ablation.json`; it shows that covariance correction is the main
gain, while the spectral top-1 increment is modest on the current folds.
