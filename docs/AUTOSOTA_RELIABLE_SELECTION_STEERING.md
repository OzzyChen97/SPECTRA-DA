# AutoSOTA steering: reliable risk selection

Use this steering for the next optimization run. The current bottleneck is not
spectral decomposition or NNLS design; it is whether covariance correction is
trustworthy under real target shifts and whether the final selector fixes
top-1 choice on real target development tasks.

```text
Do not optimize the source-simulated mean regret directly as the only target.
Do not add new shift types, new spectral filters, neural descriptor metrics, or
new recovery solvers in this run.
Use selector/objective_v2.py with an exported open-development candidate-truth
report for the four Gate-1 tasks. Do not import sealed_eval, read .sealed paths,
or query final 12-transfer labels during AutoSOTA.

Hypothesis:
The current SPECTRA risk estimate has useful ranking information but is too
aggressive at the top of the ranking when covariance transport is unreliable.
A conservative selector that penalizes transport uncertainty and fuses an
independent Transfer Score prior should improve top-1 real-target robustness.

Allowed knobs:
- covariance_shrinkage_mode
- fixed_covariance_gamma
- uncertainty_weight
- transfer_score_weight
- covariance_shrinkage
- calibration_temperature
- fusion_mode
- shortlist_fraction

Implementation path:
1. Treat the four Gate-1 tasks as open development and require a trusted
   evaluator to export candidate-level target errors for those tasks only.
2. Generate SPECTRA selection JSON files and Transfer Score selection JSON
   files for the same immutable candidate bank. Transfer Score can be produced
   with `selector/run_baseline_suite.py --selector transfer_score`.
3. Audit the pair with `selector/check_reliable_inputs.py`; reject missing
   inputs, candidate-bank mismatches, protocol violations, and missing
   uncertainty.
4. Use selector/run_reliable_grid.py to produce a pre-registered grid of
   spectra_reliable_* outputs, or selector/run_reliable_suite.py for a single
   frozen configuration.
   Include rank fusion, Transfer Score shortlist -> SPECTRA rerank, SPECTRA
   shortlist -> Transfer Score rerank, and support-adaptive mixture as separate
   registered fusion modes.
5. Score candidate selectors with `selector/objective_v2.py`, using primary
   mean normalized regret on open real-target development and guardrails for
   top-10 hit, worst task, source-sim CVaR, runtime, and zero label access.
6. Package the resulting selector root for one-time sealed evaluation.

Required diagnostics:
- selected candidate overlap with SPECTRA and Transfer Score;
- `reliable_grid_manifest.json` showing that only the allowed knobs varied;
- top-5% rank stability under source-family holdout;
- transport/source uncertainty distribution of the selected candidate;
- protocol audit: label_access_count=0 and protocol_violation_count=0;
- comparison against pure SPECTRA, pure Transfer Score, and global covariance.
- objective_v2 report showing Gate A/B status, top-weighted Kendall, NDCG@10,
  oracle/top-5 shortlist recall@5/10/20/30/50%, CVaR-20%, worst-task regret,
  worst-task delta versus Transfer Score, source-sim leave-one-shift-family
  CVaR guardrail status, and localized-gain share versus Transfer Score. Use
  shortlist recall to decide whether Transfer Score or SPECTRA should provide
  the first-stage shortlist before the other selector reranks it.

Promotion rule:
Promote only if objective_v2 confirms one of the following on the same
675-candidate open-development bank:
- Gate A: mean normalized regret below 0.20 and top-5% hit rate at least 50%;
- Gate B: mean normalized regret and selected Micro-F1 both beat Transfer Score;
- worst/CVaR improves by at least 10% without meaningful mean regression and
  top-10 hit does not fall below Transfer Score.

Never claim state of the art unless the frozen 16-transfer sealed evaluation
beats the strongest label-free baselines.
```
