# Controlled refinement results

The frozen selector achieves mean normalized regret `0.0141458660`, a
`37.915%` reduction relative to the original `0.0227846777` baseline. It does
not reach the stricter internal target `0.0134`.

This is a source-simulated development result, not a real-target
state-of-the-art claim. In the four-task Gate-1 real-target development
comparison, Transfer Score remains the strongest top-1 selector (`0.1467`
mean normalized regret versus `0.2560` for SPECTRA-DA). The main empirical
conclusion is therefore that covariance correction is useful when calibrated,
but transport reliability under real shift is unresolved.

## 675-candidate open-development update

The current open-development evidence is based on the four pre-registered
Gate-1 tasks with 675 candidates per task. It is not a sealed-final result.

| Selector | Mean NRegret | Worst NRegret | CVaR-20% | Mean selected Micro-F1 | Top-10% hit |
|---|---:|---:|---:|---:|---:|
| Transfer Score | 0.216807 | 0.641791 | 0.641791 | 0.575664 | 0.500 |
| Agreement Reference | 0.163016 | 0.222591 | 0.222591 | 0.594958 | 0.250 |
| Agreement shortlist -> Transfer Score rerank @20% | 0.087507 | 0.223881 | 0.223881 | 0.623427 | 0.750 |

The repaired shortlist/rerank selector is a strong open-development
near-miss: it improves mean normalized regret by `59.64%` relative to Transfer
Score and improves selected Micro-F1 by `0.04776`, but it is not promotion-ready
because only `2/4` tasks are no worse than Transfer Score and oracle
recall@20% is `0.50`, below the registered `0.75` shortlist guardrail. It must
not be treated as the frozen "ours" selector for sealed evaluation.

The leave-one-task-out selector-choice diagnostic reaches the same conclusion:
when each fold selects a configuration using only the other three open-dev
tasks, it repeatedly selects the same Agreement-shortlist -> Transfer-Score
rerank variant. Validation mean normalized regret remains `0.087507`, but the
held-out task non-inferiority rate is still `0.50` and oracle recall@20% is
still `0.50`. The gain is therefore real but family-localized rather than a
stable four-task promotion signal.

### Shortlist attribution correction

The earlier `spectra_reliable_uw000_tw100_cs000_ct100_str_sf020` name was
misleading: its frozen `fusion_config` identifies `agreement_reference` as the
actual shortlist owner. The explicit semantic reconstruction
`agreement20_transfer_rerank` has identical candidate-score maps and selects
the identical candidate on all four open-development tasks. The strongest
`0.087507` result must therefore be attributed to Agreement screening followed
by Transfer Score reranking, not to SPECTRA screening.

`results/gda_select/open_dev/shortlist_attribution_objective_v2.json` records
the corrected controls:

| Shortlist control | Mean NRegret | Worst NRegret | Mean selected Micro-F1 | Oracle recall@20% |
|---|---:|---:|---:|---:|
| Agreement@20% -> Transfer Score | 0.087507 | 0.223881 | 0.623427 | 0.500 |
| SPECTRA-Cal@20% -> Transfer Score | 0.261692 | 0.641791 | 0.544321 | 0.000 |
| Agreement/SPECTRA union@20% -> Transfer Score | 0.246890 | 0.641791 | 0.554043 | 0.250 |

The Agreement/SPECTRA top-20% intersection is also not a viable universal
control: it contains 26, 34, and 13 candidates on the first three open tasks,
but is empty on `BRAZIL_to_USA`. The implementation rejects an empty
intersection instead of silently falling back to another selector. These
results materially narrow the claim boundary: the current best deployment
heuristic is committee-screened Transfer Score, while a uniquely spectral
shortlist benefit has not been demonstrated.

## Selector complementarity diagnostic

`results/gda_select/open_dev/selector_complementarity_diagnostic.json` records
the label-free ranking overlap between Transfer Score, Agreement Reference,
`spectra_cal`, and the best repaired reliable selector on the same four
675-candidate Gate-1 tasks. It reads selector JSON files and already exported
open-development metrics only; it reports `label_access_count: 0` and
`protocol_violation_count: 0`.

The main finding is that the selectors are genuinely complementary, but the
current label-free overlap signals do not yet identify a safe owner:

| Pair | Mean rank Spearman | Mean top-20% Jaccard | Same selected rate |
|---|---:|---:|---:|
| Transfer Score vs Agreement Reference | 0.1354 | 0.0330 | 0.000 |
| Transfer Score vs repaired reliable selector | 0.1676 | 0.0331 | 0.000 |
| Transfer Score vs `spectra_cal` | 0.4849 | 0.3061 | 0.000 |
| Agreement Reference vs repaired reliable selector | 0.9907 | 0.9927 | 0.000 |

Transfer Score wins the two Citation transfers, while the repaired selector
wins the two Airport transfers. The low Transfer-Score/Agreement overlap
supports the hypothesis that the methods capture different top-of-ranking
signals; however, low overlap occurs on both wins and losses, so it is not by
itself a deployable trust rule.

## Stage-B consensus selectors

To avoid another large near-duplicate grid, `selector/consensus_selection.py`
implements two explicit Stage-B controls:

- `ts20_agreement_rerank`: Transfer Score top-20% shortlist followed by
  Agreement Reference reranking.
- `ts20_spectra_agreement_consensus`: Transfer Score top-20% shortlist followed
  by the mean of tie-aware SPECTRA and Agreement percentile midranks.

Open-development results are stored in
`results/gda_select/open_dev/stage_b_consensus_objective_v2.json`.

| Selector | Mean NRegret | Worst NRegret | Mean selected Micro-F1 | Top-10% hit |
|---|---:|---:|---:|---:|
| Transfer Score | 0.216807 | 0.641791 | 0.575664 | 0.500 |
| Agreement Reference | 0.163016 | 0.222591 | 0.594958 | 0.250 |
| TS@20% -> Agreement rerank | 0.200759 | 0.492537 | 0.579332 | 0.250 |
| TS@20% -> SPECTRA/Agreement consensus | 0.158871 | 0.313433 | 0.595910 | 0.250 |

The consensus variant slightly improves mean normalized regret over Agreement
Reference, but it is still not promotion-ready: worst-task regret exceeds the
`0.30` absolute guardrail, task non-inferiority versus Transfer Score remains
`0.50`, oracle recall@20% remains `0.50`, and localized-gain share is `0.884`.
The leave-one-task-out selector-choice diagnostic is worse
(`0.233912` validation mean normalized regret and `0.25` validation
non-inferiority), so this is recorded as a useful diagnostic, not a frozen
method.

Actual covariance-shrinkage controls are implemented in `spectra_cal.py`
through `--covariance-shrinkage-mode fixed` and `pair_consistency`, and
`--output-selector` now prevents those runs from overwriting the default
`spectra_cal.json`. A single CPU smoke test for
`ACMv9_to_Citationv1`/`spectra_cov_gamma000` completed with zero label access
and zero protocol violations. The full four-task `{0, 0.5, 1,
pair_consistency}` sweep remains pending and should be run as a planned
open-development diagnostic before any sealed evaluation.

## Final metrics

| Metric | Value |
|---|---:|
| Mean normalized regret | 0.0141458660 |
| CVaR-20% | 0.0479275556 |
| Worst fold | 0.0838775345 |
| Median Kendall tau | 0.9331080050 |
| Mean Spearman rho | 0.9820032712 |
| Top-weighted Kendall | 0.9946191239 |
| Risk-estimation MAE | 0.0186555482 |
| Mean oracle F1 gap | 0.0088023776 |
| Top-5% hit rate | 0.9545454545 |
| Selection stability | 0.6292613636 |
| Localized-gain share | 0.4465211863 |
| Runtime | 327.394 s |

## Eight-iteration audit

Only two changes were promoted, and both are mathematically exact runtime
optimizations:

1. caching immutable CSR recovery structures;
2. reusing the local augmented recovery workspace between the unchanged
   initial and robust refits.

The fastest accepted run took `320.695s`, down from `451.999s` (`29.050%`).
All 44 selected candidate indices and every non-runtime metric remained exact.

The strongest scientific near-miss was delete-one-donor consensus: mean regret
improved to `0.0139408`, but CVaR worsened from `0.04793` to `0.05159`, so it
was rejected. Low-rank risk-correction transport also failed cross-family
generalization; descriptor/correction-distance Spearman was `-0.04954`.

## Conclusion

The graph spectral risk decomposition is operational, and full covariance
interactions contain selection-relevant information. The remaining bottleneck
is deciding when covariance/risk correction from unlabeled shift descriptors
should be trusted under unseen shifts—not solver speed, more spectral filters,
or a scalar residual/regularization choice.

## Audit

- 17 focused/theory tests passed.
- 53 protected paths were checked with 0 mismatches in the controlled run.
- 44/44 final selections matched the frozen baseline exactly.
- Target-label accesses: 0.
- Protocol violations: 0.
