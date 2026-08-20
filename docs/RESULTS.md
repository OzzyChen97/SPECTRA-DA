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
| SPECTRA shortlist -> Transfer Score rerank @20% | 0.087507 | 0.223881 | 0.223881 | 0.623427 | 0.750 |

The repaired shortlist/rerank selector is a strong open-development
near-miss: it improves mean normalized regret by `59.64%` relative to Transfer
Score and improves selected Micro-F1 by `0.04776`, but it is not promotion-ready
because only `2/4` tasks are no worse than Transfer Score and oracle
recall@20% is `0.50`, below the registered `0.75` shortlist guardrail. It must
not be treated as the frozen "ours" selector for sealed evaluation.

The leave-one-task-out selector-choice diagnostic reaches the same conclusion:
when each fold selects a configuration using only the other three open-dev
tasks, it repeatedly selects the same SPECTRA-shortlist -> Transfer-Score
rerank variant. Validation mean normalized regret remains `0.087507`, but the
held-out task non-inferiority rate is still `0.50` and oracle recall@20% is
still `0.50`. The gain is therefore real but family-localized rather than a
stable four-task promotion signal.

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
