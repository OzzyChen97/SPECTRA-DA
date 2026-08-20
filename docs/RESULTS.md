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
