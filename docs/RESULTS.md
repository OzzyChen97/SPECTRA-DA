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
and zero protocol violations. The complete sweep and its trajectory-balanced
follow-up are reported below.

## Covariance-gamma sweep and no-go result

The planned four-task sweep is now complete. It evaluates fixed
`gamma in {0, .25, .5, .75, 1}`, the descriptor support gate, candidate-level
pair consistency, and a trajectory-balanced pair-consistency control. All
selectors read the same 675-candidate banks, report zero label accesses and
zero protocol violations, and use only the four open-development tasks for
evaluation.

| Covariance rule | Mean NRegret | Worst NRegret | Mean selected Micro-F1 |
|---|---:|---:|---:|
| Fixed gamma=0 | 0.162345 | 0.222591 | 0.594642 |
| Fixed gamma=0.25 | **0.140279** | 0.328358 | 0.591542 |
| Fixed gamma=0.50 | 0.227879 | 0.537313 | 0.558070 |
| Fixed gamma=0.75 | 0.339996 | 0.537313 | 0.495640 |
| Fixed gamma=1 | 0.339996 | 0.537313 | 0.495640 |
| Descriptor support gate | 0.232580 | 0.301272 | 0.544995 |
| Candidate-pair consistency | 0.339996 | 0.537313 | 0.495640 |
| Trajectory-balanced consistency | 0.320986 | 0.537313 | 0.509686 |

The task-wise diagnostic oracle over the frozen fixed-gamma grid reaches mean
NRegret `0.102191`: ACM prefers `0.5`, Citation ties across `0/.25/.5`,
USA prefers `0`, and BRAZIL prefers `.25`. In contrast, candidate-pair
consistency chooses `.75/1/1/.75`; trajectory balancing changes only ACM to
`.5`. Thus pair-sum residual consistency does not predict when transported
covariance improves target model selection. The covariance-gamma route is
therefore stopped as a deployment trust signal. Fixed gamma `.25` is a useful
ablation but fails the worst-task and task-noninferiority guardrails and cannot
be promoted.

The optimized sweep is scientifically identical to the direct selector path:
the cached and uncached gamma-zero outputs match exactly for candidate scores,
point estimates, uncertainty, band risks, selected candidate, and transport
diagnostics. Sharing task-level spectral disagreement and replacing the dense
complete-pair least-squares solve with its analytic projection reduced the
full 28-selector CPU wall time to `449.26` seconds, below the registered
`480`-second diagnostic budget.

## Regret factorization and trajectory-screening control

`selection_error_decomposition.py` exactly separates selected regret into
trajectory and checkpoint components, and separately into method and
within-method components. For the current best Agreement@20% -> Transfer Score
selector:

| Component | Mean normalized gap | Share of total gap |
|---|---:|---:|
| Selected trajectory vs global oracle | 0.083604 | 95.5% |
| Selected checkpoint vs trajectory oracle | 0.003903 | 4.5% |
| Selected method vs global oracle | 0.058063 | 66.4% |
| Within selected method | 0.029444 | 33.6% |

This confirms that its remaining error is overwhelmingly trajectory-level.
However, the first explicit trajectory-screening implementation does not solve
that error. It scores each `(method, config, seed)` trajectory by the mean of
its best three checkpoint ranks, selects the top 20% trajectories, expands
them, and lets Transfer Score choose a checkpoint:

| Trajectory shortlist owner | Mean NRegret | Worst NRegret | Mean selected Micro-F1 |
|---|---:|---:|---:|
| Agreement | 0.247994 | 0.582090 | 0.551518 |
| SPECTRA-Cal | 0.321541 | 0.641791 | 0.502471 |
| Agreement/SPECTRA trajectory midrank | 0.164165 | 0.343284 | 0.594632 |
| Candidate-level Agreement@20% -> TS | **0.087507** | **0.223881** | **0.623427** |

Expanding whole trajectories removes useful checkpoint-level filtering and
allows Transfer Score to select poor checkpoints or trajectories. The q=3
trajectory aggregation is therefore recorded as a failed structural control,
not promoted or tuned further on these four tasks. The next viable direction
is a stability/coverage diagnostic that preserves the successful candidate
shortlist while testing temporal thinning and trajectory coverage.

## Stage-C shortlist diagnosis and structural controls

`shortlist_error_decomposition.py` gives a more precise diagnosis than the
selected-trajectory factorization above. It expands every trajectory or method
represented by the fixed Agreement top-20% candidate shortlist and exactly
decomposes the final regret into shortlist coverage, checkpoint coverage, and
within-shortlist reranking terms:

| Agreement@20% -> TS component | Mean normalized gap | Share |
|---|---:|---:|
| Trajectory coverage | 0.000000 | 0.0% |
| Method coverage | 0.000000 | 0.0% |
| Checkpoint coverage inside represented trajectories | 0.017927 | 20.5% |
| Transfer Score reranking inside the shortlist | 0.069580 | 79.5% |
| Total | 0.087507 | 100.0% |

The shortlist already represents the global-oracle method and trajectory on
all four tasks. The dominant problem is therefore not trajectory coverage but
choosing among the shortlisted checkpoints, especially on
`USA_to_BRAZIL`. This exact report is stored in
`results/gda_select/open_dev/stage_c_shortlist_error_decomposition.json`.

Two registered cross-fitted Agreement controls remove a candidate's entire
trajectory (LOTO) or method (LOMO) before constructing the node-wise majority
reference. Both remain strictly target-label-free, but neither improves the
current candidate-level shortlist:

| Stage-C shortlist -> TS control | Mean NRegret | Worst NRegret | Mean selected Micro-F1 |
|---|---:|---:|---:|
| Agreement@20% -> TS | **0.087507** | **0.223881** | **0.623427** |
| Trajectory-cross-fitted Agreement@20% -> TS | 0.113626 | 0.328358 | 0.610068 |
| Method-cross-fitted Agreement@20% -> TS | 0.141500 | 0.343284 | 0.602002 |

The corresponding four-task Agreement score construction takes `26.71s`
(LOTO) and `24.81s` (LOMO) on CPU. These are diagnostic no-go controls, not
promotion candidates.

Finally, `trajectory_aware_rerank.py` preserves the successful candidate
shortlist but scores each represented trajectory by the mean of its best
`k` Transfer Score ranks, with worst-rank padding for missing checkpoints. It
then selects the best Transfer Score checkpoint within the chosen trajectory.
`k=1` exactly reproduces the candidate-level TS reranker; repeated trajectory
evidence with `k=2` or `k=3` is harmful:

| Trajectory-aware TS evidence | Mean NRegret | Worst NRegret | Mean selected Micro-F1 |
|---|---:|---:|---:|
| k=1 control | **0.087507** | **0.223881** | **0.623427** |
| k=2 | 0.201845 | 0.438538 | 0.592411 |
| k=3 | 0.295129 | 0.597015 | 0.544701 |

The Stage-C evidence rejects both self-vote removal and naive repeated-TS
trajectory evidence as solutions.

### Fixed-budget coverage floors

`coverage_floor_selection.py` keeps the shortlist budget exactly 135 while
reserving Agreement-ranked candidates for every trajectory or method. Unlike
the failed whole-trajectory expansion, it never restores unfiltered
checkpoints:

| Coverage-floor control | Mean NRegret | Worst NRegret | Mean selected Micro-F1 |
|---|---:|---:|---:|
| Every trajectory top-1 + global fill | 0.118994 | 0.328358 | 0.606542 |
| Every trajectory top-2 + global fill | 0.150184 | 0.229236 | 0.605405 |
| Every trajectory top-3 | 0.129347 | 0.223881 | 0.616408 |
| Every method top-27 | 0.127965 | 0.223881 | 0.617316 |
| Candidate Agreement@20% -> TS | **0.087507** | **0.223881** | **0.623427** |

Coverage is not the missing signal: the corrected objective reports oracle
trajectory recall@20% of `1.00` for the current best shortlist. The earlier
`0.50` figure is exact-candidate oracle recall, not trajectory recall.

### Fixed gamma=.25 auxiliary shortlists

The registered auxiliary controls use exact 135-candidate budgets and leave
Transfer Score unchanged:

| Auxiliary shortlist -> TS | Mean NRegret | Worst NRegret | Mean selected Micro-F1 |
|---|---:|---:|---:|
| gamma=.25 top-20% | 0.214562 | 0.567164 | 0.568489 |
| Agreement/gamma=.25 percentile midrank | 0.141917 | 0.343284 | 0.601695 |
| Agreement top-10% union gamma=.25 top-10% | 0.157698 | 0.343284 | 0.597703 |

The union is filled to 135 by the two-signal mean percentile midrank when its
raw union is smaller. No gamma auxiliary control approaches the unmodified
Agreement shortlist, so gamma=.25 provides no deployable incremental benefit.

### Bootstrap stability and router audit

`bootstrap_stability_selection.py` performs 32 deterministic 80%-node
subsamples. Its stable Agreement shortlist reaches `0.113626` mean NRegret and
`0.328358` worst NRegret, worse than the original shortlist. The hard
USA->BRAZIL cell has the lowest Agreement candidate-shortlist Jaccard
(`0.8422`) and trajectory-set Jaccard (`0.7439`).

`bootstrap_transfer_score.py` exactly recomputes the Hopkins and normalized
information-maximization terms on the same subsamples while holding the
classifier-geometry term fixed. Full Transfer Score reconstruction error is
exactly zero for every candidate. With numerical-library threads constrained
to one per 16 candidate workers, all four tasks finish in `156.73s`, below the
480-second budget.

The pre-registered router chooses Agreement@20% -> TS only when its
trajectory-set shortlist Jaccard exceeds Transfer Score's. Transfer Score is
more stable on all four tasks, so the router always falls back to Transfer
Score and selects the non-inferior expert on only `2/4` open-development tasks
under the per-task audit. It therefore fails the required `3/4` qualification
and is rejected before use.

### Stage-C promotion decision

`results/gda_select/open_dev/stage_c_promotion_audit.json` consolidates all
Stage-C controls. The best selector remains Agreement@20% -> Transfer Score:

| Registered promotion check | Evidence | Pass |
|---|---:|:---:|
| Mean NRegret <= 0.10 | 0.087507 | yes |
| Worst NRegret <= 0.20 | 0.223881 | **no** |
| Non-inferior to TS on at least 3/4 tasks | 2/4 | **no** |
| LOTO mean NRegret <= 0.15 | 0.087507 | yes |
| Oracle trajectory recall@20% >= 0.75 | 1.00 | yes |
| Source-sim family-out CVaR degradation <=5% | not evaluated for this selector | **no** |
| Runtime <480s | 449.26s conservative full-sweep evidence | yes |
| Label/protocol counts | 0 / 0 | yes |

The audit decision is:

```text
stop_no_freeze_no_sealed_evaluation
```

No Stage-C selector is frozen, no stability router is promoted, and the final
12 transfer labels remain sealed. Further four-task tuning would be
leaderboard overfitting rather than evidence of cross-family reliability.

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
should be trusted under unseen shifts and how to calibrate the final shortlist
top—not solver speed, more spectral filters, more trajectory coverage, or a
scalar residual/regularization choice.

## Audit

- 17 focused/theory tests passed.
- 53 protected paths were checked with 0 mismatches in the controlled run.
- 44/44 final selections matched the frozen baseline exactly.
- Target-label accesses: 0.
- Protocol violations: 0.
