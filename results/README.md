# Reported result scopes

The result files deliberately separate protocols that cannot be compared as
if they were one benchmark.

| Artifact | Scope | What it supports |
|---|---|---|
| `gate1_selector_comparison.json` | Four real-target development tasks, 140 frozen candidates per task, identical public inputs for 14 selectors | Fair selector comparison. Transfer Score is currently strongest in top-1 selection. |
| `core_ablation.json` | Four tasks, 44 source-simulated held-out shifts, 675 candidates per task | Component attribution. Covariance correction is the dominant source-simulated gain; the incremental spectral top-1 effect is modest. |
| `final_metrics.json` | Same 44 source-simulated folds for the pre-frozen full selector | Reproduction target for the reported `0.014145866` refinement result. |

The source-simulated `0.014145866` number must not be placed in the same result
column as the real-target Gate-1 numbers: the candidate pools and evaluators
are different. A complete 16-transfer real-target comparison remains pending
an externally operated one-time sealed evaluation.
